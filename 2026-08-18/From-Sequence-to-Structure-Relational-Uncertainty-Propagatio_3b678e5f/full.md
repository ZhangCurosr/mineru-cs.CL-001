# From Sequence to Structure: Relational Uncertainty Propagation for LLM Agents

Zhengzhao Ma<sup>1,2</sup>, Boxi Cao<sup>1</sup>, Yaojie Lu<sup>1</sup>, Hongyu Lin<sup>1</sup>, Xianpei Han<sup>1</sup>, Le Sun<sup>1</sup>

<sup>1</sup>Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences <sup>2</sup>University of Chinese Academy of Sciences, Beijing, China

Reliable uncertainty quantification (UQ) is essential for deploying large language model (LLM) agents in complex interactive environments. Existing UQ methods largely rely on local signals, such as token probabilities, predictive entropy, or per-step confidence, and therefore overlook the long-range dependencies through which errors accumulate across an execution trajectory. As a result, they may fail to identify agent failures whose causes originate several reasoning or interaction steps before the final answer. We propose RUPA (Relational Uncertainty Propagation for Agents), a trajectory-level UQ framework for LLM agents. RUPA represents an execution history as a directed trajectory graph in which reasoning states, tool interactions, and environment feedback are nodes connected by temporal and semantic dependency edges. It then propagates uncertainty over this graph to capture how execution risk accumulates and transfers across interaction steps. The propagated signal is combined with trajectory-level behavioral features and goal-alignment information to produce a confidence estimate for the full agent trajectory. We evaluate RUPA on representative agent benchmarks, including τ-2, Terminal-Bench-2, and GAIA, using 6 open-source LLMs spanning multiple model families. Experimental results show that RUPA consistently outperforms existing UQ methods by providing more accurate uncertainty estimates, enabling earlier failure detection, and improving uncertainty-guided agent execution across diverse agent tasks. These results demonstrate that explicitly modeling relational dependency is crucial to reliable UQ for long-horizon LLM agents, providing a practical foundation for trustworthy agent execution.

<sup>#</sup> Email {mazhengzhao2024,caoboxi}@iscas.ac.cn § Code https://github.com/icip-cas/RUPA

![](images/3ef9c63f611e3a1fc0173681394c3cc957a09702b5d1eb36593a290e0c9f6309.jpg)

## 1 Introduction

Large Language Models (LLMs) are rapidly evolving from passive generators into autonomous agents capable of pursuing complex objectives through multi-step reasoning, tool use, and interaction with external environments (Yao et al., 2022; Schick et al., 2023; Liu et al., 2024). Such LLM agents have demonstrated remarkable capabilities in software engineering (Jimenez et al., 2024), web automation (Zhou et al., 2024), scientific discovery (Zhou et al., 2023), and complex decision-making tasks (Mialon et al., 2024). As these systems increasingly perform long-horizon tasks involving tens or even hundreds of reasoning, action, and interaction steps, their reliability has become a critical barrier to real-world deployment (Han et al., 2024; Chen et al., 2025). Unlike traditional text generation, failures in LLM agents rarely originate from a single erroneous prediction. Instead, they emerge from the accumulation and propagation of errors across interdependent reasoning steps, tool executions, and environment interactions. Consequently, accurately estimating execution risk before failures occur has become a fundamental challenge for building reliable autonomous agents (Gawlikowski et al., 2023; Yin et al., 2024).

Uncertainty quantification (UQ) provides a natural solution to this challenge by estimating the reliability of model behavior and enabling downstream strategies such as risk detection, adaptive resampling, self-correction, and decision optimization (Jiang et al., 2021; Kadavath et al., 2022; Lin et al., 2022). However, existing UQ methods are primarily designed for isolated predictions or short-context generation and therefore struggle in long-horizon agent execution. Traditional approaches estimate uncertainty solely from the current model output while largely ignoring risks accumulated throughout the execution history (Manakul et al., 2023;

![](images/dafd303901ff29e8685c9e5710401fe3e050cbdb9bfbcf7814a77e06bfc388af.jpg)  
Figure 1 Overview of RUPA. The agent trajectory is converted into a directed dependency graph, uncertainty is propagated over historical states, and the propagated risk is combined with local uncertainty to estimate trajectory-level uncertainty.

Farquhar et al., 2024; Mao and Venkat, 2026). More recent agent-oriented UQ methods (Han et al., 2024; Zhao et al., 2025; Kirchhof et al., 2025) begin to incorporate historical information, but they typically model agent trajectories as linear sequences and aggregate uncertainty according to temporal distance or semantic similarity. In reality, dependencies between agent steps are inherently relational rather than purely sequential (Jelodar et al., 2026; Zhang et al., 2026a; Sun et al., 2026). An early mistake may have little immediate impact unless it influences subsequent reasoning or tool use, whereas a seemingly small misunderstanding can gradually evolve into catastrophic failure if it continuously afects later decisions. Without explicitly modeling these dependency structures, uncertainty estimators cannot accurately characterize how execution risks evolve throughout an agent trajectory.

In this work, we argue that uncertainty in LLM agents should be viewed as a trajectory-level property that evolves over the relational structure of agent execution, rather than as a sequence of independent confidence estimates. The key challenge is therefore not simply estimating the uncertainty of each individual step, but understanding how uncertainty propagates through dependencies among reasoning states, actions, tool invocations, and environment observations.

Motivated by this observation, we propose Relational Uncertainty Propagation for Agents (RUPA), a trajectory-level uncertainty quantification framework for autonomous LLM agents. As shown in Fig 1, rather than representing agent execution as a linear sequence, RUPA automatically models the execution trajectory as a directed relational graph, in which nodes represent diferent execution events, including reasoning states, tool invocations, user interactions, and environment observations. Edges further capture the dependency relations among these nodes, such as sequential transitions, repeated behaviors, reasoning continuation, feedback dependencies, and goal alignment. Based on this structured representation, the uncertainty of each node is jointly determined by its local uncertainty and the historical uncertainty propagated through the directed graph. Specifically, relation-aware edge weights are automatically determined according to the statistical importance of diferent dependency types, allowing uncertainty to propagate preferentially along influential execution paths. The propagated historical uncertainty is then integrated with the node’s local uncertainty to estimate its execution risk. Consequently, RUPA captures the realistic evolution of uncertainty during agent execution, where early critical mistakes continue to influence subsequent dependent reasoning and actions, while uncertainty originating from unrelated execution branches is naturally suppressed.

We evaluate RUPA on 3 representative agent benchmarks covering diverse LLM agent scenarios, including τ-2 (Barres et al., 2025), Terminal-Bench-2 (Merrill et al., 2026), and GAIA (Mialon et al., 2024), using 6 open-source LLMs ranging from 26B to 230B parameters. Extensive experiments across multiple evaluation settings consistently demonstrate that, compared with existing UQ methods, RUPA substantially improves uncertainty quantification quality, enabling more efective intervention for high-risk agent execution and stronger downstream task performance. First, RUPA achieves the best uncertainty estimation quality on all benchmarks and model families, improving the average AUROC of MiniMax-M2.7 from 0.694 to 0.718 over the strongest baseline. Second, in prefix-based evaluation, RUPA identifies potential execution failures significantly earlier than existing methods, demonstrating a superior early-risk detection capability. Third, RUPA consistently improves downstream task success rates by selecting lower-risk candidate actions during multi-sample decoding. Extensive ablation studies further show that these improvements primarily originate from relation-aware trajectory graph modeling and uncertainty propagation, highlighting the importance of modeling structural dependencies for reliable uncertainty estimation in autonomous LLM agents.

The main contributions of this work are summarized as<sup>1</sup>:

• We identify relational dependencies between execution steps as a key source of uncertainty evolution in LLM agents, revealing the limitations of existing uncertainty quantification methods that model agent trajectories as independent predictions or linear sequences.

• We propose RUPA, a graph-based uncertainty quantification framework that represents agent execution as a directed relational graph and performs relation-aware uncertainty propagation to estimate execution risk.

• Extensive experiments on 3 representative agent benchmarks and 6 LLMs demonstrate that RUPA consistently improves uncertainty estimation quality, early failure detection, and uncertainty-guided agent execution over existing uncertainty quantification methods.

## 2 Related Works

## 2.1 Uncertainty Quantification for LLM

Uncertainty quantification (UQ) has become a fundamental component of trustworthy LLMs, aiming to estimate the reliability of model predictions and support downstream tasks (Jiang et al., 2021; Kadavath et al., 2022; Yan et al., 2026). Existing UQ methods for LLMs can be categorized into probability-based, verbalized and sampling-based methods (Yin et al., 2024; Heo et al., 2024).

Probability-based approaches (Kossen et al., 2024) estimate uncertainty directly from model outputs like predictive entropy, sequence generation probability, or related confidence scores derived from the model’s output distribution (Moskvoretskii et al., 2025; Li et al., 2025). Verbalized methods prompt models to explicitly output a confidence score alongside the answer (Lin et al., 2022; Xiong et al., 2023; Yang et al., 2024), ofering a flexible and human-interpretable interface (Yoon et al., 2025). Sampling-based methods (Manakul et al., 2023; Farquhar et al., 2024) estimate uncertainty by generating multiple candidate responses and measuring their consistency. They evaluate the agreement among sampled generations to detect hallucinations and factual inconsistencies.

These methods have demonstrated strong performance (Ding et al., 2025; Damani et al., 2025; Ma et al., 2026), but they are primarily designed for single-turn prediction. Consequently, they cannot efectively characterize uncertainty across long-horizon reasoning and interaction trajectories (Oh et al., 2026).

## 2.2 Uncertainty Quantification for LLM Agents

Agent uncertainty quantification aims to estimate the probability that an entire execution trajectory will successfully accomplish the target task (Kirchhof et al., 2025; Zhang et al., 2026b). Several recent methods extend traditional uncertainty estimation from individual responses to complete agent trajectories, including SAUP (Zhao et al., 2025), Tracer (Tayebati et al., 2026), and UProp (Duan et al., 2025). These methods improve uncertainty estimation compared with conventional approaches and demonstrate the importance of utilizing execution history (Shi et al., 2026).

<table><tr><td>Domain</td><td>Random</td><td>Seq. Prob.</td><td>Verbalize</td></tr><tr><td>Airline</td><td>0.441</td><td>0.205</td><td>0.485</td></tr><tr><td>Retail</td><td>0.472</td><td>0.301</td><td>0.523</td></tr></table>

Table 1 Preliminary AUROC evaluation of traditional UQ methods on $\tau ^ { 2 }$ agent tasks.

![](images/e78fbfab0582c666c62658f67f296b3fc38bd2be0710be3cd48b18261b12cf0b.jpg)

![](images/6620e957b3c411b9320a9606762ed6e317f0b75c55e0cbc34faca24146f559f8.jpg)  
Figure 2 Structural analysis of failure-indicative signals in failed agent trajectories.

However, existing agent UQ methods predominantly represent execution trajectories as linear sequences. As a result, the underlying dependency structure among reasoning steps remains largely unexplored. Ignoring these relational dependencies makes it dificult to accurately capture how execution risks accumulate and propagate throughout long-horizon agent trajectories (Li and Cao, 2026).

## 3 Empirical Analysis of Agent Uncertainty

We first investigate why uncertainty estimation for agents is fundamentally more challenging than conventional LLM tasks. Through empirical analysis, we show that execution failures are often induced by relational dependencies distributed across the entire trajectory. These observations motivate trajectory-level relational uncertainty modeling for reliable agent uncertainty quantification.

Traditional uncertainty estimation fails on long-horizon agent tasks. We first evaluate whether UQ methods designed for conventional language generation remain efective in long-horizon agent tasks. Specifically, we conduct a preliminary study on representative τ -2 domains, Airline and Retail, using Qwen3.5- 27B model. We compare two widely adopted UQ methods, sequence probability and verbalized confidence, for trajectory-level failure prediction.

As shown in Table 1, both methods perform poorly, with AUROC values close to random guessing across the two domains. For example, on the Airline domain, sequence probability obtains an AUROC of only 0.205, while verbalized confidence reaches 0.485. Similar observations hold for the Retail domain. These results indicate that uncertainty estimated solely from local generation confidence is insuficient for identifying failures in long-horizon agent execution. This suggests that execution failures depend on information beyond the current generation, motivating a closer examination of where failure-indicative uncertainty originates.

Failure signals are distributed over relational trajectory dependencies. To better understand why traditional uncertainty estimation fails, we analyze where failure-indicative signals emerge within agent trajectories. For each failed trajectory, we compute a step-level risk score and identify the execution step with the highest estimated anomaly. Then we analyze its relative position and dependency relation profile.

As shown in Fig 2, high-risk steps are distributed throughout the execution trajectory rather than concentrating near the final answer, which suggests that failures often originate from intermediate reasoning or interaction steps and gradually propagate to subsequent decisions. Furthermore, the failure step exhibits an average repetition score of 0.981 and a stagnation score of 0.883, while feedback-conflict and correction/retry relations also appear frequently. These patterns indicate that execution failures are associated with relational, structural dependencies among trajectories.

Overall, these observations reveal that uncertainty in agent execution is inherently trajectory-dependent. Consequently, modeling execution trajectories as linear sequences is insuficient to accurately characterize risk evolution. This empirical evidence motivates the graph-based uncertainty propagation framework proposed in the following section.

## 4 Methods

Motivated by the above insight, we propose Relational Uncertainty Propagation for Agents (RUPA), a relational trajectory-aware UQ framework for LLM agents. Fig 1 presents an overview of RUPA. Given an agent execution trajectory, RUPA converts the execution trajectory into a relational dependency graph. Then it propagates uncertainty through the edges of the graph to capture how execution risks accumulate across reasoning, tool use, and environment interaction. Finally, the propagated structural uncertainty is combined with the local uncertainty of the current reasoning step to produce a trajectory-aware uncertainty estimate.

## 4.1 Relational Trajectory Graph Construction

To capture the relational trajectory-dependencies in agent uncertainty quantification, RUPA represents each execution prefix as a directed trajectory graph $\mathcal { G } = ( \nu , \mathcal { E } )$ . Each node i ∈ V corresponds to an execution event, including user instructions, assistant reasoning or actions, tool invocations, and environment observations. Directed edges $e _ { i , j } \in \mathcal { E }$ describe logical dependencies between historical events and the current reasoning step. RUPA constructs only dependency-related edges within a bounded historical context. We consider seven representative relation types:

• Sequential: the immediately preceding execution.

• Latest: the most recent environment or user instructions.

• Repetition: actions exhibiting highly similar reasoning patterns or tool usage.

• Progression: reasoning steps extending an existing solution step.

• Parallel: alternative reasoning branches under the same task context.

• Feedback: feedback to environment observations indicating execution failures or unstable outputs.

• Goal Alignment: semantic dependency between the current reasoning step and the original task objective.

The type of the edges is determined by features like embedding distance and matching cues. Together, these relations characterize how execution states interact throughout an agent trajectory, providing a structured representation for uncertainty propagation beyond simple temporal ordering. The detailed edge construction is shown in the Appendix.

## 4.2 Relation-aware Uncertainty Propagation

Given the trajectory graph G, RUPA propagates uncertainty from historical execution states to the current node. Unlike conventional sequential aggregation, RUPA uncertainty propagation is guided by dependency relations in the graph.

In particular, each node holds a local uncertainty $U _ { t }$ . For assistant nodes, local uncertainty is computed from predictive entropy. For environment nodes, uncertainty is estimated from observable interaction signals, including execution failures, empty tool responses, and conflicting environment feedback. These uncertainty values serve as the initial risk associated with each graph node.

Since diferent dependency relations should contribute unequally to possible future failures, RUPA learns a propagation weight for each graph edge, where relation types exhibiting stronger structural variation across trajectories receive larger propagation coeficients. The propagation weight of each edge is then determined by its relation reliability, relation strength, and temporal distance:

$$
w _ { i t } = \rho _ { \tau _ { i t } } \tilde { r } _ { i t } \delta ^ { \mathrm { a g e } ( i , t ) - 1 } ,\tag{1}
$$

where $\delta$ is the temporal decay factor. Consequently, structurally important historical dependencies exert greater influence, while obsolete execution states are gradually discounted. Specially, for goal-alignment edge, instead of computing an edge weight, we estimate a goal alignment score:

$$
Q _ { i t } = 1 - S ( y _ { t } , x )\tag{2}
$$

where $S ( \boldsymbol { y } _ { t } , \boldsymbol { x } )$ is a similarity function.

Finally, RUPA propagates uncertainty over the trajectory graph to estimate the structural risk of the current reasoning step. The propagated uncertainty is computed by aggregating uncertainty from all dependencyrelated historical nodes,

$$
G _ { t } = \frac { \sum _ { i \in \mathcal { N } ( t ) } w _ { i t } ( P _ { i } + Q _ { i t } ) } { \sum _ { i \in \mathcal { N } ( t ) } w _ { i t } + \epsilon } ,\tag{3}
$$

where $P _ { i }$ denotes the propagated uncertainty stored at historical node i, and $\mathcal { N } ( t )$ denotes the neighboring historical nodes connected to the current step. Besides graph propagation, RUPA also maintains an exponentially decayed uncertainty momentum to preserve long-range execution trends,

$$
m _ { t } = \frac { \sum _ { k < t } \gamma ^ { t - k } P _ { k } } { \sum _ { k < t } \gamma ^ { t - k } + \epsilon } , \qquad H _ { t } = \eta _ { g } G _ { t } + \eta _ { m } m _ { t } ,\tag{4}
$$

where $H _ { t }$ denotes the propagated historical uncertainty. This formulation allows uncertainty to accumulate along dependency paths throughout the trajectory, enabling early execution failures to continuously influence subsequent reasoning even when the current generation itself appears confident.

RUPA combines the intrinsic uncertainty of the current generation $U _ { t }$ and the structural uncertainty accumulated throughout the history $H _ { t }$ by a simple additive formulation:

$$
R _ { t } = \lambda _ { u } U _ { t } + \lambda _ { h } H _ { t } ,\tag{5}
$$

The resulting score $R _ { t }$ serves as the uncertainty estimate of the current reasoning step and is propagated to subsequent execution states. For complete trajectories, RUPA aggregates the step-level uncertainty scores to obtain a trajectory-level uncertainty estimate, where larger scores indicate a higher probability of task failure.

## 5 Experiments

In this section, we evaluate RUPA on multi-turn agent tasks. The experiments assess both uncertainty estimation quality and the efect of uncertainty estimates on downstream agent execution.

## 5.1 Experimental Settings

Datasets. We evaluate all methods on 3 representative agent benchmarks: τ-2, Terminal-Bench-2, and GAIA, covering conversational decision making, terminal-based software engineering, and open-domain complex problem solving. Together, they provide a comprehensive evaluation of UQ methods under diferent reasoning and interaction patterns.

Baselines. We compare RUPA against five representative uncertainty quantification methods. PE estimates uncertainty using predictive entropy aggregated over the trajectory. $\mathbf { S P }$ uses sequence generation probability as a confidence signal. SAUP estimates uncertainty from scene-aware execution risks. Tracer performs trajectory-level UQ by aggregating interaction signals across the execution process. UProp models uncertainty propagation using pointwise mutual information. Together, these baselines cover both conventional token-level uncertainty estimation and recent trajectory-aware agent uncertainty quantification methods. The quality of each UQ method is evaluated with AUROC, AUPRC, and the best F1 score over all decision thresholds.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="3">Average</td><td colspan="3">T-2</td><td colspan="3">Terminal-Bench</td><td colspan="3">GAIA</td></tr><tr><td>AUROC</td><td>AUPRC</td><td>F1</td><td>AUROC</td><td>AUPRC</td><td>F1</td><td>AUROC</td><td>AUPRC</td><td>F1</td><td>AUROC</td><td>AUPRC</td><td>F1</td></tr><tr><td rowspan="6">Qwen3.5-27B</td><td>Entropy Seq-prob SAUP</td><td>0.559 0.566</td><td>0.543 0.612</td><td>0.538 0.582</td><td>0.573 0.547</td><td>0.623 0.636</td><td>0.438 0.509</td><td>0.508 0.541</td><td>0.679 0.682</td><td>0.705 0.667</td><td>0.595 0.611</td><td>0.326 0.517</td><td>0.470 0.571</td></tr><tr><td></td><td>0.595</td><td></td><td>0.630</td><td></td><td></td><td></td><td></td><td></td><td>0.763</td><td>0.665</td><td>0.390</td><td>0.533</td></tr><tr><td></td><td>0.608</td><td>0.561</td><td></td><td>0.630</td><td>0.681</td><td>0.595</td><td>0.491</td><td>0.611</td><td></td><td>0.680</td><td>0.394</td><td>0.545</td></tr><tr><td>Tracer</td><td></td><td>0.559</td><td>0.641</td><td>0.634</td><td>0.678</td><td>0.649</td><td>0.511</td><td>0.605</td><td>0.730</td><td></td><td></td><td></td></tr><tr><td>Uprop</td><td>0.588</td><td>0.623</td><td>0.645</td><td>0.628</td><td>0.691</td><td>0.677</td><td>0.530</td><td>0.668</td><td>0.671</td><td>0.606</td><td>0.511</td><td>0.587</td></tr><tr><td>RUPA</td><td>0.656</td><td>0.746</td><td>0.687</td><td>0.677</td><td>0.733</td><td>0.714</td><td>0.594</td><td>0.725</td><td>0.736</td><td>0.697</td><td>0.781</td><td>0.611</td></tr><tr><td rowspan="6">Qwen3.6-35B</td><td>Entropy Seq-prob</td><td>0.579 0.596</td><td>0.627 0.583</td><td>0.583 0.616</td><td>0.631 0.644</td><td>0.717 0.709</td><td>0.706 0.694</td><td>0.569 0.610</td><td>0.671 0.672</td><td>0.622 0.693</td><td>0.538 0.534</td><td>0.493 0.367</td><td>0.421 0.462</td></tr><tr><td></td><td>0.608</td><td>0.644</td><td></td><td></td><td></td><td></td><td></td><td>0.713</td><td></td><td>0.545</td><td>0.487</td><td>0.493</td></tr><tr><td>SAUP</td><td>0.629</td><td>0.663</td><td>0.673</td><td>0.651</td><td>0.733 0.727</td><td>0.735</td><td>0.628 0.649</td><td>0.695</td><td>0.790 0.794</td><td>0.590</td><td>0.566</td><td>0.500</td></tr><tr><td>Tracer</td><td>0.595</td><td></td><td>0.677</td><td>0.647</td><td></td><td>0.736</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Uprop</td><td>0.645</td><td>0.595</td><td>0.661</td><td>0.612</td><td>0.711</td><td>0.720</td><td>0.651</td><td>0.714</td><td>0.795</td><td>0.523</td><td>0.360</td><td>0.467</td></tr><tr><td>RUPA</td><td></td><td>0.679</td><td>0.694</td><td>0.670</td><td>0.741</td><td>0.759</td><td>0.657</td><td>0.724</td><td>0.805</td><td>0.608</td><td>0.571</td><td>0.518</td></tr><tr><td rowspan="6">Gemma4-26B</td><td>Entropy</td><td>0.711 0.712</td><td>0.779 0.838</td><td>0.716 0.759</td><td>0.712 0.701</td><td>0.784</td><td>0.776</td><td>0.643</td><td>0.701</td><td>0.646</td><td>0.779 0.854</td><td>0.851 0.934</td><td>0.726 0.833</td></tr><tr><td>Seq-prob</td><td>0.761</td><td></td><td></td><td></td><td>0.810</td><td>0.761</td><td>0.580</td><td>0.769</td><td>0.683</td><td>0.874</td><td>0.930</td><td>0.839</td></tr><tr><td>SAUP</td><td>0.743</td><td>0.820</td><td>0.775</td><td>0.731</td><td>0.815</td><td>0.814</td><td>0.677</td><td>0.715</td><td>0.671</td><td>0.841</td><td>0.929</td><td>0.729</td></tr><tr><td>Tracer</td><td>0.721</td><td>0.819</td><td>0.732</td><td>0.720</td><td>0.797</td><td>0.800</td><td>0.669</td><td>0.730</td><td>0.668</td><td></td><td>0.932</td><td></td></tr><tr><td>Uprop</td><td>0.780</td><td>0.835</td><td>0.758</td><td>0.718</td><td>0.804</td><td>0.792</td><td>0.585</td><td>0.768</td><td>0.649</td><td>0.861</td><td></td><td>0.833</td></tr><tr><td>RUPA</td><td>0.806</td><td>0.851</td><td>0.784</td><td>0.756</td><td>0.823</td><td>0.819</td><td>0.713</td><td>0.781</td><td>0.690</td><td>0.872</td><td>0.949</td><td>0.842</td></tr><tr><td rowspan="6">Gemma4-31B</td><td>Entropy Seq-prob</td><td>0.818</td><td>0.919 0.910</td><td>0.782 0.806</td><td>0.815 0.827</td><td>0.887 0.870</td><td>0.772 0.764</td><td>0.728 0.740</td><td>0.916 0.915</td><td>0.714 0.783</td><td>0.875 0.886</td><td>0.954 0.944</td><td>0.861 0.871</td></tr><tr><td>SAUP</td><td>0.842</td><td>0.935</td><td>0.844</td><td>0.833</td><td>0.905</td><td>0.828</td><td>0.802</td><td>0.936</td><td>0.826</td><td>0.890</td><td>0.963</td><td>0.879</td></tr><tr><td></td><td>0.823</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.900</td><td>0.965</td><td>0.855</td></tr><tr><td>Tracer</td><td>0.836</td><td>0.920 0.918</td><td>0.777 0.842</td><td>0.763 0.887</td><td>0.858 0.897</td><td>0.741</td><td>0.807 0.751</td><td>0.937 0.918</td><td>0.735 0.832</td><td>0.869</td><td>0.940</td><td></td></tr><tr><td>Uprop RUPA</td><td>0.861</td><td></td><td></td><td></td><td></td><td>0.841</td><td></td><td></td><td></td><td></td><td></td><td>0.854</td></tr><tr><td></td><td>0.492</td><td>0.949</td><td>0.882</td><td>0.877</td><td>0.936</td><td>0.914</td><td>0.815</td><td>0.941</td><td>0.838</td><td>0.892</td><td>0.971</td><td>0.895</td></tr><tr><td rowspan="6">GPT-oss-120B</td><td>Entropy Seq-prob</td><td>0.464</td><td>0.819 0.843</td><td>0.726 0.427</td><td>0.491 0.488</td><td>0.751 0.792</td><td>0.646 0.417</td><td>0.495 0.299</td><td>0.904 0.867</td><td>0.737 0.121</td><td>0.489 0.605</td><td>0.804 0.870</td><td>0.794 0.742</td></tr><tr><td>SAUP</td><td>0.520</td><td>0.870</td><td>0.765</td><td>0.512</td><td>0.869</td><td>0.732</td><td>0.486</td><td>0.902</td><td>0.801</td><td>0.563</td><td>0.840</td><td>0.761</td></tr><tr><td></td><td>0.525</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.228</td><td>0.629</td><td>0.867</td><td>0.494</td></tr><tr><td>Tracer Uprop</td><td>0.567</td><td>0.875 0.858</td><td>0.393 0.504</td><td>0.528 0.716</td><td>0.860 0.831</td><td>0.457 0.689</td><td>0.419 0.343</td><td>0.898 0.871</td><td>0.211</td><td>0.642</td><td></td><td></td></tr><tr><td>RUPA</td><td>0.577</td><td>0.889</td><td>0.815</td><td>0.577</td><td>0.884</td><td>0.762</td><td>0.500</td><td>0.907</td><td>0.902</td><td>0.655</td><td>0.872 0.875</td><td>0.612 0.781</td></tr><tr><td>Entropy</td><td>0.666</td></table>

Table 2 Uncertainty quantification performance across agent benchmarks. The best result for each model is shown in bold.

Models. Experiments are conducted using 6 representative open-source LLMs spanning multiple model families and scales: Qwen3.5-27B, Qwen3.6-35B-A3B, Gemma-4-26B-it, Gemma-4-31B-it, GPT-OSS-120B, and MiniMax-M2.7. These models provide complete reasoning traces and token-level probabilities required by all UQ methods.

## 5.2 Overall Performance

Table 2 summarizes uncertainty quantification performance across agent tasks and model families. Overall, our proposed RUPA consistently achieves the best uncertainty estimation performance across diferent model families, demonstrating that explicitly modeling trajectory-level dependency and uncertainty propagation is an efective solution for failure detection in long-horizon agent execution.

![](images/eab8ffcf886170be1d6cbd0160fb971688afeb1c3097c75a4b3edbb98cee438c.jpg)

![](images/d9c27b9514b69d49413b19b15cf54df366c2d6daa0eeceb626fc6e6f018ec031.jpg)

![](images/618f45a70769334149e67e0f4a37af580993a368bdb2c6450c3d82e17d26f5e8.jpg)

![](images/5c9cfbb8dd71f23e9d7c9572521d0101654a692bcc70d748b2c8cf0fd28029dc.jpg)

![](images/078f64dc930c7959a8d33d96a5fbbdecaa27ed28bccb141f971bb8d6eaf0637c.jpg)

![](images/aa72c6d241959a746315c6dd3268aa545a5b8536da388a4bcaf5cb8822c57c02.jpg)

![](images/b60c467c03ce6728c5f266f4225b016d6a06e18db39b86b4f5e1431f945f7adb.jpg)

![](images/f7913f46d907ec655c71eead3a6a6bf823831f7be355c25e19d64aae6a04a30f.jpg)  
Figure 3 Prefix-based early failure detection on GAIA and Terminal-Bench with MiniMax-M2.7. Curves report AUROC and AUPRC when each method observes only a fixed percentage or a fixed number of steps from the trajectory prefix.

Traditional UQ methods struggle in long-horizon, interactive agent environments. Traditional uncertainty quantification methods, including entropy and sequence probability, consistently underperform across nearly all settings. In particular, on Qwen3.5-27B, Entropy achieves only 0.559 AUROC on average, compared with 0.656 achieved by RUPA. A similar trend can be observed on GPT-OSS-120B, where Entropy obtains only 0.492 AUROC while RUPA improves it to 0.577. These results demonstrate that traditional UQ methods only exploit the confidence of the current generation and ignore dependencies introduced by previous reasoning and environment interactions, making them inadequate for multi-turn agent trajectories.

Sequential agent UQ methods can not fully detect risk propagation in complex step relations. As shown in Table 2, recent agent-oriented uncertainty quantification methods improve over traditional confidence-based approaches by incorporating sequential execution information while they fail to capture relational structure based risk propagation. For instance, on Qwen3.6-35B, Tracer achieves 0.629 average AUROC, while RUPA further improves it to 0.645. Similarly, on Gemma4-26B, the SAUP baseline reaches 0.761 AUROC, whereas RUPA increases this score to 0.780. The improvement becomes even more evident on τ-2 and Terminal-Bench, where modeling relation-aware uncertainty propagation enables more accurate identification of failures accumulated over long execution trajectories.

RUPA achieves a favorable performance in agent trajectory failure detection. Across all six evaluated models, RUPA achieves the highest average AUROC, AUPRC, and F1 score while consistently outperforming previous agent UQ approaches on individual benchmarks. In particular, RUPA improves the average AUROC from 0.608 to 0.656 on Qwen3.5-27B, from 0.629 to 0.645 on Qwen3.6-35B, from 0.761 to 0.780 on Gemma4-26B, from 0.842 to 0.861 on Gemma4-31B, and from 0.694 to 0.718 on MiniMax-M2.7. These results demonstrate the efectiveness and strong generalization ability of trajectory graph modeling and relational uncertainty propagation for agent uncertainty estimation.

## 5.3 Detailed Analysis

RUPA enables earlier failure detection with partial trajectories. A practical UQ method should identify failure risks before the agent completes its entire execution process. To evaluate this capability, we conduct a prefix-based analysis in which each trajectory is truncated by retaining a fixed percentage or number of reasoning/action steps. Uncertainty estimation is then performed using only the partial trajectory available. Fig 3 demonstrates that RUPA consistently achieves higher uncertainty prediction performance. In particular, the advantage is particularly pronounced when only a small or moderate fraction of the trajectory is observed, which suggests that the propagated uncertainty signals emerge early during agent execution and can be leveraged to anticipate future failures before execution finished.

<table><tr><td>Model</td><td>Method</td><td>TB2</td><td>GAIA</td></tr><tr><td rowspan="5">Qwen3.5-27B</td><td>Random</td><td>0.105</td><td>0.261</td></tr><tr><td>Entropy</td><td>0.141</td><td>0.282</td></tr><tr><td>SAUP</td><td>0.150</td><td>0.273</td></tr><tr><td>Tracer</td><td>0.143</td><td>0.267</td></tr><tr><td>Ours</td><td>0.213</td><td>0.297</td></tr><tr><td rowspan="5">Gemma4-31B</td><td>Random</td><td>0.122</td><td>0.175</td></tr><tr><td>Entropy</td><td>0.143</td><td>0.206</td></tr><tr><td>SAUP</td><td>0.146</td><td>0.221</td></tr><tr><td>Tracer</td><td>0.149</td><td>0.229</td></tr><tr><td>Ours</td><td>0.213</td><td>0.242</td></tr><tr><td rowspan="5">GPT-OSS-120B</td><td>Random</td><td>0.079</td><td>0.127</td></tr><tr><td>Entropy</td><td>0.112</td><td>0.140</td></tr><tr><td>SAUP</td><td>0.124</td><td>0.157</td></tr><tr><td>Tracer</td><td>0.167</td><td>0.188</td></tr><tr><td>Ours</td><td>0.202</td><td>0.242</td></tr><tr><td rowspan="5">MiniMax-M2.7</td><td>Random</td><td>0.225</td><td></td></tr><tr><td>Entropy</td><td></td><td>0.284</td></tr><tr><td>SAUP</td><td>0.240 0.247</td><td>0.299 0.285</td></tr><tr><td>Tracer</td><td>0.259</td><td>0.316</td></tr><tr><td>Ours</td><td>0.270</td><td>0.339</td></tr></table>

Table 3 Uncertainty-guided sampling agent performance.  
![](images/c844f8b3d92b2cbb0aa7a6ad4765aa6f66e5691d8ee3eb01d4624707986d45af.jpg)

![](images/8a9b807fffe9ca03ee7b834733d9761e44e40e4199d0c806e0fb1665116de37d.jpg)  
Figure 4 Entropy-matched analysis of trajectory-graph confidence signals.

RUPA’s uncertainty is able to translate into better agent performance. To investigate whether improved uncertainty estimation can benefit agent execution, we built a trivial uncertainty-guided agent framework. At each decision step, the agent samples multiple candidate actions and selects the action associated with the lowest predicted uncertainty score. We evaluate this strategy on Terminal-Bench-2 and GAIA. Table 3 shows the final task accuracy, showing that RUPA consistently achieves the strongest downstream performance across all evaluated models and benchmarks. In particular, on Terminal-Bench-2, the accuracy of Qwen3.5-27B improves from 0.105 under random selection to 0.213 when guided by RUPA. Comparable gains are observed on GAIA, where RUPA consistently outperforms entropy-based and agentspecific uncertainty baselines. These results demonstrate that uncertainty estimates produced by RUPA are not only more accurate for failure detection but also actionable for improving agent decision making.

Graph-based trajectory modeling provides complementary uncertainty signals. To understand whether graph-based trajectory modeling of RUPA provides additional uncertainty information than traditional confidence estimation alone, we compare uncertainty prediction performance under entropy-controlled settings. Specifically, we divide trajectories into bins with similar entropy-based uncertainty scores on GAIA trajectories generated by MiniMax-M2.7 and evaluate whether graph propagation can still distinguish successful and failed executions. Fig 4 shows that graph-based uncertainty remains highly informative even when entropy values are nearly identical. In low-entropy regions (e.g., Q1), conventional probability-based uncertainty estimators achieve AUROC performance of about 0.5 because local token confidence is highly similar across trajectories. In contrast, RUPA still achieves an AUROC of approximately 0.85 and substantially improves AUPRC to around 0.93. These results indicate that problematic dependency structures are suficient in agent uncertainty propagation. Therefore, explicitly modeling trajectory relations provides complementary information beyond token-level confidence signals.

<table><tr><td>Method</td><td>AUROC</td><td>AUPRC</td><td>F1</td></tr><tr><td>Full RUPA</td><td>0.718</td><td>0.805</td><td>0.755</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  graph modeling</td><td>0.678</td><td>0.642</td><td>0.675</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  propagation</td><td>0.689</td><td>0.657</td><td>0.694</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  50% graph edges</td><td>0.705</td><td>0.654</td><td>0.739</td></tr><tr><td>w/ random graph</td><td>0.681</td><td>0.643</td><td>0.691</td></tr></table>

Table 4 Ablation study of RUPA on 3 agent task benchmarks.

## 5.4 Ablation Study

To understand the contribution of each component in RUPA, we conduct ablation studies on the MiniMax-M2.7 model. Table 4 reports the uncertainty estimation performance under diferent ablation settings.

Table 4 shows that both relational graph modeling and uncertainty propagation contribute substantially to the performance of RUPA. In particular, removing graph modeling leads to the largest performance degradation, reducing AUROC from 0.718 to 0.678 and AUPRC from 0.805 to 0.642, which demonstrates that representing agent trajectories as dependency graphs is essential for capturing failure-indicative structural information beyond local uncertainty estimates. When uncertainty propagation is further removed, performance also drops noticeably, demonstrating that explicitly propagating uncertainty across related execution states is critical for modeling long-range failure accumulation.

Furthermore, replacing the relational graph with a random topology results in a similar performance degradation to removing graph modeling, indicating that the performance gains arise from meaningful dependency structures rather than simply introducing additional graph features.

## 6 Conclusion

This paper studies uncertainty quantification for long-horizon LLM agents. We argue that uncertainty in agent execution arises from the dependency structure among reasoning steps, tool interactions, and environment feedback, rather than from isolated model predictions.

Motivated by this observation, we propose RUPA, a relation-aware uncertainty quantification framework that represents agent trajectories as dependency graphs and propagates uncertainty along meaningful execution relations to capture the accumulation of failure risks. Extensive experiments demonstrate that RUPA consistently outperforms both conventional uncertainty quantification methods and recent agent-specific baselines. RUPA also enables earlier failure detection and consistently improves downstream uncertaintyguided agent performance, highlighting the practical value of trajectory-aware uncertainty estimation for reliable autonomous agents.

## References

Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ<sup>2</sup>-bench: Evaluating conversational agents in a dual-control environment. arXiv preprint arXiv:2506.07982, 2025.

Jiawei Chen, Xinyan Guan, Qianhao Yuan, Mo Guozhao, Weixiang Zhou, Yaojie Lu, Hongyu Lin, Ben He, Le Sun, and Xianpei Han. Consistentchat: Building skeleton-guided consistent multi-turn dialogues for large language models from scratch. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 8426–8452, 2025.

Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld, Leshem Choshen, Yoon Kim, and Jacob Andreas. Beyond binary rewards: Training lms to reason about their uncertainty. arXiv preprint arXiv:2507.16806, 2025.

Hanxing Ding, Liang Pang, Zihao Wei, Huawei Shen, and Xueqi Cheng. Rowen: Adaptive retrieval-augmented generation for hallucination mitigation in llms. In Proceedings of the 2025 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, pages 12–21, 2025.

Jinhao Duan, James Difenderfer, Sandeep Madireddy, Tianlong Chen, Bhavya Kailkhura, and Kaidi Xu. Uprop: Investigating the uncertainty propagation of llms in multi-step agentic decision-making. arXiv preprint arXiv:2506.17419, 2025.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024.

Jakob Gawlikowski, Cedrique Rovile Njieutcheu Tassi, Mohsin Ali, Jongseok Lee, Matthias Humt, Jianxiang Feng, Anna Kruspe, Rudolph Triebel, Peter Jung, Ribana Roscher, et al. A survey of uncertainty in deep neural networks. Artificial intelligence review, 56(Suppl 1):1513–1589, 2023.

Jiuzhou Han, Wray Buntine, and Ehsan Shareghi. Towards uncertainty-aware language agent. In Findings of the Association for Computational Linguistics: ACL 2024, pages 6662–6685, 2024.

Juyeon Heo, Miao Xiong, Christina Heinze-Deml, and Jaya Narain. Do llms estimate uncertainty well in instructionfollowing? arXiv preprint arXiv:2410.14582, 2024.

Hamed Jelodar, Samita Bai, Mohammad Meymani, Parisa Hamedi, Roozbeh Razavi-Far, and Ali Ghorbani. Integrating graphs, large language models, and agents: Reasoning and retrieval. Information Fusion, page 104586, 2026.

Zhengbao Jiang, Jun Araki, Haibo Ding, and Graham Neubig. How can we know when language models know? on the calibration of language models for question answering. Transactions of the Association for Computational Linguistics, 9:962–977, 2021.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? In International Conference on Learning Representations, volume 2024, pages 54107–54157, 2024.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Michael Kirchhof, Gjergji Kasneci, and Enkelejda Kasneci. Position: Uncertainty quantification needs reassessment for large-language model agents. arXiv preprint arXiv:2505.22655, 2025.

Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, and Yarin Gal. Semantic entropy probes: Robust and cheap hallucination detection in llms. arXiv preprint arXiv:2406.15927, 2024.

Pengyi Li, Matvey Skripkin, Alexander Zubrey, Andrey Kuznetsov, and Ivan Oseledets. Confidence is all you need: Few-shot rl fine-tuning of language models. arXiv preprint arXiv:2506.06395, 2025.

Rui Li and Shuang Cao. From trajectories to graphs: Contract-checked editing for verifier-guided llm reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 43259–43306, 2026.

Stephanie Lin, Jacob Hilton, and Owain Evans. Teaching models to express their uncertainty in words. arXiv preprint arXiv:2205.14334, 2022.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. In International Conference on Learning Representations, volume 2024, pages 52989–53046, 2024.

Zhengzhao Ma, Xueru Wen, Boxi Cao, Yaojie Lu, Hongyu Lin, Jinglin Yang, Min He, Xianpei Han, and Le Sun. Decoupling reasoning and confidence: Resurrecting calibration in reinforcement learning from verifiable rewards. arXiv preprint arXiv:2603.09117, 2026.

Potsawee Manakul, Adian Liusie, and Mark Gales. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 9004–9017, 2023.

Zhenjiang Mao and Anirudhh Venkat. Recurrent confidence chain: Temporal-aware uncertainty quantification in large language models. arXiv preprint arXiv:2601.13368, 2026.

Mike A Merrill, Alexander G Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E Kelly Buchanan, et al. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. arXiv preprint arXiv:2601.11868, 2026.

Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants. In International Conference on Learning Representations, volume 2024, pages 9025–9049, 2024.

Viktor Moskvoretskii, Maria Marina, Mikhail Salnikov, Nikolay Ivanov, Sergey Pletenev, Daria Galimzianova, Nikita Krayko, Vasily Konovalov, Irina Nikishina, and Alexander Panchenko. Adaptive retrieval without self-knowledge? bringing uncertainty back home. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6355–6384, Vienna, Austria, July 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-long.319.

Changdae Oh, Seongheon Park, To Eun Kim, Jiatong Li, Wendi Li, Samuel Yeh, Sean Du, Hamed Hassani, Paul Bogdan, Dawn Song, et al. Uncertainty quantification in llm agents: Foundations, emerging challenges, and opportunities. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16219–16250, 2026.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551, 2023.

Kaiwen Shi, Zheyuan Zhang, Han Bao, Colby Nelson, and Yanfang Ye. Confidence laundering in agent systems: Why uncertainty needs a latent carrier. arXiv preprint arXiv:2606.20662, 2026.

Yuanfu Sun, Kang Li, Dongzhe Fan, Jiajin Liu, and Qiaoyu Tan. Agentgl: Towards agentic graph learning with llms via reinforcement learning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 25313–25335, 2026.

Sina Tayebati, Divake Kumar, Nastaran Darabi, Davide Ettori, Ranganath Krishnan, and Amit Ranjan Trivedi. Tracer: Trajectory risk aggregation for critical episodes in agentic reasoning. arXiv preprint arXiv:2602.11409, 2026.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. arXiv preprint arXiv:2306.13063, 2023.

Yandong Yan, Junwei Peng, Shijie Li, Chenxi Li, Yifei Shang, Can Deng, Ruiting Dai, Yongqiang Zhao, Jiaqi Zhu, and Yu Huang. Denoiseflow: Uncertainty-aware denoising for reliable llm agentic workflows. arXiv preprint arXiv:2603.00532, 2026.

Daniel Yang, Yao-Hung Hubert Tsai, and Makoto Yamada. On verbalized confidence scores for llms. arXiv preprint arXiv:2412.14737, 2024.

Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

Zhangyue Yin, Qiushi Sun, Qipeng Guo, Zhiyuan Zeng, Xiaonan Li, Junqi Dai, Qinyuan Cheng, Xuan-Jing Huang, and Xipeng Qiu. Reasoning in flux: Enhancing large language models reasoning through uncertainty-aware adaptive guidance. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2401–2416, 2024.

Dongkeun Yoon, Seungone Kim, Sohee Yang, Sunkyoung Kim, Soyeon Kim, Yongil Kim, Eunbi Choi, Yireun Kim, and Minjoon Seo. Reasoning models better express their confidence. arXiv preprint arXiv:2505.14489, 2025.

Caiqi Zhang, Ruihan Yang, Xiaochen Zhu, Chengzu Li, Tiancheng Hu, Yijiang River Dong, Deqing Yang, and Nigel Collier. Confidence estimation for llms in multi-turn interactions. arXiv preprint arXiv:2601.02179, 2026a.

Dengjia Zhang, Xiaoou Liu, Lu Cheng, Yaqing Wang, Kenton Murray, and Hua Wei. Selaur: Self evolving llm agent via uncertainty-aware rewards. In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pages 424–436. Springer, 2026b.

Qiwei Zhao, Dong Li, Yanchi Liu, Wei Cheng, Yiyou Sun, Mika Oishi, Takao Osaki, Katsushi Matsuda, Huaxiu Yao, Chen Zhao, et al. Uncertainty propagation on llm agent. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6064–6073, 2025.

Hongjian Zhou, Fenglin Liu, Boyang Gu, Xinyu Zou, Jinfa Huang, Jinge Wu, Yiru Li, Sam S Chen, Peilin Zhou, Junling Liu, et al. A survey of large language models in medicine: Progress, application, and challenge. arXiv preprint arXiv:2311.05112, 2023.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, volume 2024, pages 15585–15606, 2024.

<table><tr><td>Hyperparameter</td><td>Default</td></tr><tr><td>Visible Window</td><td>5</td></tr><tr><td>weight decay δ</td><td>0.80</td></tr><tr><td>momentum decay γ</td><td>0.75</td></tr><tr><td>Graph propagation weight  $\eta _ { g }$ </td><td>0.70</td></tr><tr><td>momentum weight  $\eta _ { \mathrm { m } }$ </td><td>0.30</td></tr><tr><td>History weight  $\lambda _ { h }$ </td><td>1.0</td></tr><tr><td>local weight  $\cdot \lambda _ { u }$ </td><td>1.0</td></tr></table>

Table 5 Default hyperparameter settings used in RUPA.
<table><tr><td>Edge type</td><td>relation strength</td></tr><tr><td>Sequential</td><td>0.35</td></tr><tr><td>Latest</td><td>0.75</td></tr><tr><td>Repetition</td><td>0.95</td></tr><tr><td>Feedback</td><td>0.85</td></tr><tr><td>Progression</td><td>0.65</td></tr><tr><td>Parallel</td><td>0.45</td></tr></table>

Table 6 Type-specific edge strength used in RUPA.

## A Appendix

## A.1 Implementation Details of RUPA

RUPA uses lightweight deterministic detectors to construct trajectory edges from observable prefix information. Each trajectory step is first normalized into a textual state representation by concatenating the assistant message, reasoning content, tool-call signature, and observation text when available. The resulting text is tokenized after lowercasing, punctuation removal, stop-word filtering, and numeric-token filtering. Tool calls are canonicalized as function-name–argument signatures, which allows RUPA to compare repeated tool usage across steps.

For a candidate historical node $v _ { i }$ and the current assistant node $v _ { t } ,$ RUPA computes token and tool-use matching scores by text embedding distances. In our implementation, a bge-m3 model serve as the embedding model. An alternative way is to compute token overlap in case no embedding models available.

In addition to matching scores, RUPA uses lexical cue matching to distinguish logical relations. Progression edges are detected by matching continuation or refinement cues such as next, therefore, continue, verify, and test. Parallel edges are detected by alternative-branch cues such as alternative, instead, another, diferent, $_ { t r y , }$ and fallback. Feedback edges are detected when previous observations contain instability cues, including empty observations, traceback, error, exception, failed or timeout. This design avoids using future outcomes or final verifier labels during graph construction.

Specifically, for each relation edge type τ , we compute the edge weight by reliablity and its relation strength. The reliability coeficient is computed from unlabeled training trajectories based on the variation of its normalized relation strength,

$$
q _ { \tau } = \frac { \mathrm { V a r } ( \tilde { r } _ { \tau } ) } { \mathbb { E } ( \tilde { r } _ { \tau } ) + \epsilon } , \qquad \rho _ { \tau } = | T | \frac { \exp ( q _ { \tau } / T ) } { \sum _ { \tau ^ { \prime } \in \mathcal { T } } \exp ( q _ { \tau ^ { \prime } } / T ) } ,\tag{6}
$$

where $\tau$ denotes the set of relation types and $T$ is a temperature parameter.

The detailed hyperparameters of RUPA is shown in table 5 and Table 6

![](images/03fd6042a12b8c6b6a98f2dfc932d7bf743191e868ae45589210f12515a5f596.jpg)  
Figure 5 Parameter sensitivity analysis of RUPA on GAIA with MiniMax-M2.7. Each subplot shows AUROC as a function of one hyperparameter while keeping the remaining settings fixed. The dashed gray line indicates the default values of hyperparameters, or the corresponding default edge weight used in the main experiments.

## A.2 Detailed Experiment Settings

All methods are evaluated under the same benchmark splits, prompts, and execution framework as described in the main paper. We use the Harbor framework for trajectory execution and verification, and fix the decoding temperature to 0.7 for all repeated-sampling based methods. Unless otherwise stated, each repeated-sampling baseline is run with 3 samples per query, and the final uncertainty score is computed according to the original scoring rule of the corresponding method.

For baseline reproduction, we follow the oficial implementation whenever available. In particular, Tracer is reproduced using its released codebase. For SAUP and Uprop, no oficial implementation was available at the time of experimentation, so we reimplemented both methods independently according to the descriptions in their papers and matched their reported scoring procedures as closely as possible. To ensure comparability, all baselines are evaluated on the same agent trajectories, model outputs, and task instances as RUPA. For methods requiring step-level or trajectory-level aggregation, we preserve the original aggregation strategy specified by each baseline. Hyperparameters that are not explicitly defined by a baseline are set to the paper default when available; otherwise, we use a validation-based choice on the training split without accessing test labels.

For RUPA, graph construction and uncertainty propagation use the outcome-blind calibration procedure described in the main text. All graph-related parameters, including relation weights, temporal decay, and history window size, are determined from unlabeled training trajectories only, and is fixed across experiments of diferent model families and datasets. No test labels are used during parameter selection or calibration.

## A.3 Parameter Sensitivity Ablation Analysis

In order to evaluate how the hyperparameter chosen in RUPA method affect the final failure prediction performance, we conduct a parameter sensitive ablation analysis experiment on MiniMax-M2.7 model with gaia datasets, when one parameter ablation experiment is conducted, other hyperparameter is fixed as our main experiment setting. The result is shown in Fig 5:

As shown by the resulting curves, for hyper-parameters like graph decay or momentum weight, RUPA is not overly sensitive to small perturbations around the default configuration, and its performance remains stable across a broad range of reasonable settings. Furthermore, the default hyperparameters consistently fall near a strong or near-optimal region for most parameters, suggesting that our edge-weight assignment strategy provides a sensible balance between diferent structural signals. These results indicate that the proposed parameterization is reasonable and that the edge-weight calibration method can assign meaningful importance to diferent relation types.