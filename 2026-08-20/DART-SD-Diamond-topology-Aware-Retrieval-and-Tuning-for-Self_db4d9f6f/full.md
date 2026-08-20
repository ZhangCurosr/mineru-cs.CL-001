# DART-SD: Diamond-topology Aware Retrieval and Tuning for Self-Distillation of Multi-Turn Tool-Calling Agents

Hangrui Xu<sup>1,⋆</sup>, Jiarui Wang<sup>1,⋆</sup>, Yang Yang<sup>1,⋆,†</sup>, Chuanbo Zhu<sup>1</sup>, Fangda Chen<sup>1</sup>, Ziqi Wu<sup>1</sup>, Jingming Cai<sup>1</sup>, Yan Song<sup>2,†</sup>

<sup>1</sup>ByteDance, <sup>2</sup>University of Science and Technology of China

<sup>⋆</sup>Equal contribution, <sup>†</sup>Corresponding author

## Abstract

Equipping Large Language Models (LLMs) with multi-turn tool-calling capabilities is essential for building autonomous agents. However, progress is fundamentally limited by the reliance on full-length trajectory imitation. For tasks involving multiple order-independent sub-goals, the optimal solution space forms a vast combinatorial diamond lattice. Forcing this rich topology into monolithic trajectories causes a severe topological collapse, indiscriminately penalizing valid alternative explorations and severely degrading policy diversity. To address this, we propose DART-SD (Diamond-topology Aware Retrieval and Tuning for Self-Distillation), a novel framework that shifts the paradigm from global forcing to topology-guided localized correction. DART-SD first models the execution process as a converging Interaction-State Transition Graph (ISTG), faithfully capturing the inherent diamond topology of successful and failed exploratory paths. During autonomous rollouts, the framework identifies the Critical Topological Breakpoint (CTB) and retrieves success-supported recovery references. Finally, we introduce a progressive self-distillation paradigm through CTB-guided localized supervision, ensuring that the training loss is calculated exclusively on the generated recovery steps while strictly protecting the valid reasoning prefix from destructive gradient updates. Experiments on complex multi-turn tool-calling benchmarks demonstrate that DART-SD significantly outperforms traditional full-trajectory baselines.

Correspondence: Yang Yang at yang.yves@bytedance.com, Yan Song at clksong@gmail.com

## 1 Introduction

Equipping Large Language Models (LLMs) with multi-turn tool-calling capabilities is an essential milestone toward building autonomous agents capable of solving complex, real-world tasks [21, 30, 32]. Through sequential interactions with external environments, tool-augmented agents can retrieve up-to-date information, execute code, and manipulate APIs to transcend their static parametric knowledge [3, 4, 8, 31, 34]. Currently, state-of-the-art tool-calling performance is predominantly achieved by massive and expensive frontier models [1, 6]. Consequently, agent distillation has gained significant research momentum, aiming to transfer these sophisticated planning and interaction capabilities into eficient, compact open-source models [12, 34, 37].

The prevalent agent distillation paradigm relies on full trajectory imitation via Behavior Cloning (BC) or Supervised Fine-Tuning (SFT) [22]. In long-horizon tasks, global forcing paradigms apply an indiscriminate global loss that overwrites the student model’s valid exploratory steps [17], causing models to memorize redundant trajectories instead of extracting the task’s core logical backbone. Recent eforts attempt to explore hindsight-based scafolding or Reinforcement Learning (RL) [28, 39, 44]. However, traditional hindsight methods [40] still treat multi-turn interactions as strict linear sequences. This rigid linearity forces models to confuse fatal errors with harmless exploration, severely degrading token eficiency and internal knowledge consistency [18, 45]. Meanwhile, standard RL approaches [26, 29], such as Group Relative Policy Optimization (GRPO), frequently sufer from credit misassignment due to uniformly distributed rewards spread across all intermediate tool calls and inadvertently penalize valid steps within failed trajectories (see Figure 1).

![](images/9c245b2ed1366f1538d208389cae7155235e45b58462c57cd89552b1ab61b999.jpg)  
Figure 1 Comparison of training paradigms. (a) SFT Teacher Boosting applies an indiscriminate global loss, which overwrites valid exploration. (b) Standard-RL (GRPO) misassigns credit through a uniformly distributed reward spread. (c) DART-SD dynamically identifies the Critical Topological Breakpoints (CTB) and applies localized correction while preserving valid exploration.

The monolithic SFT and standard RL paradigms are fundamentally limited by misrepresenting the multi-turn tool-calling process as a collection of isolated linear trajectories [2, 11]. This oversimplified perspective ignores the inherent graph structure of state transitions [9, 13]. For tasks with order-independent sub-goals, distinct trajectories frequently intersect at shared intermediate states, structuring the optimal solution space into a vast combinatorial diamond lattice [7]. Forcing models to flatten this interconnected topological space into single linear paths, traditional methods inevitably cause a severe topological collapse. This structural oversight induces myopic credit assignment [29], arbitrarily penalizing valid alternative explorations and severely degrading policy diversity [15, 41].

To align with this topological reality, we propose DART-SD (Diamond-topology Aware Retrieval and Tuning for Self-Distillation), a framework that shifts the paradigm from linear imitation to topology-aware localized correction. DART-SD extracts a structural prior from teacher rollouts, instantiated as an Interaction-State Transition Graph (ISTG) covering both successful and failed trajectories. By defining nodes as cumulative interaction states rather than transient actions, the ISTG captures the diamond topology, naturally resolving order-dependency conflicts. During self-distillation, DART-SD projects student states onto the successreachable region to identify the Critical Topological Breakpoint (CTB) and its recovery anchors. From each anchor, it searches the ISTG for a success-supported continuation as a reference for recovery generation. Finally, DART-SD applies a progressive self-distillation paradigm through CTB-guided localized supervision, ensuring that the loss is calculated only on recovery steps to protect valid reasoning prefixes.

In summary, our main contributions are as follows:

• We identify a fundamental topological collapse in existing agent-training paradigms. To resolve this, we introduce the Interaction-State Transition Graph (ISTG), which represents tool execution through cumulative interaction states and captures the diamond topology induced by order-independent exploration.

• We define the Critical Topological Breakpoint (CTB) by projecting student interaction states onto the empirical success-reachable region. We propose CTB-guided localized supervision, which retrieves success-supported references and supervises the generated recovery steps while strictly preserving valid prefixes.

• We propose a progressive self-distillation paradigm that repeatedly rolls out the student, identifies its evolving capability boundary, and performs CTB-guided localized supervision. This iterative process enables the student to continuously extend valid interaction prefixes and progressively master more complex tool-use behaviors.

• Extensive experiments across five benchmarks and two model scales demonstrate that DART-SD consistently outperforms supervised and reinforcement-learning baselines, while reducing redundant tool calls and extending valid interaction prefixes.

## 2 Related Work

Distillation-Based Paradigms. Early attempts at building autonomous agents rely on Behavior Cloning (BC) or Supervised Fine-Tuning (SFT) over complete expert trajectories [22]. However, this global forcing paradigm forces students to mimic entire sequences, often leading to compounding errors in long-horizon tasks as deviations propagate autoregressively [16]. Furthermore, indiscriminate token-level imitation triggers exposure bias [25] and overwrites the student’s valid exploration habits. To address the limitations of static global forcing, recent distillation methods have shifted toward student-centered and dynamic strategies [14]. SCoRe-SFT [16] provides post-hoc scafolding through student-centered knowledge distillation, while On-Policy Self-Distillation (OPSD) [44] mitigates ofline distribution shifts by dynamically rolling out and distilling trajectories from the current student policy. Similarly, hindsight-based distillation methods such as HINT-SD [40] leverage targeted scafolding for correction.

Reinforcement Learning Paradigms. To further refine policy optimization beyond distillation, Reinforcement Learning (RL) approaches leverage environment feedback to directly align agent behaviors. Methods such as FTRL-GRPO [39] and ToolRL [20] optimize agents using sparse outcome rewards and verifiable execution feedback. However, sparse terminal signals in these methods often lead to severe credit misassignment, inadvertently penalizing valid intermediate steps within failed trajectories [5, 17, 26, 27, 39, 43]. Even finergrained RL methods such as MatchTIR [23], which aligns trajectories via bipartite matching to provide turnlevel rewards, still fundamentally treat multi-turn interactions as rigid linear sequences, thereby overlooking the inherent topology of exploration. In contrast, our proposed DART-SD shifts the paradigm from linear imitation and complex matching to topology-aware localized correction. By calculating the optimization loss exclusively on retrieved recovery steps, DART-SD strictly protects valid reasoning prefixes from destructive gradient updates.

## 3 Methodology

Figure 2 provides an overview of the DART-SD framework. DART-SD first constructs an ISTG from teacher rollouts, then identifies the CTB of failed student rollouts through success-reachable projection. The identified CTB is used to generate localized recovery continuations and to drive progressive self-distillation.

## 3.1 ISTG Abstraction and Construction

Information Atom Abstraction. For each task x, useful facts extracted from tool responses are normalized into a task-specific set of information atoms $\kappa _ { x } .$ Semantically equivalent responses share the same atom, while non-informative responses contribute none. We realize this in two stages. A deterministic stage canonicalizes each tool response and decides whether it carries task-usable data at all. Parsing the response into fields, it is treated as non-informative only when every field is either a status signal or an empty or placeholder value, so a payload retaining a single substantive value still qualifies as informative, whereas an error or not-found message does not. Non-informative responses are collapsed per tool into a single class, which removes the bulk of the corpus without semantic judgement.

![](images/23ef5483978258171471a4d30d07289bb9982478a1bb5f3a34004267e1618de3.jpg)  
Figure 2 Overview of DART-SD. (1) DART-SD constructs an ISTG from teacher rollouts, where main and auxiliary nodes model information acquisition and useless exploration. (2–3) Failed student rollouts are collected and replayed in the same interaction-state space, then projected onto the budget-filtered success-reachable region $\mathcal { R } _ { x } ^ { + }$ , where the first projectable-to-non-projectable transition defines the CTB. (4–5) Conditioned on the retained student prefix and privileged teacher references, DART-SD generates a recovery continuation after the CTB and applies localized supervision only to the generated assistant tokens. (6) This CTB-guided SFT loop is progressively repeated across self-distillation rounds.

A semantic stage then assigns atoms to the surviving candidates. Let e denote a tool call, with tool identity tl(e) and canonicalized response o¯(e). All candidates of a task are examined jointly, conditioned on the task question and one successful rollout, yielding a set-valued atom map

$$
\alpha _ { x } : \ \mathrm { ( t l } ( e ) , \bar { o } ( e ) \big ) \ \longmapsto \ \alpha _ { x } ( e ) \subseteq { \mathcal K } _ { x } , \qquad | \alpha _ { x } ( e ) | \leq 1 .\tag{1}
$$

An informative response is mapped to the singleton containing its atom, while a non-informative response, or one left unjudged, is mapped to ∅. Responses supplying the same fact are constrained to receive the same atom, so each element of $\kappa _ { x }$ is the canonical representative of one class of information-equivalent calls, namely those whose images under $\alpha _ { x }$ coincide and are non-empty. Judging a task’s candidates jointly is what makes this equivalence decidable: atoms are assigned with all alternatives in view, rather than labeling responses independently and relying on the labels to agree.

The map $\alpha _ { x }$ supplies the acquisition increment used in Eq. (4). Writing $B _ { t }$ for the tool call or concurrent call bundle executed at step t,

$$
\begin{array} { r } { \varDelta I _ { t } = \left\{ k \in \mathcal { K } _ { x } \ | \ \exists e \in B _ { t } , \ k \in \alpha _ { x } ( e ) , \ k \notin I _ { t - 1 } \right\} , \qquad I _ { t } = I _ { t - 1 } \cup \varDelta I _ { t } . } \end{array}\tag{2}
$$

Two properties of this construction determine the resulting topology. First, $\alpha _ { x }$ identifies responses across surface form and across tools, so the same fact returned as free text by one tool and as a structured record by another yields one atom; the two acquisition paths therefore reconverge at the same main information state instead of forking into states the projection would treat as unrelated. Second, mapping non-informative and unjudged responses to ∅ is what makes them contribute none: such a call contributes no element to $\varDelta I _ { t }$ in Eq. (2) and leaves the information set unchanged, instead of receiving an identifier derived from its own text, which would manufacture a distinct information state that no evidence supports and inflate the information dimension of the graph. Because only informative responses yield atoms, residual judgement errors predominantly omit an atom rather than invent one, leaving the graph sparser but never fabricating states or reachability.

Main and Auxiliary Nodes. For a task $x ,$ the interaction state at step t is represented as

$$
X _ { t } = ( I _ { t } , U _ { t } ) ,\tag{3}
$$

where $I _ { t }$ is the set of canonical information atoms acquired up to step $t ,$ and $U _ { t }$ records the multiset of useless operations performed since the most recent main node.

The root state $X _ { 0 } = ( \varnothing , \varnothing )$ is defined as a main node. For $t \geq 1$ , the type of $X _ { t }$ is determined by the preceding transition:

$$
\operatorname { t y p e } ( X _ { t } ) = { \left\{ \begin{array} { l l } { \operatorname* { m a i n } , } & { \varDelta I _ { t } \neq \emptyset , } \\ { \operatorname { a u x } , } & { \varDelta I _ { t } = \emptyset { \mathrm { ~ a n d ~ } } U _ { t } \neq \emptyset . } \end{array} \right. }\tag{4}
$$

where $\varDelta I _ { t }$ is the set of previously unseen information atoms acquired at step t, given by Eq. (2).

Main nodes include mixed steps acquiring new information alongside useless operations and are represented with $U _ { t } = \emptyset$ , whereas auxiliary nodes record useless operations after the main node without changing the information set.

Interaction-State Transition Graph (ISTG). For each task x, we construct a directed multigraph

$$
G _ { x } = ( V _ { x } , E _ { x } ) .\tag{5}
$$

The graph is built from all successful and failed teacher rollouts. All rollouts share a common root and terminate at either a success or a failure terminal. Each vertex $v \in V _ { x }$ represents a state $X ( v ) = ( I ( v ) , U ( v ) )$ with type type(v) ∈ {main, aux}, and each edge represents one tool call or concurrent call bundle. Parallel edges are retained.

Main nodes form the information-acquisition backbone, while auxiliary nodes represent useless exploration attached to the most recent main node. Both node types and transitions are retained, since auxiliary exploration contributes to execution depth and matching. The set-valued information component allows order-independent acquisition paths to diverge and reconverge at the same main information state, producing the diamond structures of the ISTG. A student rollout is replayed under the same state-update rule:

$$
{ \cal P } _ { x } ^ { s } = ( s _ { 0 } , \ldots , s _ { T } ) , \qquad X ( s _ { t } ) = ( I _ { t } ^ { s } , U _ { t } ^ { s } ) .\tag{6}
$$

The node type of $s _ { t }$ is determined by the preceding transition. Teacher and student executions are therefore embedded in a shared interaction-state space, enabling separate matching of information acquisition and useless exploration without requiring action-wise correspondence.

## 3.2 Success-Reachable Projection and CTB

Empirical Success-Reachable Region. Let ${ \mathcal { T } } _ { x } ^ { + }$ denote the successful teacher rollouts for task $x ,$ and let $d _ { x } ^ { \operatorname* { m i n } }$ be the minimum number of graph transitions from the root to a success terminal among these rollouts. We define the task-specific reachability budget as

$$
B _ { x } = \operatorname* { m i n } \ ( d _ { x } ^ { \operatorname* { m i n } } + \varDelta _ { x } , \ B _ { x } ^ { \operatorname* { m a x } } ) ,\tag{7}
$$

where $\varDelta _ { x }$ is the task-dependent allowance beyond the shortest successful depth and $B _ { x } ^ { \mathrm { m a x } }$ is the budget cap. For a node v on a successful rollout τ, let $r _ { \tau } ( v )$ denote the number of graph transitions remaining from v to the success terminal of τ. The empirical success-reachable region is defined by

$$
\mathcal { R } _ { x } ^ { + } = \left\{ v \in V _ { x } \vert \exists \tau \in \mathcal { T } _ { x } ^ { + } , v \in \tau , r _ { \tau } ( v ) \leq B _ { x } \right\} .\tag{8}
$$

Thus, $\mathcal { R } _ { x } ^ { + }$ contains both main and auxiliary nodes on successful teacher rollouts whose remaining distance to success is at most $B _ { x }$ . The distance is measured backward from the success terminal along the rollout, rather than as forward depth from the root. States observed exclusively on failed rollouts remain in $G _ { x }$ , but are excluded from $\mathcal { R } _ { x } ^ { + }$ and cannot serve as recovery anchors.

Type-specific State Projection. For each student state $s _ { t } ~ = ~ ( I _ { t } ^ { s } , U _ { t } ^ { s } )$ , we further apply the projection rule associated with its node type. Let ${ \mathcal { R } } _ { x , \operatorname* { m a i n } } ^ { + }$ and ${ \mathcal { R } } _ { x , \mathrm { a u x } } ^ { + }$ denote the main and auxiliary nodes in the budget-filtered empirical success-reachable region.

For a student main node, the acceptable teacher anchors are defined over non-root successful nodes with valid information acquisition:

$$
\begin{array} { r } { \mathcal { A } _ { t } ^ { \operatorname* { m a i n } } = \left\{ v \in \mathcal { R } _ { x , \operatorname* { m a i n } } ^ { + } \ \middle | \ I ( v ) \subseteq I _ { t } ^ { s } \right\} . } \end{array}\tag{9}
$$

Thus, a student main node can be aligned with any reachable teacher main node whose information set is contained in the student’s acquired information. Equality of information sets and the useless-operation component are not required. For mixed transitions, the criterion depends only on the information component, since any transition with information gain is classified as a main transition.

For a student auxiliary node, we first consider the reachable teacher main nodes whose information is contained in $I _ { t } ^ { s }$ :

$$
\mathcal { M } _ { t } = \left\{ m \in \mathcal { R } _ { x , \operatorname* { m a i n } } ^ { + } \ \Big | \ I ( m ) \subseteq I _ { t } ^ { s } \right\} .\tag{10}
$$

If $\mathcal { M } _ { t } \neq \emptyset$ , let $m _ { t }$ be a node in $\mathcal { M } _ { t }$ with the largest information-set cardinality. We then define the acceptable auxiliary anchors as:

$$
\mathcal { A } _ { t } ^ { \mathrm { a u x } } = \left\{ v \in \mathcal { R } _ { x , \mathrm { a u x } } ^ { + } \ \middle | \ \mathrm { p a r } ( v ) = m _ { t } , \ | U ( v ) | = | U _ { t } ^ { s } | \right\} ,\tag{11}
$$

where $\operatorname { p a r } ( v )$ denotes the reachable teacher main node from which the auxiliary node v is reached. Thus, the projection compares only the number of useless operations performed after $m _ { t } .$ , without requiring the unsuccessful tools to be identical. If $\mathcal { M } _ { t } = \emptyset$ , then $\mathcal { A } _ { t } ^ { \mathrm { a u x } } = \emptyset$ . Let $\boldsymbol { A } _ { t }$ be $\boldsymbol { A } _ { t } ^ { \mathrm { m a i n } }$ or $\mathcal { A } _ { t } ^ { \mathrm { a u x } }$ according to the type of $s _ { t } ,$ , and define

$$
\rho _ { t } = \mathbb { I } [ A _ { t } \neq \infty ] .\tag{12}
$$

If $\rho _ { t } = 1$ , the student state corresponds to at least one reachable teacher node of the same type. We formally denote the selected valid teacher anchor as $\pi ( s _ { t } )$ : for a main node, $\pi ( s _ { t } )$ is chosen from $\mathcal { A } _ { t } ^ { \mathrm { m a i n } }$ as a node with the maximal information set; for an auxiliary node, $\pi ( s _ { t } )$ can be any anchor in $\mathcal { A } _ { t } ^ { \mathrm { a u x } }$ . If $\rho _ { t } = 0$ , no valid same-type anchor exists and the student state is treated as outside the success-reachable region.

Critical Topological Breakpoint (CTB). Since the initial state is shared by the teacher and student, $\rho _ { 0 } = 1$ The CTB is defined as the first transition from a projectable to a non-projectable student state:

$$
\begin{array} { r l } & { t _ { \mathrm { C } } = \operatorname* { m i n } \{ t : \rho _ { t - 1 } = 1 , \ \rho _ { t } = 0 \} , } \\ & { a _ { \mathrm { C } } = \pi ( s _ { t _ { \mathrm { C } } - 1 } ) , } \end{array}\tag{13}
$$

where $\pi ( \cdot )$ denotes the projection mapping that assigns a valid student node to its teacher anchor. Thus, $t _ { \mathrm { C } }$ identifies the first departure from the teacher-supported region, and the anchor $a _ { \mathrm { C } }$ is the last valid teacher projection before that departure. This boundary captures the earliest point at which the student either acquires an unsupported information combination or exceeds the teacher-supported exploration count under the corresponding main node. If the student remains projectable until the end of the rollout but still fails, we instead view the terminal state as the correction boundary and use its latest valid projection as the recovery anchor.

## 3.3 CTB-Guided Localized Supervision

Privileged-context Retrieval. We randomly sample successful and failed teacher traces from the teacher graph as privileged references, denoted as ${ \mathcal { C } } ^ { \mathrm { { p r i v } } }$ . The reference traces are not directly concatenated with the student prefix, since the two trajectories may contain diferent calls and observations. Instead, an augmented generator is conditioned on the task, the student prefix retained before the CTB, and the privileged references:

$$
c _ { t _ { \mathrm { C } } } ^ { * } = \mathrm { A u g G e n } \big ( x , \tau _ { < t _ { \mathrm { C } } } ^ { s } , \mathcal { C } ^ { \mathrm { p r i v } } \big ) .\tag{14}
$$

The training trajectory is formed by appending $c _ { t _ { \mathrm { C } } } ^ { * }$ to the retained student prefix. The trajectory is then tokenized, and CTB-localized supervision is applied to the resulting training sequence. Thus, the realized student context is preserved, while the sampled teacher traces provide privileged information for generating the subsequent continuation.

CTB-Localized Supervision. Let $\widetilde { \boldsymbol { y } } = ( \widetilde { y } _ { 1 } , \dots , \widetilde { y } _ { L } )$ denote the tokenized training trajectory. We train with the masked causal language modeling objective:

$$
\mathcal { L } _ { \mathrm { D A R T } } = - \sum _ { i = 1 } ^ { L } m _ { i } \log p _ { \theta } \left( \widetilde { y } _ { i } \mid \widetilde { y } _ { < i } , x \right) .\tag{15}
$$

Although the loss is computed at the token level, the mask is defined over response steps: $m _ { i } = 1$ only if token $\widetilde { y } _ { i }$ belongs to an assistant response step generated after the CTB and before the final-answer step. Tokens in the retained student prefix, user messages, tool observations, and the final-answer step receive zero weight. Thus, supervision is restricted to post-CTB assistant response steps, while the pre-CTB behavior and final answer are not directly optimized.

## 3.4 Progressive Self-Distillation Paradigm

The task-specific ISTG is maintained throughout the self-distillation. At each iteration, the current student produces new rollouts, which are mapped into the shared interaction-state space and processed by the type-specific projection and breakpoint-localization procedure. Failed rollouts yield breakpoint-localized privileged-context distillation instances. As the student improves, its states remain projectable for longer portions of the interaction trajectory. The detected CTBs therefore track the evolving boundary at which the current policy first departs from the teacher-supported region. DART-SD consequently induces a selfpaced curriculum that progressively shifts supervision toward recovery behaviors beyond the current tool-use capability.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We train the models on the FTRL dataset [39], which comprises over 2,000 automatically constructed tool-use environments with verifiable feedback. FTRL covers four types of task structures: Single, containing a single sub-question; Para-Single, consisting of multiple independent sub-questions executable in parallel; Multi, comprising a sequence of dependent sub-questions; and Para-Multi, combining both independent and dependent sub-questions. We evaluate DART-SD on five tool-use benchmarks, including FTRL as the in-domain test set, and BFCL [19], ToolHop [38], τ-bench [35], and RoTBench [36] as out-of-domain benchmarks to assess generalization.

Baselines. We evaluate DART-SD using both Qwen3-4B and Qwen3-8B backbones [33]. For each backbone, we first report the performance of the pretrained model without task-specific adaptation, denoted as Base. The remaining baselines are grouped into two optimization paradigms. (1) Distillation-based Paradigms: standard SFT, SCoRe-SFT [16], and OPSD [44]. (2) RL Paradigms: FTRL-GRPO [39], ToolRL [20], and MatchTIR [23] (evaluated using both MatchTIR-OT and MatchTIR-KM). To ensure a strictly controlled comparison, all methods are trained and evaluated under a no-thinking configuration.

Evaluation Metrics. To ensure a rigorous and standardized assessment, we strictly adhere to the oficial evaluation protocols for each respective benchmark. (1) FTRL: We evaluate trajectory precision and task completeness using Solve-P and Solve-R along with their harmonic mean, Solve-F1. (2) BFCL: We report the average Multi-Turn score over Base, Miss Function, Miss Parameter and Long Context. (3) ToolHop: We evaluate the model’s ability to resolve complex multi-hop tool dependencies using Answer Correctness (AC). (4) τ-bench: We assess performance on real-world API orchestration tasks using Pass^1. (5) RoTBench: We evaluate fine-grained execution robustness through Tool Selection (TS), Parameter Identification (PI), and Content Filling (CF).

<table><tr><td rowspan=2 colspan=1>Methods</td><td rowspan=1 colspan=3>FTRL</td><td rowspan=1 colspan=1>BFCL</td><td rowspan=1 colspan=1>ToolHop</td><td rowspan=1 colspan=1>τ-bench</td><td rowspan=1 colspan=3>RoTBench</td><td rowspan=2 colspan=1>Avg.</td></tr><tr><td rowspan=1 colspan=3>Solve-P Solve-R Solve-F1</td><td rowspan=1 colspan=1>Multi-Turn</td><td rowspan=1 colspan=1>AC</td><td rowspan=1 colspan=1>Pass^1</td><td rowspan=1 colspan=3>TS    PI    CF</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>Qwen3-4B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>21.00</td><td rowspan=1 colspan=1>26.54</td><td rowspan=1 colspan=1>21.81</td><td rowspan=1 colspan=1>10.14</td><td rowspan=1 colspan=1>20.20</td><td rowspan=1 colspan=1>15.15</td><td rowspan=1 colspan=3>69.5226.31 16.07</td><td rowspan=1 colspan=1>25.19</td></tr><tr><td rowspan=1 colspan=1> SFT</td><td rowspan=1 colspan=1>34.26</td><td rowspan=1 colspan=1>49.02</td><td rowspan=1 colspan=1>37.96</td><td rowspan=1 colspan=1>14.57</td><td rowspan=1 colspan=1>40.50</td><td rowspan=1 colspan=1>21.82</td><td rowspan=1 colspan=1>71.67</td><td rowspan=1 colspan=1>44.40</td><td rowspan=1 colspan=1>24.40</td><td rowspan=1 colspan=1>37.62</td></tr><tr><td rowspan=1 colspan=1>SCoRe-SFT</td><td rowspan=1 colspan=1>23.79</td><td rowspan=1 colspan=1>28.20</td><td rowspan=1 colspan=1>24.61</td><td rowspan=1 colspan=1>12.75</td><td rowspan=1 colspan=1>25.83</td><td rowspan=1 colspan=1>11.52</td><td rowspan=1 colspan=1>60.36</td><td rowspan=1 colspan=1>31.19</td><td rowspan=1 colspan=1>18.33</td><td rowspan=1 colspan=1>26.29</td></tr><tr><td rowspan=2 colspan=1> OPSD FTRL-GRPO</td><td rowspan=1 colspan=1>22.37</td><td rowspan=1 colspan=1>29.21</td><td rowspan=1 colspan=1>23.50</td><td rowspan=1 colspan=1>11.00</td><td rowspan=1 colspan=1>20.20</td><td rowspan=1 colspan=1>23.03</td><td rowspan=1 colspan=1>68.93</td><td rowspan=1 colspan=1>35.24</td><td rowspan=1 colspan=1>21.67</td><td rowspan=1 colspan=1>28.35</td></tr><tr><td rowspan=1 colspan=1>36.83</td><td rowspan=1 colspan=1>41.71</td><td rowspan=1 colspan=1>37.84</td><td rowspan=1 colspan=1>13.50</td><td rowspan=1 colspan=1>29.25</td><td rowspan=1 colspan=1>20.61</td><td rowspan=1 colspan=1>70.36</td><td rowspan=1 colspan=1>32.62</td><td rowspan=1 colspan=1>20.24</td><td rowspan=1 colspan=1>33.66</td></tr><tr><td rowspan=1 colspan=1> ToolRL</td><td rowspan=1 colspan=1>26.89</td><td rowspan=1 colspan=1>33.23</td><td rowspan=1 colspan=1>28.47</td><td rowspan=1 colspan=1>9.88</td><td rowspan=1 colspan=1>20.60</td><td rowspan=1 colspan=1>18.18</td><td rowspan=1 colspan=1>69.64</td><td rowspan=1 colspan=1>38.21</td><td rowspan=1 colspan=1>24.05</td><td rowspan=1 colspan=1>29.91</td></tr><tr><td rowspan=1 colspan=1> MatchTIR (OT)</td><td rowspan=1 colspan=1>23.55</td><td rowspan=1 colspan=1>29.27</td><td rowspan=1 colspan=1>24.95</td><td rowspan=1 colspan=1>10.38</td><td rowspan=1 colspan=1>26.23</td><td rowspan=1 colspan=1>16.36</td><td rowspan=1 colspan=1>72.02</td><td rowspan=1 colspan=1>32.74</td><td rowspan=1 colspan=1>20.36</td><td rowspan=1 colspan=1>28.43</td></tr><tr><td rowspan=1 colspan=1> MatchTIR(KM)</td><td rowspan=1 colspan=1>25.54</td><td rowspan=1 colspan=1>31.10</td><td rowspan=1 colspan=1>26.50</td><td rowspan=1 colspan=1>10.00</td><td rowspan=1 colspan=1>26.63</td><td rowspan=1 colspan=1>21.82</td><td rowspan=1 colspan=1>72.02</td><td rowspan=1 colspan=1>33.81</td><td rowspan=1 colspan=1>21.43</td><td rowspan=1 colspan=1>29.87</td></tr><tr><td rowspan=1 colspan=1>DART-SD (Ours)</td><td rowspan=1 colspan=2>36.70    48.16</td><td rowspan=1 colspan=1>39.77</td><td rowspan=1 colspan=1>23.88</td><td rowspan=1 colspan=1>42.11</td><td rowspan=1 colspan=1>23.03</td><td rowspan=1 colspan=1>72.02</td><td rowspan=1 colspan=1>42.38</td><td rowspan=1 colspan=1>24.52</td><td rowspan=1 colspan=1>39.17</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Qwen3-8B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=2>21.18    30.71</td><td rowspan=1 colspan=1>23.48</td><td rowspan=1 colspan=1>18.38</td><td rowspan=1 colspan=1>28.54</td><td rowspan=1 colspan=1>10.13</td><td rowspan=1 colspan=1>75.52</td><td rowspan=1 colspan=1>36.29</td><td rowspan=1 colspan=1>22.19</td><td rowspan=1 colspan=1>29.60</td></tr><tr><td rowspan=1 colspan=1>SFT</td><td rowspan=1 colspan=1>38.08</td><td rowspan=1 colspan=1>50.95</td><td rowspan=1 colspan=1>41.89</td><td rowspan=1 colspan=1>19.25</td><td rowspan=1 colspan=1>43.52</td><td rowspan=1 colspan=1>26.06</td><td rowspan=1 colspan=1>75.95</td><td rowspan=1 colspan=1>48.81</td><td rowspan=1 colspan=1>30.24</td><td rowspan=1 colspan=1>41.64</td></tr><tr><td rowspan=1 colspan=1>SCoRe-SFT</td><td rowspan=1 colspan=1>31.28</td><td rowspan=1 colspan=1>34.48</td><td rowspan=1 colspan=1>31.58</td><td rowspan=1 colspan=1>19.25</td><td rowspan=1 colspan=1>30.25</td><td rowspan=1 colspan=1>18.18</td><td rowspan=1 colspan=1>70.36</td><td rowspan=1 colspan=1>40.48</td><td rowspan=1 colspan=1>24.52</td><td rowspan=1 colspan=1>33.38</td></tr><tr><td rowspan=1 colspan=1>OPSD</td><td rowspan=1 colspan=1>24.37</td><td rowspan=1 colspan=1>34.88</td><td rowspan=1 colspan=1>26.68</td><td rowspan=1 colspan=1>20.50</td><td rowspan=1 colspan=1>41.11</td><td rowspan=1 colspan=1>21.21</td><td rowspan=1 colspan=1>75.24</td><td rowspan=1 colspan=1>43.21</td><td rowspan=1 colspan=1>27.38</td><td rowspan=1 colspan=1>34.95</td></tr><tr><td rowspan=1 colspan=1> FTRL-GRPO</td><td rowspan=1 colspan=2>37.66    45.49</td><td rowspan=1 colspan=1>40.22</td><td rowspan=1 colspan=1>35.25</td><td rowspan=1 colspan=1>34.57</td><td rowspan=1 colspan=1>23.03</td><td rowspan=1 colspan=1>77.02</td><td rowspan=1 colspan=2>42.74 27.02</td><td rowspan=1 colspan=1>40.33</td></tr><tr><td rowspan=1 colspan=1> ToolRL</td><td rowspan=1 colspan=2>32.49    41.07</td><td rowspan=1 colspan=1>35.00</td><td rowspan=1 colspan=1>20.50</td><td rowspan=1 colspan=1>44.72</td><td rowspan=1 colspan=1>25.45</td><td rowspan=1 colspan=1>75.71</td><td rowspan=1 colspan=2>51.43 33.10</td><td rowspan=1 colspan=1>39.94</td></tr><tr><td rowspan=1 colspan=1> MatchTIR (OT)</td><td rowspan=1 colspan=2>29.45    34.89</td><td rowspan=1 colspan=1>30.65</td><td rowspan=1 colspan=1>21.38</td><td rowspan=1 colspan=1>38.79</td><td rowspan=1 colspan=1>24.24</td><td rowspan=1 colspan=1>75.00</td><td rowspan=1 colspan=2>40.36 25.36</td><td rowspan=1 colspan=1>35.57</td></tr><tr><td rowspan=1 colspan=1> MatchTIR (KM)</td><td rowspan=1 colspan=2>33.07    40.42</td><td rowspan=1 colspan=1>35.37</td><td rowspan=1 colspan=1>23.25</td><td rowspan=1 colspan=1>41.71</td><td rowspan=1 colspan=1>26.06</td><td rowspan=1 colspan=1>75.71</td><td rowspan=1 colspan=2>42.26 26.90</td><td rowspan=1 colspan=1>38.31</td></tr><tr><td rowspan=1 colspan=1>DART-SD (Ours)</td><td rowspan=1 colspan=2>42.00    54.13</td><td rowspan=1 colspan=1>45.66</td><td rowspan=1 colspan=1>27.63</td><td rowspan=1 colspan=1>45.03</td><td rowspan=1 colspan=1>27.12</td><td rowspan=1 colspan=1>75.83</td><td rowspan=1 colspan=2>57.38  35.48</td><td rowspan=1 colspan=1>45.58</td></tr></table>

Table 1 Performance comparison of diferent training methods on five tool-use benchmarks using Qwen3-4B and Qwen3-8B backbones. Training-based baselines are organized into distillation-based ♠ and reinforcement learning ♢ paradigms. All trainable methods are trained on FTRL and evaluated on both the in-domain FTRL test set and four out-of-domain benchmarks. The best and second-best results within each backbone are highlighted.

Implementation Details. We initialize the student model from Qwen3-4B and Qwen3-8B. Teacher trajectories are collected from a mixed pool of Qwen3.6-27B [24] and GLM-5.2 [42]. We perform five iterations of CTB-guided localized SFT over all 2,215 tasks in the FTRL training set. At each iteration, the current student checkpoint generates eight trajectories per task with a temperature of 0.7, a maximum generation length of 4,096 tokens, and at most nine interaction turns. The model is then fine-tuned for one epoch using breakpoint-localized loss masking, with a learning rate of $5 \times 1 0 ^ { - 7 }$ and a batch size of 32. Each augmented training context contains two positive references and one negative reference, and each positive reference is accompanied by a teacher-generated analysis.

## 4.2 Experimental Results

Overall Performance. As shown in Table 1, DART-SD demonstrates robust and superior performance on both the in-domain FTRL dataset and four out-of-domain tool-use benchmarks. Compared with distillation-based methods and reinforcement learning methods, DART-SD delivers consistent empirical gains and achieves the strongest overall performance. This robust cross-domain superiority confirms that our topology-aware localized tuning learns generalizable and transferable tool-use capabilities, rather than merely overfitting to the FTRL training distribution. Crucially, DART-SD maintains strong performance across diferent parameter scales. It achieves the highest average performance with both the Qwen3-8B and Qwen3-4B backbones, while obtaining the best or competitive results on most individual metrics. These results indicate that the efectiveness of topology-aware localized tuning is not limited to a particular model size and remains robust when applied to a smaller backbone.

Effectiveness of Capability Distillation. Figure 3 compares the performance of the base Qwen3-8B student, DART-SD, and the teacher across five tool-use benchmarks. For visualization, we report Solve-F1 on FTRL, the average Multi-Turn score on BFCL, AC on ToolHop, Pass^1 on τ-bench, and PI on RoTBench. Initialized from the same 8B base model, DART-SD delivers substantial improvements across all five benchmarks. Notably, the resulting student surpasses the teacher on FTRL, ToolHop, and τ-bench, despite using substantially fewer parameters. These results suggest that DART-SD goes beyond direct trajectory imitation. By leveraging topology-aware recovery paths, it efectively extracts and generalizes reusable tool-use behaviors from the structural knowledge encoded in the ISTG, enabling the smaller student to outperform its teacher on several benchmarks.

![](images/670327b70227e177c488b0543755c1e0b49a43f9b338a04168924ca7580b37cc.jpg)  
Figure 3 Performance comparison of Qwen3-8B, DART-SD, and the teacher across five tool-use benchmarks. DART-SD improves upon Qwen3-8B on all benchmarks and surpasses the teacher on FTRL, ToolHop, and τ-bench.

<table><tr><td rowspan="2">Task</td><td colspan="5">DART-SD</td><td rowspan="2">Golden</td></tr><tr><td>Iter1</td><td>Iter2</td><td>Iter3</td><td>Iter4</td><td>Iter5</td></tr><tr><td>Single</td><td>1.23</td><td>1.15</td><td>1.10</td><td>1.09</td><td>1.07</td><td>1.00</td></tr><tr><td>Multi</td><td>3.83</td><td>4.63</td><td>4.54</td><td>4.49</td><td>4.41</td><td>4.71</td></tr><tr><td>Para-Single</td><td>3.29</td><td>2.65</td><td>2.27</td><td>2.26</td><td>2.22</td><td>2.11</td></tr><tr><td>Para-Multi</td><td>7.25</td><td>6.48</td><td>5.97</td><td>5.82</td><td>5.62</td><td>6.97</td></tr><tr><td>Overall</td><td>4.23</td><td>3.99</td><td>3.71</td><td>3.65</td><td>3.55</td><td>4.02</td></tr><tr><td>Solve-F1</td><td>40.37</td><td>42.95</td><td>43.78</td><td>44.67</td><td>45.66</td><td></td></tr></table>

Table 2 Average tool-call length of successful trajectories across progressive SFT iterations on the FTRL test set. DART-SD progressively improves Solve-F1 while shortening its tool traces, eventually producing more eficient traces than the golden solutions given during data construction.

Tool-Call Length of Successful Trajectories. To verify whether DART-SD improves reasoning rather than memorizing references, we track the evolution of successful trajectories across iterations. We measure task performance (Solve-F1 ) and eficiency (average tool calls) on the FTRL test set, and compare the final model against the golden references. The golden reference denotes the tool trajectory provided by the FTRL construction pipeline. As shown in Table 2, Solve-F1 steadily improves while trajectory length decreases. The average number of tool calls drops from 4.23 at Iter1 to 3.55 at Iter5. This confirms that the model learns more efficient tool-use strategies instead of relying on brute-force exploration or longer execution chains. This reduction is especially clear in complex multi-step and parallel tasks, suggesting that our method efectively removes redundant tool calls while maintaining high success rates. Remarkably, the trajectories of our final model are even shorter than the golden references (3.55 vs. 4.02). This highlights that DART-SD successfully discovers optimized shortcuts rather than blindly following the provided subtask structures.

CTB Position Shifts within Failed Trajectories. To analyze how the student’s capability boundary expands during progressive training, we track the average CTB position within failed trajectories across iterations. A later CTB position indicates that the model remains within the empirically recoverable region for a longer prefix before its first departure. As shown in Table 3, the average CTB position steadily advances from 0.348 at Iter1 to 1.452 at Iter5, indicating that the first departure from recoverable behavior is progressively delayed. This trend is particularly evident in complex multi-hop and parallel multi-hop tasks, where increasingly longer valid prefixes are learned. These results validate the efectiveness of CTB-guided localized supervision. By identifying the current capability boundary, DART-SD performs teacher-guided resampling from the first departure point and supervises only the recovery sufix while preserving already-mastered prefixes. As the boundary moves deeper across iterations, the student progressively acquires more complex tool-use behaviors, demonstrating the efectiveness of DART-SD for long-horizon tasks.

<table><tr><td rowspan="2">Task</td><td colspan="5">CTB Position</td></tr><tr><td>Iter1</td><td>Iter2</td><td>Iter3</td><td>Iter4</td><td>Iter5</td></tr><tr><td>Single</td><td>0.034</td><td>0.186</td><td>0.137</td><td>0.249</td><td>0.077</td></tr><tr><td>Multi</td><td>0.500</td><td>1.624</td><td>1.729</td><td>1.876</td><td>1.953</td></tr><tr><td>Para-Single</td><td>0.095</td><td>0.219</td><td>0.299</td><td>0.328</td><td>0.313</td></tr><tr><td>Para-Multi</td><td>0.395</td><td>1.524</td><td>1.692</td><td>1.791</td><td>1.816</td></tr><tr><td>Overall</td><td>0.348</td><td>1.185</td><td>1.310</td><td>1.421</td><td>1.452</td></tr><tr><td>∆ vs. Iter1</td><td></td><td>+0.837</td><td>+0.962</td><td>+1.073</td><td>+1.104</td></tr></table>

Table 3 Average CTB positions of failed training trajectories across progressive SFT. Larger values indicate that the first departure from empirically recoverable behavior occurs later, meaning that the model correctly executes a longer trajectory prefix before localized recovery is required.

<table><tr><td>Method</td><td>FTRL</td><td>BFCL</td><td>ToolHop</td></tr><tr><td>Qwen3-8B</td><td>29.74</td><td>40.00</td><td>42.21</td></tr><tr><td>FTRL-GRPO</td><td>32.85</td><td>41.50</td><td>36.72</td></tr><tr><td>ToolRL</td><td>26.72</td><td>34.25</td><td>32.93</td></tr><tr><td>MatchTIR (KM)</td><td>37.33</td><td>47.13</td><td>46.16</td></tr><tr><td>DART-SD</td><td>41.03</td><td>49.75</td><td>46.43</td></tr></table>

Table 4 Performance comparison under the thinking setting. Results are reported using FTRL Solve-F1, the average BFCL Multi-Turn score, and ToolHop AC.

Effect of Thinking Mode. Table 4 evaluates DART-SD under the thinking setting. All trainable baselines are trained with thinking enabled, whereas the Qwen3-8B base model enables thinking only at inference time. For DART-SD, teacher trajectories are generated in thinking mode, and both the teacher’s reasoning traces and tool-call trajectories are incorporated into the resampling context for student training. The efect of explicit thinking varies across methods and benchmarks. To provide a fair and controlled comparison, we therefore use the no-thinking configuration in all other experiments unless otherwise specified, allowing us to more directly evaluate the proposed training framework. Importantly, DART-SD still achieves the best performance across all three benchmarks under the thinking setting, demonstrating that CTB-guided localized supervision remains efective when explicit thinking is enabled.

<table><tr><td>Method</td><td>IFEval</td><td>AIME24</td><td>AIME25</td><td>MMLU</td><td>Avg.</td></tr><tr><td>Qwen3-8B</td><td>34.75</td><td>46.67</td><td>23.33</td><td>70.94</td><td>43.92</td></tr><tr><td>SFT</td><td>35.30</td><td>43.33</td><td>26.67</td><td>71.43</td><td>44.18</td></tr><tr><td>DART-SD</td><td>45.29</td><td>50.00</td><td>30.00</td><td>74.27</td><td>49.89</td></tr></table>

Table 5 General capability evaluation on representative benchmarks. The best results are highlighted.

General Capability Preservation. To evaluate whether DART-SD preserves the model’s general capabilities beyond tool-use tasks, we further evaluate it under a thinking-enabled setting on four standard benchmarks: IFEval [46] for instruction following, AIME24 and AIME25 for mathematical reasoning, and MMLU [10] for general knowledge. We report strict prompt-level accuracy on IFEval, pass@10 on AIME24 and AIME25, and accuracy on MMLU. As shown in Table 5, DART-SD consistently outperforms both the pretrained model and standard SFT across all benchmarks, improving the average score from 43.92 to 49.89. These results suggest that topology-aware localized supervision not only enhances tool-use ability but also preserves the model’s general reasoning and instruction-following capabilities.

<table><tr><td>Method</td><td>Solve-P</td><td>Solve-R</td><td>Solve-F1</td></tr><tr><td>Qwen3-8B</td><td>21.18</td><td>30.71</td><td>23.48</td></tr><tr><td>+SD</td><td>36.23</td><td>44.46</td><td>38.10</td></tr><tr><td>+CTB</td><td>36.62</td><td>46.32</td><td>39.51</td></tr><tr><td>+Progressive SFT</td><td>41.32</td><td>49.65</td><td>43.93</td></tr><tr><td>+ISTG (Ours)</td><td>42.00</td><td>54.13</td><td>45.66</td></tr></table>

Table 6 Component ablation of DART-SD on the FTRL test set using Solve-P, Solve-R, and Solve-F1.

Ablation Study on Different Components. Table 6 presents the component ablation of DART-SD to evaluate the contribution of each core design. First, applying Self-Distillation (SD) directly over the base model yields substantial gains, demonstrating the eficacy of learning from the agent’s own exploratory trajectories. Second, introducing CTB-guided localized supervision further improves performance by restricting the loss calculation exclusively to the post-breakpoint recovery sufix, thereby avoiding direct global supervision on already-mastered prefixes. Third, Progressive Self-Distillation introduces iterative rollout and localized correction rounds, confirming the necessity of a continuous iterative adaptation paradigm over one-shot training. Finally, ISTG replaces LLM-judge-based breakpoint detection with topology-aware state projection and structured recovery, enabling more accurate CTB identification and structurally compatible supervision. Together, these complementary components yield the strongest performance.

## 5 Conclusion

In this paper, we propose DART-SD, a topology-aware self-distillation framework for multi-turn tool-calling agents. DART-SD models execution as an Interaction-State Transition Graph (ISTG), revealing the diamond topology induced by order-independent valid exploration. Building on the ISTG, DART-SD identifies the Critical Topological Breakpoint (CTB), retrieves recovery references from the ISTG, and performs progressive self-distillation via CTB-guided localized supervision while preserving the valid interaction prefix. Extensive experiments demonstrate that DART-SD consistently outperforms both distillation and reinforcement learning baselines across multiple in-domain and out-of-domain tool-use benchmarks. Beyond improving task success, DART-SD enables more eficient tool-use behaviors by preserving exploration, reducing redundant tool calls, and progressively mastering more complex tool-use behaviors throughout self-distillation. More broadly, our findings suggest that efective agent distillation should be guided by interaction-state topology rather than rigid trajectory imitation. We hope this perspective encourages future research on structure-aware agent training and graph representations for long-horizon reasoning and decision-making.

## References

[1] Anthropic. Introducing claude 4, 2025. URL https://www.anthropic.com/news/claude-4.

[2] Huacan Chai, Zijie Cao, Maolin Ran, Yingxuan Yang, Jianghao Lin, Xin Peng, Hairui Wang, Renjie Ding, Ziyu Wan, Muning Wen, et al. Parl-mt: Learning to call functions in multi-turn conversation with progress awareness. arXiv preprint arXiv:2509.23206, 2025.

[3] Zhuohong Chen, Zhengxian Wu, Zirui Liao, Shenao Jiang, Hangrui Xu, Yang Chen, Chaokui Su, Xiaoyu Liu, and Haoqian Wang. R3g: A reasoning-retrieval-reranking framework for vision-centric answer generation. In ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8602–8606, 2026. doi: 10.1109/ICASSP55912.2026.11463453.

[4] Zhuohong Chen, Zhenxian Wu, Yunyao Yu, Hangrui Xu, Zirui Liao, Zhifang Liu, Xiangwen Deng, Pen Jiao, and Haoqian Wang. Learning to search: A decision-based agent for knowledge-based visual question answering. arXiv preprint arXiv:2604.07146, 2026.

[5] Xin Cheng, Shuo He, Lang Feng, HaiYang Xu, Ming Yan, Lei Feng, and Bo An. Beyond trajectory-level attribution: Graph-based credit assignment for agentic reinforcement learning. In International Conference on Machine Learning (ICML), 2026.

[6] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[7] Joan Vendrell Gallart, Solmaz Kia, Russell Bent, and Michael Grosskopf. Chain-based adaptive reconfiguration over lattices for hallucination reduction. arXiv preprint arXiv:2605.27706, 2026.

[8] Zorik Gekhman, Roee Aharoni, Eran Ofek, Mor Geva, Roi Reichart, and Jonathan Herzig. Thinking to recall: How reasoning unlocks parametric knowledge in llms. arXiv preprint arXiv:2603.09906, 2026.

[9] Lakshya Gupta, Litao Li, Yizhe Liu, Sriram Ganapathi Subramanian, Kaheer Suleman, Zichen Zhang, Haoye Lu, and Sumit Pasupalak. World of workflows: a benchmark for bringing world models to enterprise systems. arXiv preprint arXiv:2601.22130, 2026.

[10] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

[11] Hanjiang Hu, Changliu Liu, Na Li, and Yebin Wang. Training task reasoning llm agents for multi-turn task planning via single-turn reinforcement learning. IEEE Control Systems Letters, 9:2879–2884, 2025.

[12] Minki Kang, Jongwon Jeong, Seanie Lee, Jaewoong Cho, and Sung Ju Hwang. Distilling llm agent into small models with retrieval and code tools. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pages 106501–106538, 2026.

[13] Chinmaya Kausik, Adith Swaminathan, and Nathan Kallus. The context gathering decision process: A pomdp framework for agentic search. arXiv preprint arXiv:2605.07042, 2026.

[14] Baohao Liao, Hanze Dong, Christof Monz, Xinxing Xu, Li Dong, and Furu Wei. Multi-turn on-policy distillation with prefix replay. arXiv preprint arXiv:2607.04763, 2026.

[15] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations (ICLR), pages 39578–39601, 2024.

[16] Yuanjie Lyu, Chengyu Wang, Jun Huang, and Tong Xu. From correction to mastery: Reinforced distillation of large language model agents. arXiv preprint arXiv:2509.14257, 2025.

[17] Haoming Meng, Kexin Huang, Shaohang Wei, Chiyu Ma, Shuo Yang, Xue Wang, Guoyin Wang, Bolin Ding, and Jingren Zhou. Sparse but critical: A token-level analysis of distributional shifts in rlvr fine-tuning of llms. arXiv preprint arXiv:2603.22446, 2026.

[18] Seyed Mahed Mousavi, Simone Alghisi, and Giuseppe Riccardi. What does loss optimization actually teach, if anything? knowledge dynamics in continual pre-training of llms. arXiv preprint arXiv:2601.03858, 2026.

[19] Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E Gonzalez. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In International Conference on Machine Learning (ICML), 2025.

[20] Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tur, Gokhan Tur, and Heng Ji. Toolrl: Reward is all tool learning needs. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pages 105523–105553, 2026.

[21] Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Xuanhe Zhou, Yufei Huang, Chaojun Xiao, et al. Tool learning with foundation models. ACM Computing Surveys, 57(4):1–40, 2024.

[22] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. In International Conference on Learning Representations (ICLR), pages 9695–9717, 2024.

[23] Changle Qu, Sunhao Dai, Hengyi Cai, Jun Xu, Shuaiqiang Wang, and Dawei Yin. Matchtir: Fine-grained supervision for tool-integrated reasoning via bipartite matching. arXiv preprint arXiv:2601.10712, 2026.

[24] Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model, April 2026. URL https://qwen.ai/ blog?id=qwen3.6-27b.

[25] Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. Sequence level training with recurrent neural networks. arXiv preprint arXiv:1511.06732, 2015.

[26] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[27] Guoqing Wang, Sunhao Dai, Guangze Ye, Zeyu Gan, Wei Yao, Yong Deng, Xiaofeng Wu, and Zhenzhe Ying. Information gain-based policy optimization: A simple and efective approach for multi-turn llm agents. arXiv preprint arXiv:2510.14967, 2025.

[28] Shengjie Wang, Guanghe Li, Zonghan Yang, and Yang Gao. Hindsight hint distillation: Scafolded reasoning for swe agents from cot-free answers. arXiv preprint arXiv:2605.11556, 2026.

[29] Yunan Wang, Minghui Song, Zihan Zhang, Shaohan Huang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, and Qi Zhang. Group-graph policy optimization for long-horizon agentic reinforcement learning. arXiv preprint arXiv:2606.22995, 2026.

[30] Zhengxian Wu, Kai Shi, Chuanrui Zhang, Zirui Liao, Jun Yang, Ni Yang, Qiuying Peng, Luyuan Zhang, Hangrui Xu, Tianhuang Su, et al. When models judge themselves: Unsupervised self-evolution for multimodal reasoning. arXiv preprint arXiv:2603.21289, 2026.

[31] ZhengXian Wu, Hangrui Xu, Kai Shi, Zhuohong Chen, Yunyao Yu, Chuanrui Zhang, Zirui Liao, Jun Yang, Zhenyu Yang, Haonan Lu, et al. Promsa: Progressive multimodal search agents for knowledge-based visual question answering. arXiv preprint arXiv:2606.27974, 2026.

[32] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al. The rise and potential of large language model based agents: A survey. Science China Information Sciences, 68(2):121101, 2025.

[33] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

[34] Wanli Yang, Hongyu Zang, Junwei Zhang, Wenjie Shi, Du Su, Jingang Wang, Xueqi Cheng, and Fei Sun. Beyond reasoning: Reinforcement learning unlocks parametric knowledge in llms. arXiv preprint arXiv:2605.07153, 2026.

[35] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045, 2024.

[36] Junjie Ye, Yilong Wu, Songyang Gao, Caishuang Huang, Sixian Li, Guanyu Li, Xiaoran Fan, Qi Zhang, Tao Gui, and Xuanjing Huang. Rotbench: A multi-level benchmark for evaluating the robustness of large language models in tool learning. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 313–333, 2024.

[37] Junjie Ye, Yilong Wu, Sixian Li, Yuming Yang, Zhiheng Xi, Tao Gui, Qi Zhang, Xuanjing Huang, Peng Wang, Zhongchao Shi, et al. Tl-training: A task-feature-based framework for training large language models in tool use. arXiv preprint arXiv:2412.15495, 2024.

[38] Junjie Ye, Zhengyin Du, Xuesong Yao, Weijian Lin, Yufei Xu, Zehui Chen, Zaiyuan Wang, Sining Zhu, Zhiheng Xi, Siyu Yuan, et al. Toolhop: A query-driven benchmark for evaluating large language models in multi-hop tool use. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2995–3021, 2025.

[39] Junjie Ye, Changhao Jiang, Zhengyin Du, Yufei Xu, Xuesong Yao, Zhiheng Xi, Xiaoran Fan, Qi Zhang, Tao Gui, Xuanjing Huang, et al. Feedback-driven tool-use improvements in large language models via automated build environments. arXiv preprint arXiv:2508.08791, 2025.

[40] Woongyeng Yeo, Yumin Choi, Taekyung Ki, and Sung Ju Hwang. Hint-sd: Targeted hindsight self-distillation for long-horizon agents. arXiv preprint arXiv:2605.17873, 2026.

[41] Jiahao Yu, Zelei Cheng, Xian Wu, and Xinyu Xing. Building coding agents via entropy-enhanced multi-turn preference optimization. arXiv preprint arXiv:2509.12434, 2025.

[42] Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

[43] Siliang Zeng, Quan Wei, William Brown, Oana Frunza, Yuriy Nevmyvaka, Yang Katie Zhao, and Mingyi Hong. Reinforcing multi-turn reasoning in llm agents via turn-level credit assignment. In ICML 2025 Workshop on Computer Use Agents, 2025.

[44] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

[45] Danna Zheng, Mirella Lapata, and Jef Z Pan. How reliable are llms as knowledge bases? re-thinking facutality and consistency. arXiv preprint arXiv:2407.13578, 2024.

[46] Jefrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.