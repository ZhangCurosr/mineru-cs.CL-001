Mint Agent Technical Report

# Mint-Agent: Introducing Finance-Native Agentic Foundation Models

Mint Agent Team (See the Contributors section for the full author list.)∗

Financial agents must do more than recall domain knowledge: they must be both reliable, executing precise operations over grounded evidence, and executive, sustaining longhorizon research whose conclusions remain auditable. We present Mint-Agent, a family of finance-native agentic models designed around these two scales of financial intelligence. Mint-Agent is built upon three pillars: data, harness, and algorithm. Our data engine constructs clean, specialized tasks for atomic financial capabilities and long-horizon agentic execution from real-world financial sources. MintHarness enables stable interaction with open-ended environments and maintains auditable evidence trails across extended research trajectories. Our training recipe combines SFT, critical-step OPD, and RLVR to develop separate financial reasoning and agentic execution experts, which are then unified through model merging and multi-teacher on-policy distillation into compact, general-purpose financial agents. This pipeline yields two flagship models, Mint-Cu (9B) and Mint-Ag (27B). Across professional financial benchmarks, our models demonstrate two defining strengths: ❶ Reliability: Mint-Ag achieves 98.33% on RFC-Bench, surpassing GPT-5.6- Sol and Claude Opus 4.8 by 3.66 and 3.00 points; and ❷ Executability: Mint-Cu reaches 69.86% on FinSearchComp T2, outperforming Agents-A1-35B and Nex-N2-mini by 22.83 and 12.78 points, while Mint-Ag achieves 76.00% and 60.49% on FinanceAgentBench v1.1 and v2, respectively. These results establish a path toward trustworthy financial intelligence in which domain expertise, long-horizon execution, and auditable evidence are jointly engineered as a unified foundation for frontier agentic models.

 Website: https://mint-fin.github.io/mint-agent

Mint-Agent benchmark results  
![](images/8a39d39be1206d80f761e110f096e7e21438c23ef3213fbbe0820c09acaf19da.jpg)

![](images/900e662cfef95f187d251b1025c791fcc5b54521294063468f479fd76b4f94f9.jpg)  
Mint-Ag Mint-Cu Brand-colored baselines

![](images/c94539c6fb0e69d90644a10e3cebc96704aa9626bc6e11ef17d1f13072115cc3.jpg)

![](images/bcf60f83a14f0d9a98bac2abb3defe47bab280383994332bbbb348216797baac.jpg)

![](images/2ba0223e63e8f105edfa2f74b561477ebb50f33c1d2e5680dd5112edc18f9c43.jpg)

![](images/c090a3d8ae640da7e38ef089054a20b250ca7e9e20ea8bf44be11220fe7b2855.jpg)

![](images/447ad691c881c5a533e393be5488da7b1b4a2f65dd58d4690baca370c55d3726.jpg)  
Figure 1 Overview of the Mint Agent stack. Mint Agent is jointly developed along three complementary dimensions: data, harness, and algorithm. The data engine builds verifiable atomic and long-horizon financial tasks; the harness supports persistent context and multi-source interaction; and the algorithmic pipeline combines supervised fine-tuning, reinforcement learning with verifiable rewards, and on-policy distillation. Together, these components produce the Mint-Cu and Mint-Ag models.

## 1 Introduction

Finance research is a demanding setting for autonomous agents because the answer is rarely a single fact. It often emerges from reading a disclosure in context, identifying the correct reporting period, comparing values across sources, and carrying the arithmetic into a decision. Existing agents can break this chain even when individual steps look plausible: a source may be relevant but not authoritative, a table may be read under the wrong fiscal period, or a computed figure may silently mix units. In finance, such errors are not merely inaccuracies; without a traceable route from source to calculation to conclusion, the result cannot be trusted or repaired.

This work argues that a capable financial agent should be optimized around one shared contract: every answer should be treated as a claim backed by recoverable evidence and reproducible calculation. If tasks are generated without provenance, the model may learn the surface form of financial analysis rather than the discipline behind it. If execution does not preserve state, the most informative failures disappear before they can become supervision. If training only rewards the final text, the decisions that determine whether a research trajectory succeeds remain underspecified. The problem is therefore to make the entire pipeline reinforce the same notion of financial correctness.

To this end, we present Mint-Agent, a series of finance-native agentic models for auditable financial research. Mint-Agent develops capable and reliable financial intelligence through three complementary pillars spanning data, harness, and learning algorithms. (I) Data: Our data engine begins with real-world financial sources and constructs tasks that comprehensively evaluate financial data analysis, calculation, and reasoning. Through evidence crossover, it further synthesizes complex yet well-grounded multi-hop queries for financial research, testing financial tool use, long-horizon execution, and information seeking. (II) Harness: We introduce MintHarness, an evidence-first agent harness designed specifically for financial intelligence. It supports heterogeneous data sources and extended search trajectories while prioritizing evidence collection, thereby making the entire research process transparent and auditable. (III) Algorithm: Our training recipe first develops atomic financial capabilities and long-horizon agentic execution separately through reinforcement learning with verifiable rewards, and then integrates them through multi-teacher on-policy distillation. Together, these three pillars yield two unified financial agents, Mint-Cu and Mint-Ag, with 9B and 27B parameters, respectively. Both models learn financial research as a recoverable sequential decision process rather than isolated answer generation. Despite its compact scale, Mint-Cu surpasses all counterparts in the <20B parameter class, while Mint-Ag achieves performance competitive with substantially larger general-purpose models, including GPT-5.6-Sol and GLM-5.2.

We evaluate Mint-Agent on seven professional financial benchmarks that jointly stress reliable financial reasoning and executable long-horizon research. Across this entire evaluation suite, Mint-Ag consistently outperforms leading frontier models, including GPT-5.6-Sol, Claude Opus 4.8, and GLM-5.2, and attains the highest reported score on every benchmark. Specifically, it achieves 55.71%, 91.33%, and 98.33% on BizFinBench, FinanceBench, and RFC-Bench, together with 76.00% and 60.49% on FinanceAgentBench v1.1 and v2 and 89.04% and 54.07% on FinSearchComp T2 and T3. Meanwhile, the compact Mint-Cu reaches 69.86% on FinSearchComp T2, demonstrating that strong long-horizon financial execution can also be retained at the 9B scale. These results indicate that finance-native agentic models can combine domain reliability, extended execution, and auditability within one vertical intelligence stack.

Our main contributions are summarized as follows:

• Full-stack finance-native intelligence. We introduce a unified data–harness–algorithm framework that connects provenance-grounded task construction, persistent stateful execution, and verifiable policy learning under one recoverable evidence contract.

• Flagship finance-native models. We introduce Mint-Cu (9B) and Mint-Ag (27B), two compact yet capable financial agents that combine reliable domain reasoning, long-horizon execution, and auditable evidence-grounded behavior.

• Comprehensive empirical validation. We evaluate seven professional benchmarks and conduct cost, trajectory, failure-mode, and integration analyses, demonstrating leading financial reasoning, strong long-horizon execution, and auditable evidence-grounded behavior.

## 2 Related Work

## 2.1 Agentic Models

Agentic models are emerging through a gradual transfer of control from hand-written workflows into the model policy: the harness still defines the environment and exposes actions, but posttraining increasingly determines how an interaction unfolds. Tool interaction. What began as schema-constrained invocation is becoming a learned multi-turn policy that selects, sequences, and verifies actions against environmental feedback (Luo et al., 2025; Cheng et al., 2025; Zhao et al., 2025; Chai et al., 2025; Zhou et al., 2026; Zhang et al., 2026c; Li et al., 2025a; Yu et al., 2025). Planning and execution. Credit assignment now reaches beyond isolated calls to the decisions that allocate effort, revise a plan, continue exploration, or terminate, allowing behavior to adapt to the state encountered along a trajectory (Luo et al., 2025; Cheng et al., 2025; Paglieri et al., 2025; Yu et al., 2025; Zhang et al., 2026b,a; Tongyi DeepResearch Team et al., 2025; Hu et al., 2025a). Memory and context. Long-horizon agency similarly shifts context management from passive retention to learned construction: policies decide what to preserve, recover, compress, and revisit as the raw interaction outgrows the context window (Yan et al., 2025; Wang et al., 2025; Shi et al., 2025; Yuan et al., 2025; Yu et al., 2026b; Shen et al., 2025; Lu et al., 2025b; Ye et al., 2025; Wu et al., 2025). Recovery and coordination. Reflection, progress estimation, and delegation extend the same principle to failure repair and multi-agent computation, making recovery paths and subagent allocation part of the policy rather than a fixed orchestration rule (Ma et al., 2026b; Su et al., 2025; Zhou et al., 2026; Huang, 2026; Cai et al., 2026; Yan et al., 2026; Câmara et al., 2026; Tian et al., 2025; Gandhi et al., 2026; Cui et al., 2026). As these abilities are trained separately, their integration becomes a further problem: weight-space methods cheaply compose related checkpoints but must resolve parameter interference, whereas on-policy distillation transfers specialist behavior on states generated by the student itself (Yadav et al., 2023; Patiño et al., 2025; Niu et al., 2026; Ma et al., 2026a). The emerging agent-native model is therefore not independent of its harness; it assumes more responsibility within an execution contract that must still preserve state, constrain actions, and expose failures for learning.

## 2.2 Financial Intelligence

Financial intelligence has progressed from knowing the language of finance toward performing work whose numerical and evidential boundaries are explicit. From domain adaptation to grounded reasoning. Financial pre-training and instruction tuning establish vocabulary, conventions, and broad task coverage, while more recent post-training targets quantitative reasoning and verifiable problem solving (Wu et al., 2023; Yang et al., 2023; Xie et al., 2023; Liu et al., 2025). The corresponding evaluation surface has expanded from calculations over supplied tables and text to disclosure-grounded question answering and the rejection of plausible but unsupported claims (Chen et al., 2021; Zhu et al., 2021; Chen et al., 2022; Islam et al., 2023; Jiang et al., 2026). This progression changes the standard of correctness: selecting the right entity and reporting period, preserving scale and unit, and recovering a valid derivation matter as much as matching the final value. From answers to financial work. A second transition removes the assumption that relevant evidence is already present. Analyst-style tasks require models to discover point-in-time information, reconcile professional documents, and carry quantitative conclusions across multiple sources, while execution-oriented settings test whether financial tools can be selected and applied under domain constraints (Hu et al., 2025b; Bigeard et al., 2025; Zhu et al., 2025; Haque et al., 2026; Li et al., 2026a; Cheng et al., 2026; Wang et al., 2026a; Luan et al., 2026; Lu et al., 2026; Huang et al., 2026; Pauli et al., 2026; Srivastava, 2026; Xiao et al., 2026). The unit of evaluation consequently moves from an isolated answer to a recoverable analyst workflow. This makes auditability a domain requirement rather than a presentation feature: a material conclusion is valid only when its source authority, time boundary, financial semantics, and calculation can be reconstructed, a requirement that neither domain adaptation nor a generic tool scaffold guarantees on its own.

## 2.3 Deep Research

Deep research operationalizes this wider notion of intelligence as an extended process of identifying what remains unknown, acquiring evidence, and revising a conclusion until the research objective is resolved. Two complementary lines have developed. Harness-based systems. Explicit research architectures maintain plans and intermediate state outside the model, coordinate parallel exploration, and assemble evidence before synthesis; this makes the trajectory observable and controllable, although performance remains coupled to a hand-designed topology (Li et al., 2025b; Yang et al., 2026; Zhang et al., 2026d; Cai et al., 2026; MindDR Team and Li Auto Inc, 2026; Yan et al.,

2026; Câmara et al., 2026; Jin et al., 2026). Training-based systems. A second line internalizes the research loop through synthesized interactions and agentic post-training, first learning sustained search from long trajectories and then assigning denser supervision to plans, evidence structures, reflection, and context management (Tongyi DeepResearch Team et al., 2025; Hu et al., 2025a; Li et al., 2026b; Du et al., 2026; Dong et al., 2026; Xu et al., 2026; Xie et al., 2026; Hussain et al., 2026; Zhu et al., 2026b; Yu et al., 2026a; Zhu et al., 2026a; Wu et al., 2025; Lu et al., 2025b; Ye et al., 2025). These routes improve autonomy, but autonomy alone does not make research trustworthy: fluent reports may misattribute a source, cite evidence that does not entail the claim, or follow plausible misinformation into a false conclusion (Venkit et al., 2025; Rasheed et al., 2026; Onweller et al., 2026; Wang et al., 2026b; Zhu et al., 2026c; Nie et al., 2026). Reliable deep research therefore requires the two lines to meet—the policy must learn how to investigate, while the harness must retain enough evidence and computation state to audit what it concludes. Mint-Agent instantiates this joint requirement in finance by grounding task construction, execution, verification, and specialist training in the same recoverable evidence contract.

## 3 Data Engine

The Mint data engine develops financial executability at two scales. Atomic finance concerns a bounded operation whose evidence is already available; long-horizon execution concerns the policy that must discover and compose such evidence through interaction. We synthesize the corresponding streams $\mathcal { D } _ { \mathsf { a t o m } }$ and $\mathcal { D } _ { \mathrm { l o n g } }$ separately, and denote their union by $\mathcal { D } _ { \mathrm { M N T } }$ . Atomic tasks teach the model to complete a well-scoped piece of financial work from bounded evidence, while long-horizon tasks teach it to find that evidence and organize the required operations across an extended trajectory.

## 3.1 Atomic Financial Capabilities

The first objective of the data engine is to build the model’s atomic financial capabilities: the basic skills required to complete a self-contained financial task from a bounded context. We divide them into five groups:

• Knowledge covers financial concepts, conventions, and the meaning of commonly used metrics.

• Extraction identifies the fact requested by a question and recovers it from financial text or tables.

• Calculation applies a valid financial formula and returns a numerically correct result with the appropriate scale and unit.

• Analysis relates reported facts to explain a change, comparison, or financial implication.

• Verification determines whether a financial claim is supported by the given evidence.

Each task is assigned one primary capability $\kappa \in { \mathcal { K } }$ , where K denotes the five groups above. This label controls what the question is intended to test; the financial topic of the question is recorded separately.

Data Sources. We construct these tasks from primary financial materials rather than from existing question-answer datasets. Corporate disclosures are collected from EDGAR and issuer investor-relations archives, with their structured values aligned to the corresponding XBRL facts. Market observations come from exchange records, macroeconomic series from FRED, and financial definitions from the governing accounting taxonomies. Together, these sources form S, while the taxonomies define the semantic environment $\Gamma _ { \mathrm { f i n } }$ used to check metric compatibility. Every source $s \in { \mathcal { S } }$ is stored with a stable locator and its reporting metadata, so any fact used during synthesis can be traced back to the exact document span or structured field from which it was obtained.

![](images/dc5cdf3948d81d26c0af2e0be36798e3fcce1708edbbf1b4b94af5bf34e363e8.jpg)  
Figure 2 Capability-guided construction of atomic financial tasks. A target capability is selected from knowledge, extraction, calculation, analysis, and verification. Primary financial sources are sampled and segmented into semantic chunks, from which structured, provenance-linked facts are extracted. A capability-conditioned constructor generates a question, reasoning trace, and answer, which are retained only after passing checks for evidential support, reproducibility, and answer uniqueness.

Table 1 Sources used to construct atomic financial tasks.
<table><tr><td>Source layer</td><td>Collection</td><td>Role</td></tr><tr><td>Disclosures</td><td>EDGAR and issuer archives</td><td>Grounded financial text and tables</td></tr><tr><td>Structured facts</td><td>XBRL records</td><td>Normalized reported values</td></tr><tr><td>Financial series</td><td>Exchange records and FRED</td><td>Market and macro observations</td></tr><tr><td>Definitions</td><td>Accounting taxonomies</td><td>Metric semantics and valid formulas</td></tr></table>

Task Construction. Starting from $s ,$ we sample a source $s ,$ divide it into semantically coherent chunks, and extract provenance-linked financial facts. Each fact records a source locator together with its entity, reporting period, metric, value, and unit; any fact whose supporting span cannot be recovered is removed. We then sample a target capability $\kappa ,$ select an evidence subset E from a chunk $c ,$ and ask a capability-conditioned constructor $G _ { \phi }$ to generate a question $q ,$ solution trace $r ,$ and answer y. A candidate is retained only if its derivation is fully supported, replayable, and unambiguous:

$$
\begin{array} { r } { \operatorname { s u p p } ( r ) \subseteq E , \qquad \operatorname { E v a l } ( r ; E ) = y , \qquad \left| \operatorname { A n s } ( q ; E ) \right| = 1 . } \end{array}\tag{1}
$$

where supp(r) is the evidence used by the trace, Eval $( r ; E )$ replays its financial operations, and $\operatorname { A n s } ( q ; E )$ is the set of answers admitted by the evidence. These checks anchor generation to the sampled source rather than to the constructor’s parametric knowledge. The retained task is $\mathcal { T } _ { \mathrm { a t o m } } = ( ( c , q ) , y , \mathcal { M } _ { \mathrm { a t o m } } )$ , with ${ \mathcal M } _ { \mathrm { a t o m } } = ( s , E , r , \kappa )$ preserving the provenance and solution structure used for verification.

![](images/35bbd0416f312aa365f275880fa26d19cce42a2c778b98cc620c51e04de72092.jpg)  
Figure 3 Construction and verification of long-horizon agentic tasks. The agent observes only a research objective and must recover the answer through iterative search, inspection, calculation, and synthesis. The underlying source pack and evidence graph remain hidden, preserving the task’s exploratory nature while enabling deterministic verification.

## 3.2 Long-Horizon Agentic Execution

After building the atomic capabilities, we turn to long-horizon execution. A long-horizon task provides a well-defined financial objective but withholds the supporting context, because the model must learn to locate the relevant information and assemble it across an open-ended environment. When the task is instantiated in MintHarness, this information is recovered through an extended interaction with the real-world environment. The data engine therefore constructs each query together with a complete hidden answer graph, allowing that interaction to be trained and evaluated against a result fixed in advance.

Data Sources. Long-horizon synthesis reuses the grounded corpus $s$ from the atomic pipeline and extends it with a time-indexed financial surface $\mathcal { W } _ { \boldsymbol { t _ { \mathrm { c u t } } } } ,$ where $t _ { \mathrm { c u t } }$ denotes the collection cutoff. Each item in $\mathcal { W } _ { \mathrm { { t _ { c u t } } } }$ is stored as a snapshot, so evidence drawn from a changing page or feed remains reproducible after its live version is updated. We also maintain a seed bank $\mathcal { Q } _ { \mathrm { s e e d } }$ built from analyst-written requests after removing all overlap with the evaluation sets.

Evidence Graph Construction. We sample a seed query $q _ { 0 } \in \mathcal { Q } _ { \mathrm { s e e d } }$ and parse it into a task specification $z _ { 0 }$ that preserves the required reasoning structure while discarding the original entities and values. The engine then mines $s \cup \mathcal { W } _ { t _ { \mathrm { c u t } } }$ for a new source pack ${ \mathcal P } _ { 0 , }$ , extracts its grounded facts into $E _ { 0 } ,$ , and composes a directed acyclic evidence graph $\mathbb { G } _ { 0 } = ( V _ { F } \dot { \cup } V _ { O } , A , \rho )$ . Fact nodes $V _ { F }$ contain extracted or derived values, operation nodes $V _ { O }$ encode valid transformations under $\Gamma _ { \mathrm { f i n } } ,$ and the provenance map $\rho$ links every source-backed leaf to its exact locator in $\mathcal { P } _ { 0 }$ . The engine selects a terminal fact $v _ { y }$ and retains its complete ancestor graph only when it meets minimum reasoning-depth and source-breadth requirements:

$$
\mathbb { G } _ { y } = \operatorname { A n c } ( v _ { y } ; \mathbb { G } _ { 0 } ) , \qquad y = \operatorname { V a l } ( v _ { y } ) , \qquad \operatorname { d e p t h } ( \mathbb { G } _ { y } ) \geq h _ { \operatorname* { m i n } } , \qquad \left| \operatorname { S r c } ( \mathbb { G } _ { y } ) \right| \geq b _ { \operatorname* { m i n } } ,\tag{2}
$$

![](images/12ef795229800710d17c5fca7e9c22ad16cdea6f2bc18ccb1299d68bbdcb3fe4.jpg)  
Figure 4 MintHarness overview. Given a financial task and task-specific configuration, MintHarness orchestrates a stateful execution loop in which the agent proposes actions, the harness validates and executes tool calls, and resulting observations update the execution state. Evidence, working memory, and the complete action–observation trajectory are maintained outside the bounded model context. Upon submission, budget exhaustion, or error, an isolated evaluator compares the prediction against hidden reference data and produces a score and auditable execution record.

where Anc $\left( v _ { y } ; \mathbb { G } _ { 0 } \right)$ is the subgraph containing $v _ { y }$ and all nodes required to derive it, $\mathrm { V a l } ( v _ { y } )$ is the value carried by the terminal fact, depth $\left( \mathbb { G } _ { y } \right)$ counts the greatest number of operation nodes on a leaf-to-terminal path, and $\mathrm { S r c } ( \mathbb { G } _ { y } )$ is the set of source records supporting its leaves. The thresholds $h _ { \mathrm { m i n } }$ and $b _ { \mathrm { m i n } }$ impose the minimum reasoning depth and source breadth, respectively. Every operation in $\mathbb { G } _ { y }$ is then replayed. This construction fixes a nontrivial answer and its complete derivation before a new query is written.

Query Realization and Audit. A query constructor $R _ { \psi }$ realizes $q$ from the seed intent $q _ { 0 }$ and the sampled pair $\left( \mathbb { G } _ { y } , y \right)$ : the seed determines the task form, while the graph determines its new financial content. A candidate passes the audit only when replay returns the fixed answer, the answer is unique, every leaf has localized provenance, and the query does not expose its hidden route:

$$
\begin{array} { r } { \operatorname { E v a l } ( \mathrm { G } _ { y } ) = y , \qquad \operatorname { A n s } ( q ; \mathrm { G } _ { y } ) = \{ y \} , \qquad \operatorname { L o c } ( \mathrm { G } _ { y } ) = 1 , \qquad \operatorname { E x p o s e } ( q ; \mathrm { G } _ { y } ) = 0 . } \end{array}\tag{3}
$$

where Loc checks the provenance map and Expose detects leaked locators or intermediate facts. We additionally reject any query that reproduces its seed or overlaps an evaluation task, and require time-sensitive queries to state the boundary $t _ { \mathrm { c u t } }$ under which their answer is valid. The retained task is $\mathcal { T } _ { \mathrm { l o n g } } = ( q , y , \mathcal { M } _ { \mathrm { l o n g } } )$ , where $\mathcal { M } _ { \mathrm { l o n g } } = ( q _ { 0 } , t _ { \mathrm { c u t } } , \mathcal { P } _ { 0 } , \mathbb { G } _ { y } )$ . Only the self-contained objective q is exposed during execution; the answer and audit metadata remain hidden.

## 4 MintHarness

MintHarness instantiates bounded financial questions and open-ended research objectives as a common stateful execution process. It normalizes each task, exposes a task-specific action space, records the interaction, and maintains persistent state outside the model context.

## 4.1 Execution Loop

Task Setup. Let $\xi$ denote the configuration of a task $\tau ,$ specifying its tools, budgets, output format, and evaluator. MintHarness maps $\tau$ to a visible input $x _ { T }$ and an admissible action set $\mathcal { A } _ { \xi }$ . For an atomic task, $x _ { \mathcal { T } } = ( c , q )$ and $\mathcal { A } _ { \xi } = \{ \mathrm { S U B M I T } \} ;$ ; for a long-horizon task, $x _ { \mathcal { T } } = q$ and $\mathcal { A } _ { \xi } = \mathcal { U } _ { \xi } \cup \left\{ \mathrm { S U B M I T } \right\}$ , where $\mathcal { U } _ { \xi }$ contains the configured tools. These objects initialize the harness state:

$$
\begin{array} { r } { \mathbf { S } _ { 0 } = \operatorname { I n i t } _ { \xi } ( T ) \equiv \left( x _ { \mathcal { T } } , \mathcal { A } _ { \xi } , \mathbf { b } _ { 0 } , \mathbf { L } _ { 0 } , \mathbf { W } _ { 0 } , \pmb { \tau } _ { 0 } \right) , } \end{array}\tag{4}
$$

where $\mathbf { b } _ { 0 }$ contains the initial step, tool, and context budgets, while $\mathbf { L } _ { 0 } , \mathbf { W } _ { 0 } ,$ , and $\tau _ { 0 }$ initialize the evidence ledger, working memory, and trajectory as empty objects. The reference answer y and verification metadata M remain on the evaluator side and never enter $\mathbf { S } _ { 0 }$

Action Cycle. At turn $t , \mathrm { P a c k } _ { B } ( \mathbf { S } _ { t } )$ constructs the model-visible context under token budget B from the task, compact working memory, relevant ledger entries, and recent turns. The policy proposes an action from this view, and the harness maps it to either an executable action or an explicit error through the task gate:

$$
a _ { t } \sim \pi _ { \theta } ( \cdot \mid \operatorname { P a c k } _ { B } ( \mathbf { S } _ { t } ) ) , \qquad \bar { a } _ { t } = \operatorname { G a t e } _ { \xi } ( a _ { t } ; \mathbf { S } _ { t } ) ,\tag{5}
$$

where Gate<sub>ξ</sub> checks the action type, tool schema, and remaining execution budget. This validation changes only whether a proposed action can be executed; the research decision itself remains with the policy.

Evidence Update. If $\bar { a } _ { t } \in \mathcal { U } _ { \xi }$ , let $r _ { t } = \mathcal { U } _ { \bar { a } _ { t } } ( \arg s ( \bar { a } _ { t } ) )$ be the raw tool result and set $o _ { t } = r _ { t }$ . MintHarness distills this result into localized financial records before merging them into the ledger:

$$
\begin{array} { r } { \mathbf L _ { t + 1 } = \mathbf { M e r g e } ( \mathbf L _ { t } , \mathrm { D i s t i l l } ( r _ { t } ; x _ { \mathcal T } ) ; \nu _ { t } ) , \bar { a } _ { t } \in \mathcal U _ { \xi } , } \end{array}\tag{6}
$$

where Distill produces typed, provenance-linked financial records, and $\nu _ { t }$ retains acquisition time, tool status, and artifact references. A failed call contributes no evidence, but its error-valued observation remains available to the next turn. Submission and error actions likewise produce an observation $o _ { t }$ while leaving $\mathbf { L } _ { t + 1 } = \mathbf { L } _ { t }$

State Transition. The transition operator appends $\left( \bar { a } _ { t } , o _ { t } \right)$ to the trajectory, advances the remaining budgets, and refreshes the working memory from the updated trajectory and ledger. The ledger remains persistent even when only a selected view is packed into the next prompt:

$$
\begin{array} { r } { { \mathbf { S } } _ { t + 1 } = F _ { \xi } \big ( { \mathbf { S } } _ { t } ; \bar { a } _ { t } , o _ { t } , { \mathbf { L } } _ { t + 1 } \big ) , } \end{array}\tag{7}
$$

where $F _ { \xi }$ is the configured state transition. Equations (5)–(7) repeat until the model submits an answer, the step budget is exhausted, or an unrecoverable environment error occurs.

Completion. At termination, $\hat { y } _ { T }$ is the submitted answer, or ⊥ if execution ends without one. Let $\mathbf { R } _ { T } = \left( \mathbf { L } _ { T } , \pmb { \tau } _ { T } , \mathrm { U s a g e } _ { T } , \mathrm { S t a t u s } _ { T } \right)$ collect the auditable execution record. The task-specific evaluator receives only the extracted prediction and isolated reference answer:

$$
\operatorname { O u t } _ { \xi } ( \mathcal { T } ) = \big ( \hat { y } _ { T } , \operatorname { E v a l } _ { \xi } ( \hat { y } _ { T } ; y ) , \mathbf { R } _ { T } \big ) .\tag{8}
$$

The output includes the prediction, score, and execution record.

![](images/8b23d04df344d395ab4f05bcc7bbfb3986b157e94d3b9a24e4ec79a0923429a1.jpg)  
Figure 5 Twin-specialist training and integration pipeline. Starting from a shared base policy, we train a financial-reasoning specialist on atomic tasks using supervised fine-tuning and reinforcement learning with solution-level verifiers, and an agentic-execution specialist on long-horizon tasks using supervised fine-tuning, critical-step on-policy distillation, and trajectory-level reinforcement learning. The two specialists are first merged with TIES and then integrated through multi-teacher on-policy distillation with origin-matched routing, producing a unified policy instantiated as Mint-Cu and Mint-Ag.

## 4.2 Module Designs

Evidence Ledger. Equation (6) stores distilled facts as persistent typed records with source locators, financial scope, and acquisition metadata. The ledger also records failed retrieval paths separately from supporting evidence, preserving the distinction between missing support and inaccessible sources.

Working Memory. The working memory W<sub>t</sub> is a persistent but compact semantic state that carries the investigation across turns. After each state transition, it is refreshed from the trajectory and evidence ledger to retain the current plan, confirmed conclusions, unresolved questions, and intermediate calculations, with factual claims linked back to their ledger records.

Context Management. Equations (5) and (7) separate durable execution state from the bounded context shown to the model. Recent turns remain verbatim, older interactions are compressed into working memory, and long documents or tables remain accessible through artifact references. The packing function selects the objective, relevant state, and ledger entries for each turn under the token budget.

## 5 Training Pipeline

Starting from a common base policy $\pi _ { \theta _ { 0 } }$ , we train a financial-reasoning specialist on $\mathcal { D } _ { \mathsf { a t o m } }$ and an agentic-execution specialist on $\mathcal { D } _ { \mathrm { l o n g } } ,$ then integrate them into a unified policy.

## 5.1 Financial Reasoning

The financial reasoning branch teaches the model to execute the five atomic capabilities defined in Section 3.1 when the relevant context is already available. Its input remains $( c , q )$ , and its response contains a financial derivation followed by the answer. We train this branch in two stages: SFT first anchors the policy to valid solution forms, after which RLVR explores alternative derivations and optimizes them directly against the hidden evidence and solution structure carried by $\mathcal { M } _ { \mathrm { a t o m } }$

SFT. For every $\mathcal { T } _ { \mathrm { a t o m } } \in \mathcal { D } _ { \mathrm { a t o m } } ,$ , we prompt GLM-5.2 (Z.ai, 2026) and DeepSeek-V4-Pro (DeepSeek-AI, 2026) with $( c , q )$ and sample multiple candidate traces, each containing a complete financial derivation and its final answer. We pass these traces through three gates before using them as supervision. Correctness. The answer must agree with $y ,$ and the derivation must be replayable against E under $\Gamma _ { \mathrm { f i n } }$ . Pattern filtering. Deterministic rules reject a trace when it repeats the same step, breaks before a usable answer, or spends most of its length restating the prompt. Trace selection. If several candidates remain for the same input, a separate LLM judge selects the strongest one under a fixed rubric. The rubric favors a complete financial derivation, penalizes unsupported jumps, and prefers the clearest trace among equally correct alternatives. The selected teacher traces, together with the audited trace r stored by the data engine, form the response targets for one round of supervised fine-tuning from $\pi _ { \theta _ { 0 } }$ . We denote the resulting checkpoint by $\pi _ { \mathrm { R } } ^ { \mathrm { S F T } }$ where the subscript R identifies the financial reasoning branch.

Verifier. With the supervised checkpoint in place, we turn to RLVR. Its first requirement is a verifier that converts the audit information retained by the data engine into a training reward. Let $\boldsymbol { o } = \left( \hat { r } , \hat { y } \right)$ denote a rollout containing a generated derivation rˆ and final answer yˆ. For $\mathcal { T } _ { \mathrm { a t o m } } =$ $\left( ( c , q ) , y , \mathcal { M } _ { \mathrm { a t o m } } \right)$ with $\mathcal { M } _ { \mathrm { a t o m } } = \left( s , E , r , \kappa \right)$ , we define

$$
R _ { \mathrm { a t o m } } ( o , \mathcal { T } _ { \mathrm { a t o m } } ) = \mathsf { A } _ { \kappa } ( \hat { y } , y ) \mathsf { S } ( \hat { r } ; s , E ) \mathsf { C } _ { \kappa } ( \hat { r } , \hat { y } ; r , \Gamma _ { \mathrm { f i n } } ) \in \{ 0 , 1 \} ,\tag{9}
$$

where $\mathsf { A } _ { \kappa }$ is the capability-aware answer-equivalence check, S verifies that every material premise in rˆ is supported by a locator in E and its source s, and ${ \mathsf { C } } _ { \kappa }$ replays the derivation under $\Gamma _ { \mathrm { f i n } } ,$ including its period and unit constraints. The reference trace r specifies the required financial relation but does not impose a particular wording or path, so an alternative derivation receives credit whenever it is grounded, internally consistent, and reaches the verified answer.

RLVR. Given this verifier, we initialize RLVR from $\pi _ { \mathbb { R } } ^ { \mathrm { S F T } }$ and organize the training set as a difficultyaware curriculum. For a task $\mathcal { T } _ { j } \in \mathcal { D } _ { \mathrm { a t o m } }$ with input $( c _ { j } , q _ { j } )$ , we draw ten independent rollouts from the SFT checkpoint and let $p _ { j } ^ { ( 0 ) }$ be the fraction accepted by $R _ { \mathrm { a t o m } }$ . We retain the initial RL pool ${ \mathcal { D } } _ { \mathtt { R } } ^ { ( 0 ) } = \{ T _ { j } : p _ { j } ^ { ( 0 ) } \leq 0 . 8 \}$ ; tasks solved in more than eight trials are already stable and do not enter RL. Difficulty is then refreshed online rather than frozen at this initial estimate. At update $m ,$ the old policy $\pi _ { \theta _ { m - 1 } }$ produces a group of G responses for each candidate task, giving

$$
p _ { j } ^ { ( m ) } = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } R _ { \mathrm { a t o m } } \left( o _ { j , g } , \mathcal { T } _ { j } \right) , \qquad \mathcal { B } _ { m } = \left\{ \mathcal { T } _ { j } \in \mathcal { D } _ { \mathrm { R } } ^ { ( 0 ) } : 0 < p _ { j } ^ { ( m ) } < 1 \right\} .\tag{10}
$$

Here, $o _ { j , g } \sim \pi _ { \theta _ { m - 1 } } ( \cdot \mid c _ { j } , q _ { j } )$ is the g-th rollout, $p _ { j } ^ { ( m ) }$ is the group’s verified pass rate, and $B _ { m }$ is the active batch. Candidate tasks are oversampled until $B _ { m }$ is full; groups with all-zero or all-one rewards are deferred because they provide no relative learning signal. Within each active group, we normalize the verified rewards and optimize them with sequence-level GSPO (Zheng et al., 2025), whose importance ratio is the geometric mean of token likelihood ratios under the current and rollout policies. This makes each update correspond to the complete financial solution scored by the verifier. The optimized checkpoint, denoted $\pi _ { \mathrm { R } } ,$ has therefore learned the atomic capabilities from verified execution rather than from teacher traces alone.

## 5.2 Financial Agentic Execution

Financial agentic execution trains the model to keep working when the evidence required by a query is not given in advance. Within MintHarness, the agent must collect information over an extended interaction, separate useful evidence from noise, and carry the surviving evidence into an answer that can be reproduced and audited. We develop this policy in three stages. SFT first establishes stable interaction behavior; a targeted on-policy distillation stage then repairs consequential decisions exposed by that checkpoint; finally, RLVR optimizes complete trajectories against the hidden answer graph.

SFT. For each $\mathcal { T } _ { \mathrm { l o n g } } = ( q , y , \mathcal { M } _ { \mathrm { l o n g } } )$ in $\scriptstyle { \mathcal { D } } _ { \mathrm { l o n g } } ,$ we use GLM-5.2 and DeepSeek-V4-Pro to execute $q$ in MintHarness and collect trajectories $\boldsymbol { \tau } = \left( a _ { 1 } , o _ { 1 } , \ldots , a _ { T } , o _ { T } , \hat { y } \right)$ , where $a _ { t }$ and $o _ { t }$ are the action and environment observation at turn $t ,$ and $\hat { y }$ is the final answer. We filter this data at two levels. Trajectory gate. A trajectory is retained only when $\hat { y }$ agrees with y and its evidence can be replayed against the source pack $\mathcal { P } _ { 0 }$ and answer graph $\mathbb { G } _ { y }$ stored in $\mathcal { M } _ { \mathrm { l o n g } }$ . Turn gate. Within a retained trajectory, we remove a turn from the SFT loss if it repeats an action without acquiring new evidence, makes a call that contributes nothing to the answer graph, or invokes a tool that does not exist in MintHarness. The surrounding history and observations remain in the conditioning context, so filtering a turn does not break the state seen by later actions. We fine-tune $\pi _ { \theta _ { 0 } }$ only on the retained model actions and final answers; environment observations are never prediction targets. The resulting checkpoint is denoted $\pi _ { \mathrm { A } } ^ { \mathrm { S F T } }$ , where A identifies the agentic branch.

Critical-Step OPD. SFT provides a viable interaction policy, but its failed trajectories reveal decisions for which sparse outcome supervision is too coarse. We roll out $\pi _ { \mathrm { A } } ^ { \mathrm { S F T } }$ on $\mathcal { D } _ { \mathrm { l o n g } }$ and ask an LLM judge to identify critical turns $ { \mathcal { T } } ( \tau ^ { - } )$ in each failed trajectory $\tau ^ { - }$ . A turn is critical when changing its action can restore a valid route to $\mathbb { G } _ { y }$ without altering the preceding interaction. For every selected turn, we preserve the pre-turn state $h _ { t } = ( q , a _ { 1 } , o _ { 1 } , \ldots , a _ { t - 1 } , o _ { t - 1 } )$ ; the current student resamples the next action from $h _ { t } ,$ and a frozen DeepSeek-V4-Pro teacher scores that same state and student continuation. Distillation therefore concentrates on consequential decisions rather than imitating complete teacher trajectories (Ma et al., 2026a).

Because the teacher and student use different tokenizers, we align their tokenizations into minimal chunks with identical decoded text following GOLD-style cross-tokenizer OPD (Patiño et al., 2025; Niu et al., 2026). Let $z _ { t , 1 : L _ { t } }$ be the student action tokens. For a matched chunk $c , C _ { t , c } ^ { S }$ denotes its student-token positions, while $\ell _ { t , c } ^ { \mathsf { S } }$ and $\ell _ { t , c } ^ { \mathrm { T } }$ are the chunk log-likelihoods under the rollout student $\pi _ { \theta _ { \mathrm { o l d } } }$ and teacher $\pi _ { \mathrm { T } }$ . We distribute the teacher’s chunk log-likelihood across the aligned student tokens and define the resulting token-level advantage as

$$
\begin{array} { r l } & { \widetilde { \ell } _ { t , l } ^ { \mathrm { T } } = \frac { \ell _ { t , c } ^ { \mathrm { T } } } { \ell _ { t , c } ^ { \mathrm { S } } } \log \pi _ { \theta _ { \mathrm { o l d } } } ( z _ { t , l } \mid h _ { t } , z _ { t , < l } ) , } \\ & { \widehat { A } _ { t , l } ^ { \mathrm { O P D } } = { s } { \mathrm { g } } \left[ \widetilde { \ell } _ { t , l } ^ { \mathrm { T } } - \log \pi _ { \theta _ { \mathrm { o l d } } } ( z _ { t , l } \mid h _ { t } , z _ { t , < l } ) \right] , \qquad l \in C _ { t , c } ^ { \mathrm { S } } , } \end{array}\tag{11}
$$

where $\widetilde { \ell } _ { t , l } ^ { \mathrm { T } }$ is the teacher target assigned to student token l and sg stops gradients through the advantage. We optimize these advantages over the selected action tokens with the standard clipped

token-level OPD surrogate, using the current-to-rollout student likelihood ratio. This yields $\pi _ { \mathrm { A } } ^ { \mathrm { O P D } }$ correcting the student precisely where its own execution first becomes consequentially wrong.

RLVR. We then optimize $\pi _ { \mathrm { A } } ^ { \mathrm { O P D } }$ through full interactions with MintHarness. For a rollout τ of $\mathcal { T } _ { \mathrm { l o n g } } = ( q , y , \mathcal { M } _ { \mathrm { l o n g } } )$ , where $\mathcal { M } _ { \mathrm { l o n g } } = ( q _ { 0 } , t _ { \mathrm { c u t } } , \mathcal { P } _ { 0 } , \mathbb { G } _ { y } )$ , the trajectory reward is

$$
R _ { \mathrm { l o n g } } ( \tau , \mathcal { T } _ { \mathrm { l o n g } } ) = \mathsf { A } _ { \mathrm { l o n g } } ( \hat { y } , y ) \mathsf { G } ( \tau ; \mathcal { P } _ { 0 } , \mathbb { G } _ { y } ) \mathsf { H } ( \tau ; t _ { \mathrm { c u t } } ) \in \{ 0 , 1 \} ,\tag{12}
$$

where $\mathsf { A } _ { \mathrm { l o n g } }$ checks the final answer, G verifies that the evidence recovered in τ can replay the required path through $\mathbb { G } _ { y }$ with provenance in $\mathcal { P } _ { 0 }$ , and H checks that the interaction uses valid MintHarness actions and respects the collection cutoff $t _ { \mathrm { c u t } }$ . Thus, a correct-looking answer receives no reward when its supporting route cannot be recovered. For every query, the policy samples a group of G complete trajectories. We reuse the active-batch rule in Eq. (10), replacing $R _ { \mathrm { a t o m } }$ with $R _ { \mathrm { l o n g } } ,$ and apply the same sequence-level GSPO update described above. The likelihood ratio covers model-generated reasoning, actions, and the final answer across the complete interaction; environment observations remain conditioning state and are excluded from the loss. The resulting specialist $\pi _ { \mathrm { A } }$ learns to sustain financial research until it produces an answer whose evidence and execution can both be audited.

## 5.3 Expert Integration

We integrate the reasoning expert $\pi _ { \mathrm { R } }$ and execution expert $\pi _ { \mathrm { A } }$ through TIES merging followed by multi-teacher on-policy distillation.

TIES Model Merging. Both experts descend from the common base policy $\pi _ { \theta _ { 0 } }$ . We apply TIES (Yadav et al., 2023) to their task vectors, defined as the parameter displacements of $\pi _ { \mathrm { R } }$ and $\pi _ { \mathrm { A } }$ from that base. TIES first trims low-magnitude coordinates from each task vector, then elects a shared sign at every coordinate and discards expert updates that disagree with it. The remaining aligned updates are averaged, scaled, and added back to the base parameters; coordinates with no surviving update retain their base value. The resulting initialization $\pi _ { \mathrm { M } }$ therefore preserves the experts’ dominant compatible directions while suppressing direct sign conflicts. Weight-space compatibility alone, however, does not guarantee that each expert’s behavior is retained on the states where it matters.

Multi-Teacher On-Policy Distillation. We therefore initialize a student $\pi _ { \theta }$ from $\pi _ { \mathrm { M } }$ and retain $\pi _ { \mathrm { R } }$ and $\pi _ { \mathrm { A } }$ as frozen teachers. MOPD (Ma et al., 2026a) assigns supervision by task origin rather than asking the two teachers to compete on every sample. Specifically, we define the training mixture and its teacher router as

$$
\mathcal { D } _ { \mathrm { m i x } } = \alpha _ { \mathrm { R } } \mathcal { D } _ { \mathrm { a t o m } } + \alpha _ { \mathrm { A } } \mathcal { D } _ { \mathrm { l o n g } } , \qquad \pi _ { T } ^ { \mathrm { T } } = \left\{ \begin{array} { l l } { \pi _ { \mathrm { R } } , \quad T \in \mathcal { D } _ { \mathrm { a t o m } } , } \\ { \pi _ { \mathrm { A } } , \quad T \in \mathcal { D } _ { \mathrm { l o n g } } , } \end{array} \right. \qquad \alpha _ { \mathrm { R } } + \alpha _ { \mathrm { A } } = 1 ,\tag{13}
$$

where $\alpha _ { \mathrm { R } }$ and $\alpha _ { \mathrm { A } }$ are the sampling probabilities of the two empirical task distributions, $\tau$ is a sampled task, and $\pi _ { \mathcal { T } } ^ { \mathrm { { T } } }$ is its routed teacher. Thus, atomic problems receive the financial reasoning signal, whereas long-horizon interactions receive the agentic execution signal. For a sampled task $\tau ,$ the student generates model tokens $z _ { 1 : L }$ on-policy, where L is the rollout length. Let $h _ { t }$ denote the complete conditioning state before token z : it contains the bounded financial input for an atomic task, or the accumulated interaction history and environment observations for a long-horizon task. The routed teacher is prefixed on these same student-generated states, and MOPD minimizes the token-averaged reverse KL from the student to that teacher. Since both specialists share the same base architecture and tokenizer, their distributions align token by token. We optimize the policy-gradient form of this objective; if $\pi _ { \theta } .$ − is the student snapshot that produced the rollout, the teacher–student log-probability difference supplies a dense advantage at every generated token:

$$
\begin{array} { r } { \begin{array} { c } { A _ { t } ^ { \mathrm { M O P D } } = \displaystyle { \mathrm { s g } \biggl [ \log \pi _ { T } ^ { \mathrm { T } } ( z _ { t } \mid h _ { t } ) - \log \pi _ { \theta ^ { - } } ( z _ { t } \mid h _ { t } ) \biggr ] } , } \\ { \mathcal { L } _ { \mathrm { M O P D } } ( \theta ) = \displaystyle { - \mathbb { E } \biggl [ \frac { 1 } { L } \sum _ { t = 1 } ^ { L } \mathrm { c l i p } \Bigl ( A _ { t } ^ { \mathrm { M O P D } } , - A _ { \mathrm { m a x } } , A _ { \mathrm { m a x } } \Bigr ) \log \pi _ { \theta } ( z _ { t } \mid h _ { t } ) \biggr ] } , } \end{array} } \end{array}\tag{14}
$$

where sg stops gradients through the advantage and $A _ { \operatorname* { m a x } } > 0$ bounds its magnitude. Because the states come from the student while the signal comes from the appropriate expert, this stage directly repairs the behavioral interference left by merging without introducing an off-policy imitation gap. Minimizing Eq. (14) yields the unified policy $\pi _ { \mathrm { U } } ,$ , which retains atomic financial reasoning and long-horizon agentic execution in a single checkpoint.

## 6 Experiments

## 6.1 Experimental Setup

Models. Mint-Cu (9B) and Mint-Ag (27B) use Qwen3.5-9B and Qwen3.6-27B as their respective base models. We evaluate them against three groups of baselines. Frontier models include Gemini-3.5-Flash (Google DeepMind, 2026), Claude Opus 4.8 (Anthropic, 2026), GPT-5.6-Sol (OpenAI, 2026b), GLM-5.2 (Z.ai, 2026), MiniMax-M3 (MiniMax, 2026), Kimi-K2.7-Code (Moonshot AI, 2026), DeepSeek-V4-Pro and DeepSeek-V4-Flash (DeepSeek-AI, 2026), MiMo-V2.5-Pro (Xiaomi MiMo Team, 2026), and Qwen3.7-Plus (Qwen Team, 2026). Open-source models include Agents-A1-35B (Bai et al., 2026), Nex-N2-mini (Nex AGI, 2026), ASearcher-32B (Gao et al., 2025), OpenThinkerAgent-32B (Raoof et al., 2026), OpenResearcher-30B-A3B (Li et al., 2026b), and Tongyi-DeepResearch-30B-A3B (Tongyi DeepResearch Team et al., 2025). Agent systems include Codex with GPT-5.6 (OpenAI, 2026a,b), Cursor with Grok 4.5 (Cursor, 2026b), and Cursor with Composer 2.5 (Cursor, 2026a).

Benchmarks. We evaluate two complementary capabilities. Atomic financial reasoning is measured by BizFinBench (Lu et al., 2025a), FinanceBench (Islam et al., 2023), and RFC-Bench Task 2 (Jiang et al., 2026), covering business analysis, disclosure-grounded question answering, and counterfactual-misinformation classification. Agentic execution is measured by FinanceAgent-Bench v1.1 and v2 (Bigeard et al., 2025; Vals AI, 2026) and FinSearchComp T2 and T3 (Hu et al., 2025b), which require multi-step retrieval, calculation, and synthesis. Because no held-out FinanceAgentBench validation split is publicly available, and our inquiry to the benchmark maintainers regarding access to an additional split had not received a response at the time of evaluation, we use the 50 public v1.1 tasks and 27 public v2 tasks. FinSearchComp uses the original released test data. Detailed benchmark-specific verification protocols and judge prompts are provided in Section A.

Protocol. All models are evaluated in a single turn on the three atomic financial-reasoning benchmarks. For FinanceAgentBench v1.1 and $\mathbf { v } 2 ,$ we use the corresponding official evaluation harnesses.<sup>1</sup> For FinSearchComp, we evaluate Mint-Cu, Mint- $\cdot \mathrm { A g } ,$ and the frontier models with MintHarness. Among the open-source baselines, Tongyi DeepResearch<sup>2</sup>, ASearcher<sup>3</sup> and OpenResearcher<sup>4</sup> retain their official inference harnesses, while the remaining open-source models are evaluated with MintHarness. We set the temperature to 0, repetition penalty to 1.1 and a 128K-token context per task. We follow each benchmark’s official answer matching and report accuracy. FinanceAgent Bench v2 is run three times; Table 2 reports the mean and across-run variance.

Table 2 Performance comparison on financial benchmarks. All scores are percentages. FinanceAgentBench v2 reports the mean of three runs; the subscripted ± value is 100× the across-run variance computed on [0, 1] accuracies. Green cells, from darkest to lightest, denote the best, second-best, and third-best result in each column.
<table><tr><td rowspan="2">Model</td><td rowspan="2">BizFin- Bench</td><td rowspan="2">Finance Bench</td><td rowspan="2">RFC Bench</td><td colspan="2">FinanceAgentBench</td><td colspan="2">FinSearchComp</td></tr><tr><td>v1.1</td><td>v2</td><td>T2</td><td>T3</td></tr><tr><td>Frontier models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini-3.5-Flash</td><td>50.71</td><td>85.33</td><td>92.33</td><td>62.00</td><td> $2 5 . 9 3 _ { \pm 0 . 0 9 }$ </td><td>36.53</td><td>44.19</td></tr><tr><td>Claude Opus 4.8</td><td>49.71</td><td>90.00</td><td>95.33</td><td>66.00</td><td> $5 5 . 5 6 _ { \pm 0 . 0 0 }$ </td><td>45.66</td><td>51.16</td></tr><tr><td>GPT-5.6-Sol</td><td>51.57</td><td>86.67</td><td>94.67</td><td>66.00</td><td> $5 6 . 7 9 _ { \pm 0 . 1 2 }$ </td><td>41.10</td><td>50.58</td></tr><tr><td>GLM-5.2</td><td>47.57</td><td>88.00</td><td>94.33</td><td>58.00</td><td> $5 1 . 8 5 _ { \pm 0 . 0 9 }$ </td><td>68.49</td><td>34.88</td></tr><tr><td>MiniMax-M3</td><td>40.71</td><td>86.00</td><td>91.33</td><td>70.00</td><td> $3 8 . 2 7 _ { \pm 0 . 0 3 }$ </td><td>60.27</td><td>30.23</td></tr><tr><td>Kimi-K2.7-Code</td><td>46.57</td><td>82.00</td><td>92.67</td><td>62.00</td><td> $4 3 . 2 1 _ { \pm 0 . 1 2 }$ </td><td>56.62</td><td>25.00</td></tr><tr><td>DeepSeek-V4-Pro</td><td>44.00</td><td>87.33</td><td>91.33</td><td>60.00</td><td> $4 0 . 7 4 _ { \pm 0 . 0 9 }$ </td><td>60.73</td><td>31.40</td></tr><tr><td>DeepSeek-V4-Flash</td><td>45.71</td><td>88.67</td><td>89.67</td><td>72.00</td><td> $4 3 . 2 1 _ { \pm 0 . 0 3 }$ </td><td>59.82</td><td>25.00</td></tr><tr><td>MiMo-V2.5-Pro</td><td>45.00</td><td>87.33</td><td>91.33</td><td>66.00</td><td> $3 7 . 0 4 _ { \pm 0 . 0 9 }$ </td><td>58.45</td><td>26.16</td></tr><tr><td>Qwen3.7-Plus</td><td>48.43</td><td>88.00</td><td>93.33</td><td>64.00</td><td> $4 8 . 1 5 _ { \pm 0 . 0 9 }$ </td><td>62.10</td><td>31.98</td></tr><tr><td>Open-source models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Agents-A1-35B</td><td>41.86</td><td>84.00</td><td>90.00</td><td>42.00</td><td> $1 1 . 1 1 _ { \pm 0 . 0 0 }$ </td><td>47.03</td><td>11.63</td></tr><tr><td>Nex-N2-mini</td><td>27.43</td><td>69.33</td><td>83.33</td><td>38.00</td><td> $7 . 4 1 _ { \pm 0 . 0 9 }$ </td><td>57.08</td><td>20.35</td></tr><tr><td>ASearcher-32B</td><td>28.14</td><td>81.33</td><td>84.00</td><td>24.00</td><td> $3 . 7 0 _ { \pm 0 . 0 0 }$ </td><td>42.92</td><td>4.07</td></tr><tr><td>OpenThinkerAgent-32B</td><td>25.00</td><td>66.67</td><td>71.00</td><td>34.00</td><td> $0 . 0 0 _ { \pm 0 . 0 0 }$ </td><td>36.07</td><td>14.53</td></tr><tr><td>OpenResearcher-30B-A3B</td><td>10.71</td><td>50.00</td><td>22.00</td><td>42.00</td><td> $0 . 0 0 _ { \pm 0 . 0 0 }$ </td><td>53.88</td><td>16.86</td></tr><tr><td>Tongyi-DeepResearch-30B-A3B</td><td>19.57</td><td>42.67</td><td>57.67</td><td>58.00</td><td> $9 . 8 8 _ { \pm 0 . 0 3 }$ </td><td>72.15</td><td>18.02</td></tr><tr><td>Agent systems</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Codex (GPT-5.6)</td><td>46.14</td><td>85.33</td><td>86.33</td><td>66.00</td><td> $5 4 . 3 2 _ { \pm 0 . 0 3 }$ </td><td>80.37</td><td>54.07</td></tr><tr><td>Cursor (Grok 4.5)</td><td>48.29</td><td>88.00</td><td>63.33</td><td>66.00</td><td> $5 3 . 0 9 _ { \pm 0 . 1 2 }$ </td><td>81.74</td><td>52.33</td></tr><tr><td>Cursor (Composer 2.5)</td><td>47.00</td><td>84.67</td><td>69.67</td><td>72.00</td><td> $4 9 . 3 8 _ { \pm 0 . 0 3 }$ </td><td>77.63</td><td>51.16</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mint-Cu</td><td>53.86</td><td>90.00</td><td>96.67</td><td>68.00</td><td> $4 1 . 9 8 _ { \pm 0 . 2 1 }$ </td><td>69.86</td><td>34.88</td></tr><tr><td>Mint-Ag</td><td>55.71</td><td>91.33</td><td>98.33</td><td>76.00</td><td> $6 0 . 4 9 _ { \pm 0 . 0 3 }$ </td><td>89.04</td><td>54.07</td></tr></table>

## 6.2 Overall Performance

Atomic Reasoning. Table 2 shows that the two Mint checkpoints consistently occupy the leading tier on bounded financial tasks. Mint-Ag achieves the best BizFinBench score at 55.71, exceeding the strongest non-Mint result by 4.14 points, and also leads FinanceBench at 91.33 and RFC-Bench at 98.33, improving over the strongest external baselines by 1.33 and 3.00 points, respectively. The joint strength across analysis, disclosure QA, and claim verification indicates that scaling the agentic checkpoint does not trade away precise financial reasoning.

![](images/e9dd91f7c3509bb4537536dbd25273809f46433a38c2d17544ae52c4a2d576b9.jpg)  
(a) FinanceAgentBench v1.1.

![](images/bb1e7c41277390626da5358a6ae5525d3773b37cf68e074cb571a08bfba2bae6.jpg)  
(b) FinanceAgentBench v2.  
Figure 6 Cost–performance trade-offs on FinanceAgentBench. Correct outcomes are plotted against average API cost per question. The dashed vertical line marks the mean cost of the compared models, and the highlighted curve traces the empirical Pareto frontier.

Agentic Execution. The advantage is larger when tasks require open-ended research. Mint-Ag reaches 76.00 on FinanceAgentBench v1.1, 60.49 on v2, and 89.04 on FinSearchComp T2, outperforming the best external scores on these benchmarks—72.00 from DeepSeek-V4-Flash and Cursor (Composer 2.5), 56.79 from GPT-5.6-Sol, and 81.74 from Cursor (Grok 4.5)—by 4.00, 3.70, and 7.30 points. It also reaches the top score of 54.07 on FinSearchComp T3, while the 9B Mint-Cu obtains 69.86 on T2.

## 6.3 Cost Analysis

Pareto Efficiency. Figure 6 jointly compares task accuracy and average API cost per question. Both Mint checkpoints lie on the empirical Pareto frontier in both FinanceAgentBench versions: no evaluated alternative is simultaneously more accurate and less expensive. On v1.1, Mint-Cu anchors the low-cost region with 68.0% accuracy at \$0.016 per task, while Mint-Ag reaches the overall-best 76.0% at \$0.090—72.5% below the \$0.327 comparison mean. On v2, Mint-Cu is again the least expensive point at \$0.069, and Mint-Ag attains the highest accuracy of 60.49% at \$0.213. Relative to the strongest external model, GPT-5.6-Sol, Mint-Ag improves accuracy by 3.70 points while reducing cost by 77.8% (\$0.213 versus \$0.959). Thus, the advantage is not low cost in isolation: the two checkpoints trace complementary low-cost and high-performance operating points on the Pareto frontier. For these locally hosted checkpoints, we estimate API-equivalent cost by applying the prices of their respective base models (Qwen3.5-9B for Mint-Cu and Qwen3.6-27B for Mint-Ag) to measured token usage.

## 6.4 Case Study

We inspect two successful FinanceAgent trajectories covering temporal disambiguation, longhorizon recomputation, and multi-source reconciliation. The cases illustrate how the Evidence Ledger supports these research patterns. An additional trajectory illustrating retrieval recovery and replayable calculation is provided in Section B.

TSMC Forecast. The 37-step trajectory in Figure 7 stresses persistent state over a substantially longer investigation. It encounters blocked filings, wrong-quarter pages, annual datasets, and an initially incorrect four-year averaging window, but retains the valid Q1 2025 guidance and the monthly revenue records already recovered. After restoring the intended 2022–2024 window, the agent recomputes the historical March growth rate, projects March 2025 revenue, and estimates quarterly revenue at NT\$825,110 million. Because the guidance midpoint and exchange-rate conversion remain locked in the ledger, the final NT\$8,010 million miss can be reproduced without trusting the intermediate detours.

![](images/a56cad8fd30961e573872c1115f3d124742de5a60c1a878f1a01f96ff03179d6.jpg)  
Figure 7 TSMC forecast. Persistent evidence and compact working state carry valid guidance and monthly revenue records through 37 steps of source failures, scope corrections, and recomputation.

BKR Acquisition. Figure 8 combines deal terms, accounting inputs, market data, and acquisition rationale. Most fields are recovered early from the announcement materials and GTLS filing, but the unaffected closing price remains unresolved across several blocked or ambiguous sources. The agent keeps those pages as discovery leads rather than treating them as evidence, then locks the exact \$171.65 close from the definitive proxy. With that missing input resolved, the ledger supports a 22.3% premium and reproduces the \$325 million synergy ratios against both revenue and SG&A, while preserving the primary source behind every reported quantity.

![](images/881123dde16cbeffb22065c29c4baad275e30c3605afe7f1bec1ecf8e97fa327.jpg)  
Figure 8 BKR acquisition. Multiple evidence streams are reconciled only after the unaffected closing price is confirmed in the definitive proxy, enabling reproducible premium and synergy calculations.

DeepSeek-V4-Flash

FinanceAgentBench· Error Mode Taxonomy  
![](images/2d5ab8f7917a32f77ade01fb55444cc7995f0e9f8f567847318d5b240598693f.jpg)  
• FINANCEAGENTBENCH v1.1

![](images/112413795810910b86ceb4016f270a2d68489ac2ec17d20fcce3f34cf2c191ae.jpg)

![](images/4883fb496125f1a6e5af3563f38de715a11007755dc6e0094073e7fd1b806afc.jpg)

![](images/107b88a3bb5b6a8bcd916452544fd44e4ee4b4a4edaf78d8894a1b06f894045f.jpg)

![](images/c5721533cbe859d5ef22db62a21ac90a082987bd797a6d02b321c84af9866c13.jpg)  
• FINANCEAGENTBENCH v2

![](images/b9d52eda3aff3580825b0552f44613d820142f1ebed2bf020e42ddc7cd9fd858.jpg)

![](images/3b2d2c1871c963437c8b0fc71075f12f36bcafb87186d015c7c29e9a990a1111.jpg)

![](images/9335b920625bfbfbf5c6c7444779bc92d989ec0e5bc6808bfba6f785e15bb087.jpg)

![](images/7326271a72d4c93d1639c40c51f6369adc77f394542473161b47a46870d31a8a.jpg)  
Figure 9 Failure taxonomy on FinanceAgentBench. Correct outcomes and primary failure modes are shown for representative frontier models and Mint-Cu on v1.1 and v2.

Table 3 Effect of expert integration for the 9B model. All scores are percentages; — denotes a specialist not evaluated outside its target domain.
<table><tr><td>Checkpoint</td><td>Finance Bench</td><td>BizFin Bench</td><td>FAB v1.1</td><td>FinSearch T2</td></tr><tr><td>Reasoning expert (πR)</td><td>88.00</td><td>53.86</td><td></td><td></td></tr><tr><td>Execution expert (πA)</td><td></td><td></td><td>66.21</td><td>72.15</td></tr><tr><td>TIES merge (πM)</td><td>88.67</td><td>50.00</td><td>64.00</td><td>66.67</td></tr><tr><td>MOPD (πU)</td><td>90.00</td><td>53.86</td><td>68.00</td><td>69.86</td></tr></table>

## 6.5 Model Analysis

Failure Modes. To construct the failure taxonomy, PhD researchers with backgrounds in both finance and AI manually review every trajectory and assign its earliest consequential error. Figure 9 summarizes these annotations. On v1.1, Mint-Cu solves 68% of the tasks; its largest residual failure mode is answer omission (M10, 18%), whereas task misunderstanding, retrieval, evidence extraction, and calculation each account for only 2%. On v2, accuracy falls to 41.98% and the dominant error shifts to evidence extraction (M5, 18.5%). Task misunderstanding, financial interpretation, and answer omission each contribute a further 7.9%, while retrieval, calculation, fabrication, and non-termination remain individually small. This shift suggests that the harder benchmark is constrained less by accessing sources than by converting retrieved material into complete, correctly scoped financial evidence.

Effect of MOPD. Table 3 isolates the integration stage, with each specialist evaluated only in its target domain. TIES produces uneven transfer across the four benchmarks, whereas MOPD improves the merged checkpoint by 1.33, 3.86, 4.00, and 3.19 points, respectively. The resulting scores align with the Mint-Cu results in Table 2, showing that MOPD restores performance across both capability groups.

Mechanism Analysis Read together with the failure taxonomy, this specialization-and-integration pattern suggests a structured cognitive architecture. MintHarness first turns observations into an external perceptual memory: the Evidence Ledger preserves what was observed, from which source, and at what time, mitigating memory decay and source confusion over long trajectories. Compact working memory then selects from this durable record only the current goal, unresolved subquestions, and evidence needed for the next decision, limiting overload and attentional drift without duplicating the ledger. Above these memory layers, the reasoning expert supplies a deliberative core for finance-sensitive interpretation and calculation, while the execution expert supplies an executive core for search, tool use, progress monitoring, and termination. MOPD consolidates the two into one policy, retaining their division of labor while reducing skill interference and the coordination cost of switching between separate specialists.

This organization connects the architectural analogy to observed behavior. Separating durable perception from active manipulation helps protect precise reasoning from context noise, while separating deliberation from executive control keeps long-horizon search tied to an answerable objective. The dominant answer-omission failures on v1.1 resemble a breakdown in executive closure: useful intermediate work is not converted into a final response. On v2, the shift toward evidence-extraction failures moves the bottleneck earlier, to the perceptual-memory–workingmemory interface: a source may be reached, yet the financially relevant fact is not selected and encoded with the correct scope. The gains after MOPD indicate that the two cores can be unified without sacrificing their complementary strengths; the remaining taxonomy points instead to selective attention over evidence as the next capability to improve.

## 7 Conclusion

We presented Mint-Agent, a full-stack approach to building finance-native agentic foundation models around a single notion of financial correctness. Its data engine jointly constructs atomic financial capabilities and long-horizon research trajectories from provenance-linked sources; MintHarness turns open-ended interaction into a recoverable execution process through the evidence ledger and context management; and the training stack combines supervised fine-tuning, critical-step OPD, and RLVR before integrating reasoning and execution experts with TIES and multi-teacher OPD. Together, these components connect what the model learns, how it acts, and how its conclusions are verified, rather than treating domain knowledge, tool use, and auditability as separate concerns.

This design is realized in two complementary models. Mint-Cu shows that a compact 9B model can sustain strong financial research performance at favorable inference cost, while Mint-Ag extends the same foundation to more demanding reasoning and execution. Across seven benchmarks, Mint-Ag reaches 98.33 on RFC-Bench, 76.00 and 60.49 on FinanceAgentBench v1.1 and v2, and 89.04 on FinSearchComp T2. The trajectory studies further show that these gains are not limited to final-answer accuracy: the models can preserve evidence, calculations, and intermediate decisions across multi-step investigations.

The broader lesson is that financial intelligence cannot be reduced to producing a plausible answer.

A trustworthy financial agent must be able to recover the evidence it relied on, respect the temporal boundary of the question, expose the transformations that connect facts to claims, and revise its conclusion when any link fails. From this perspective, auditability is not a reporting layer added after inference; it is a shared substrate for data construction, agent execution, and policy learning. Mint-Agent is a step toward that foundation: financial agents whose competence is measured not only by what they conclude, but also by whether each conclusion can be traced and tested. In high-stakes research, intelligence becomes useful at scale only when it leaves such a trail.

## 8 Contributors

Project Lead B. Zhang

Core Contributors Yaze Geng, Lei Tang, Yaoyang Yi, Zonghan Wu

Contributors Yifan Hu

Corresponding authors Kun Wang, Qingsong Wen, Yilei Shao

## References

Anthropic. Claude opus 4.8. Official model page, 2026. https://www.anthropic.com/claude/opus.

Lei Bai, Zongsheng Cao, Yang Chen, Zhiyao Cui, Shangheng Du, Yue Fan, Shiyang Feng, Zijie Guo, Haonan He, Liang He, et al. Scaling the horizon, not the parameters: Reaching trillion-parameter performance with a 35b agent. arXiv preprint arXiv:2606.30616, 2026. https://arxiv.org/abs/2606.30616.

Antoine Bigeard, Langston Nashold, Rayan Krishnan, and Shirley Wu. Finance agent benchmark: Benchmarking LLMs on real-world financial research tasks. arXiv preprint arXiv:2508.00828, 2025. https://arxiv.org/abs/2508.00828.

Yuxuan Cai, Xinyi Lai, Peng Yuan, Weiting Liu, Huajian Li, Mingda Li, Xinghua Wang, Shengxie Zheng, Yanchao Hao, Yuyang Yin, and Zheng Wei. Yunque deepresearch technical report. arXiv preprint arXiv:2601.19578, 2026. https://arxiv.org/abs/2601.19578.

Arthur Câmara, Vincent Slot, and Jakub Zavrel. Self-optimizing multi-agent systems for deep research. arXiv preprint arXiv:2604.02988, 2026. https://arxiv.org/abs/2604.02988.

Jiajun Chai, Guojun Yin, Zekun Xu, Chuhuai Yue, Yi Jia, Siyu Xia, Xiaohan Wang, Jiwen Jiang, Xiaoguang Li, Chengqi Dong, Hang He, and Wei Lin. RLFactory: A plug-and-play reinforcement learning post-training framework for LLM multi-turn tool-use. arXiv preprint arXiv:2509.06980, 2025. https://arxiv.org/abs/2509.06980.

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. FinQA: A dataset of numerical reasoning over financial data. arXiv preprint arXiv:2109.00122, 2021. https://arxiv.org/abs/2109.00122.

Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. ConvFinQA: Exploring the chain of numerical reasoning in conversational finance question answering. arXiv preprint arXiv:2210.03849, 2022. https://arxiv.org/abs/2210.03849.

Mingyue Cheng, Shuo Yu, Daoyu Wang, Qingchuan Li, Xiaoyu Tao, Jie Ouyang, Yucong Luo, Yitong Zhou, Qi Liu, and Enhong Chen. Agent-r1: A unified and modular framework for agentic reinforcement learning. arXiv preprint arXiv:2511.14460, 2025. https://arxiv.org/abs/2511.14460.

Xianfu Cheng, Shiwei Zhang, Jiyu Zhao, Jian Yang, Xinyuan Wang, Ming Zhou, Weixiao Zhou, Xiangyuan Guan, Xiang Li, Zhenhe Wu, Ziyi Ni, Zhoujun Li, and Bingjing Xu. Financecomplexqa: Benchmarking agentic reasoning on industrial-grade financial documents. arXiv preprint arXiv:2607.19238, 2026. https://arxiv.org/abs/2607.19238.

Zhiqing Cui, Haotong Xie, Jiahao Yuan, Cheng Yang, Hanqing Wang, Yuxin Wu, Yifan Wu, Siru Zhong, Tao Yu, Yifu Guo, Siyu Zhang, Xinlei Yu, Qibing Ren, and Usman Naseem. Uno-orchestra: Parsimonious agent routing via selective delegation. arXiv preprint arXiv:2605.05007, 2026. https://arxiv.org/abs/2605.05007.

Cursor. Composer 2.5. Official model page, 2026a. https://cursor.com/composer.

Cursor. Grok 4.5. Official model page, 2026b. https://cursor.com/grok.

DeepSeek-AI. DeepSeek-V4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026. https://arxiv.org/abs/2606.19348.

Yao Dong, Xinglin Xiao, Liwei Dong, Xinlong Jin, Zhengbo Li, Heng Zhang, Duyun Wang, and Nan Xu. S1-DeepResearch: Beyond search, toward real-world long-horizon research agents. arXiv preprint arXiv:2606.15367, 2026. https://arxiv. org/abs/2606.15367.

Yuwen Du, Rui Ye, Shuo Tang, Xinyu Zhu, Yijun Lu, Yuzhu Cai, and Siheng Chen. Openseeker: Democratizing frontier search agents by fully open-sourcing training data. arXiv preprint arXiv:2603.15594, 2026. https://arxiv.org/abs/ 2603.15594.

Apurva Gandhi, Satyaki Chakraborty, Xiangjun Wang, Aviral Kumar, and Graham Neubig. Recursive agent optimization. arXiv preprint arXiv:2605.06639, 2026. https://arxiv.org/abs/2605.06639.

Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, and Yi Wu. Beyond ten turns: Unlocking long-horizon agentic search with large-scale asynchronous RL. arXiv preprint arXiv:2508.07976, 2025. https://arxiv.org/abs/2508.07976.

Google DeepMind. Gemini 3.5: Intelligence built for high-volume work. Official model release, 2026. https://blog. google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/.

Mirazul Haque, Antony Papadimitriou, Samuel Mensah, Zhiqiang Ma, Zhijin Guo, Joy Prakash Sain, Simerjot Kaur, Charese Smiley, and Xiaomo Liu. Deep FinResearch bench: Evaluating AI’s ability to conduct professional financial investment research. arXiv preprint arXiv:2604.21006, 2026. https://arxiv.org/abs/2604.21006.

Chen Hu, Haikuo Du, Heng Wang, Lin Lin, Mingrui Chen, Peng Liu, Ruihang Miao, Tianchi Yue, Wang You, Wei Ji, et al. Step-deepresearch technical report. arXiv preprint arXiv:2512.20491, 2025a. https://arxiv.org/abs/2512.20491.

Liang Hu, Jianpeng Jiao, Jiashuo Liu, Yanle Ren, Zhoufutu Wen, Kaiyuan Zhang, Xuanliang Zhang, Xiang Gao, Tianci He, Fei Hu, et al. Finsearchcomp: Towards a realistic, expert-level evaluation of financial search and reasoning. arXiv preprint arXiv:2509.13160, 2025b. https://arxiv.org/abs/2509.13160.

Caishuang Huang, Yang Qiao, Rongyu Zhang, Junjie Ye, Pu Lu, Wenxi Wu, Meng Zhou, Xiku Du, Tao Gui, Qi Zhang, and Xuanjing Huang. FinToolSyn: A forward synthesis framework for financial tool-use dialogue data with dynamic tool retrieval. arXiv preprint arXiv:2603.24051, 2026. https://arxiv.org/abs/2603.24051.

Fan Huang. ReFlect: An effective harness system for complex long-horizon LLM reasoning. arXiv preprint arXiv:2605.05737, 2026. https://arxiv.org/abs/2605.05737.

Mustafa Anis Hussain, Xinle Wu, and Yao Lu. Planner-centric reinforcement learning for deep research with structureaware reward. arXiv preprint arXiv:2605.30824, 2026. https://arxiv.org/abs/2605.30824.

Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie Vidgen. Financebench: A new benchmark for financial question answering. arXiv preprint arXiv:2311.11944, 2023. https://arxiv.org/abs/2311. 11944.

Yuechen Jiang, Zhiwei Liu, Yupeng Cao, Yueru He, Ziyang Xu, Chen Xu, Zhiyang Deng, Prayag Tiwari, Xi Chen, Alejandro Lopez-Lira, Jimin Huang, Junichi Tsujii, and Sophia Ananiadou. All that glisters is not gold: A benchmark for reference-free counterfactual financial misinformation detection. arXiv preprint arXiv:2601.04160, 2026. https: //arxiv.org/abs/2601.04160.

Jiarui Jin, Zexuan Yan, Shijian Wang, Wenxiang Jiao, and Yuan Lu. AgentDisCo: Towards disentanglement and collaboration in open-ended deep research agents. arXiv preprint arXiv:2605.11732, 2026. https://arxiv.org/abs/ 2605.11732.

Weiya Li, Zhiwei Tang, Yizhou He, Chenghao Wang, Liang Feng, Xiao Sun, Dongrui Liu, Zichen Wen, Hu Wei, Jinghang Wang, Yi Luo, Li Guo, and Linfeng Zhang. ICBCBench: An industry consortium benchmark for financial deep research. arXiv preprint arXiv:2606.17458, 2026a. https://arxiv.org/abs/2606.17458.

Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Guanting Dong, Jiajie Jin, Yinuo Wang, Hao Wang, Yutao Zhu, Ji-Rong Wen, Yuan Lu, and Zhicheng Dou. Deepagent: A general reasoning agent with scalable toolsets. arXiv preprint arXiv:2510.21618, 2025a. https://arxiv.org/abs/2510.21618.

Zhuofeng Li, Dongfu Jiang, Xueguang Ma, Haoxiang Zhang, Ping Nie, Yuyu Zhang, Kai Zou, Jianwen Xie, Yu Zhang, and Wenhu Chen. Openresearcher: A fully open pipeline for long-horizon deep research trajectory synthesis. arXiv preprint arXiv:2603.20278, 2026b. https://arxiv.org/abs/2603.20278.

Zijian Li, Xin Guan, Bo Zhang, Shen Huang, Houquan Zhou, Shaopeng Lai, Ming Yan, Yong Jiang, Pengjun Xie, Fei Huang, Jun Zhang, and Jingren Zhou. Webweaver: Structuring web-scale evidence with dynamic outlines for open-ended deep research. arXiv preprint arXiv:2509.13312, 2025b. https://arxiv.org/abs/2509.13312.

Zhaowei Liu, Xin Guo, Zhi Yang, Fangqi Lou, Lingfeng Zeng, Jinyi Niu, Mengping Li, Qi Qi, Zhiqiang Liu, Yiyang Han, Dongpo Cheng, Ronghao Chen, Huacan Wang, Xingdong Feng, Huixia Judy Wang, Chengchun Shi, and Liwen Zhang. Fin-R1: A large language model for financial reasoning through reinforcement learning. arXiv preprint arXiv:2503.16252, 2025. https://arxiv.org/abs/2503.16252.

Guilong Lu, Xuntao Guo, Rongjunchen Zhang, Wenqiao Zhu, and Ji Liu. Bizfinbench: A business-driven real-world financial benchmark for evaluating LLMs. arXiv preprint arXiv:2505.19457, 2025a. https://arxiv.org/abs/2505.19457.

Jiaxuan Lu, Kong Wang, Yemin Wang, Qingmei Tang, Hongwei Zeng, Xiang Chen, Jiahao Pi, Shujian Deng, Lingzhi Chen, Yi Fu, Kehua Yang, and Xiao Sun. Fintoolbench: Evaluating LLM agents for real-world financial tool use. arXiv preprint arXiv:2603.08262, 2026. https://arxiv.org/abs/2603.08262.

Miao Lu, Weiwei Sun, Weihua Du, Zhan Ling, Xuesong Yao, Kang Liu, and Jiecao Chen. Scaling LLM multi-turn RL with end-to-end summarization-based context management. arXiv preprint arXiv:2510.06727, 2025b. https: //arxiv.org/abs/2510.06727.

Beidi Luan, Rui Sun, Sinuo Wang, Yan Gu, Chao Li, Zhenliang Xiong, Jing Li, and Zuo Bai. FinResearchBench II: A deep research benchmark with consensus-derived gold rubrics for distinguishing financial report quality. arXiv preprint arXiv:2607.12252, 2026. https://arxiv.org/abs/2607.12252.

Xufang Luo, Yuge Zhang, Zhiyuan He, Zilong Wang, Siyun Zhao, Dongsheng Li, Luna K. Qiu, and Yuqing Yang. Agent lightning: Train ANY AI agents with reinforcement learning. arXiv preprint arXiv:2508.03680, 2025. https: //arxiv.org/abs/2508.03680.

Wenhan Ma, Jianyu Wei, Liang Zhao, Hailin Zhang, Bangjun Xiao, Lei Li, Qibin Yang, Bofei Gao, Yudong Wang, Rang Li, Jinhao Dong, Zhifang Sui, and Fuli Luo. MOPD: Multi-teacher on-policy distillation for capability integration in LLM post-training. arXiv preprint arXiv:2606.30406, 2026a. https://arxiv.org/abs/2606.30406.

Xinbei Ma, Congmin Zheng, Jiyang Qiu, Jiale Hong, Yao Yao, Xiangmou Qu, Jiaxin Yin, Xingyu Lou, Jun Wang, Weiwen Liu, Weinan Zhang, Zhuosheng Zhang, and Hai Zhao. Retrospective progress-aware self-refinement for LLM agent training. arXiv preprint arXiv:2606.14302, 2026b. https://arxiv.org/abs/2606.14302.

MindDR Team and Li Auto Inc. Mind deepresearch technical report. arXiv preprint arXiv:2604.14518, 2026. https: //arxiv.org/abs/2604.14518.

MiniMax. MiniMax-M3. Official model page, 2026. https://www.minimax.io/models/text/m3.

Moonshot AI. Kimi k2.7 code. Official model release, 2026. https://www.kimi.com/zh-cn/resources/kimi-k2-7-code.

Nex AGI. Nex-N2-mini. Official model card, 2026. https://huggingface.co/nex-agi/Nex-N2-mini.

Jun Nie, Zhiqin Yang, Zhenheng Tang, Yonggang Zhang, Xiaowen Chu, Xinmei Tian, and Bo Han. DRNOISE: Benchmarking deep research agents in misleading evidence environments. arXiv preprint arXiv:2607.17291, 2026. https://arxiv.org/abs/2607.17291.

Yifan Niu, Han Xiao, Dongyi Liu, Zelong Wang, Dihong Gong, Yasheng Wang, and Jia Li. Breaking the tokenizer barrier: On-policy distillation across model families. arXiv preprint arXiv:2606.09456, 2026. https://arxiv.org/abs/2606.09456.

Hailey Onweller, Elias Lumer, Austin Huber, Pia Ramchandani, Vamse Kumar Subbiah, and Corey Feld. Cited but not verified: Parsing and evaluating source attribution in LLM deep research agents. arXiv preprint arXiv:2605.06635, 2026. https://arxiv.org/abs/2605.06635.

OpenAI. Codex. Official product page, 2026a. https://openai.com/codex/.

OpenAI. GPT-5.6: Frontier intelligence that scales with your ambition. Official model release, 2026b. https://openai. com/index/gpt-5-6/.

Davide Paglieri, Bartłomiej Cupiał, Jonathan Cook, Ulyana Piterbarg, Jens Tuyls, Edward Grefenstette, Jakob Nicolaus Foerster, Jack Parker-Holder, and Tim Rocktäschel. Learning when to plan: Efficiently allocating test-time compute for LLM agents. arXiv preprint arXiv:2509.03581, 2025. https://arxiv.org/abs/2509.03581.

Carlos Miguel Patiño, Kashif Rasul, Quentin Gallouédec, Ben Burtenshaw, Sergio Paniego, Vaibhav Srivastav, Thibaud Frere, Ed Beeching, Lewis Tunstall, Leandro von Werra, and Thomas Wolf. Unlocking on-policy distillation

for any model family. Hugging Face Technical Report, 2025. https://huggingface.co/spaces/HuggingFaceH4/ on-policy-distillation.

Wolfgang M. Pauli, Sarah Panda, Kidus Admassu, Said Bleik, Ademola Okerinde, and Jeremy Reynolds. FORCE-Bench: A benchmark, dataset, and evaluation harness for agentic AI in enterprise finance. arXiv preprint arXiv:2607.19409, 2026. https://arxiv.org/abs/2607.19409.

Qwen Team. Qwen3.7-Plus. Official model page, 2026. https://www.qwencloud.com/models/qwen3.7-plus.

Negin Raoof, Richard Zhuang, Marianna Nezhurina, Etash Guha, Atula Tejaswi, Ryan Marten, Charlie F. Ruan, Tyler Griggs, Alexander Glenn Shaw, Hritik Bansal, et al. OpenThoughts-Agent: Data recipes for agentic models. arXiv preprint arXiv:2606.24855, 2026. https://arxiv.org/abs/2606.24855.

Razeen A. Rasheed, Somnath Banerjee, Animesh Mukherjee, and Rima Hazra. From fluent to verifiable: Claim-level auditability for deep research agents. arXiv preprint arXiv:2602.13855, 2026. https://arxiv.org/abs/2602.13855.

Weizhou Shen, Ziyi Yang, Chenliang Li, Zhiyuan Lu, Miao Peng, Huashan Sun, Yingcheng Shi, Shengyi Liao, Shaopeng Lai, Bo Zhang, Dayiheng Liu, Fei Huang, Jingren Zhou, and Ming Yan. Qwenlong-l1.5: Post-training recipe for long-context reasoning and memory management. arXiv preprint arXiv:2512.12967, 2025. https://arxiv.org/abs/ 2512.12967.

Yaorui Shi, Yuxin Chen, Siyuan Wang, Sihang Li, Hengxing Cai, Qi Gu, Xiang Wang, and An Zhang. Look back to reason forward: Revisitable memory for long-context LLM agents. arXiv preprint arXiv:2509.23040, 2025. https: //arxiv.org/abs/2509.23040.

Rishi Srivastava. CFAgentBench: A reproducible environment and benchmark for autonomous construction-finance agents. arXiv preprint arXiv:2606.22000, 2026. https://arxiv.org/abs/2606.22000.

Junhao Su, Yuanliang Wan, Junwei Yang, Hengyu Shi, Tianyang Han, Junfeng Luo, and Yurui Qiu. Failure makes the agent stronger: Enhancing accuracy through structured reflection for reliable tool interactions. arXiv preprint arXiv:2509.18847, 2025. https://arxiv.org/abs/2509.18847.

Aaron Xuxiang Tian, Ruofan Zhang, Jiayao Tang, Young Min Cho, Xueqian Li, Qiang Yi, Ji Wang, Zhunping Zhang, Danrui Qi, Zekun Li, Xingyu Xiang, Sharath Chandra Guntuku, Lyle Ungar, Tianyu Shi, and Chi Wang. Beyond the strongest LLM: Multi-turn multi-agent orchestration vs. single LLMs on benchmarks. arXiv preprint arXiv:2509.23537, 2025. https://arxiv.org/abs/2509.23537.

Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, et al. Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701, 2025. https://arxiv.org/abs/2510.24701.

Vals AI. Finance agent benchmark v2. Open-source evaluation harness, 2026. https://github.com/vals-ai/ finance-agent-v2.

Pranav Narayanan Venkit, Philippe Laban, Yilun Zhou, Kung-Hsiang Huang, Yixin Mao, and Chien-Sheng Wu. Deeptrace: Auditing deep research AI systems for tracking reliability across citations and evidence. arXiv preprint arXiv:2509.04499, 2025. https://arxiv.org/abs/2509.04499.

Alex Wang, Georg Meinhardt, Jacob Katz, Joseph H. Kim, Pratyush K. Chaudhary, Chase Blagden, and Eric Xu. Bigfinancebench: A workflow-grounded benchmark for financial-research agents. arXiv preprint arXiv:2606.03829, 2026a. https://arxiv.org/abs/2606.03829.

Jiaming Wang, Ziteng Feng, Jiangtao Wu, Ruihao Li, Qianqian Xie, Yuxiang Ren, He Zhu, Xueming Han, Fanyu Meng, Junlan Feng, and Jiaheng Liu. Where do deep-research agents go wrong? span-level error localization in agent trajectories. arXiv preprint arXiv:2606.02060, 2026b. https://arxiv.org/abs/2606.02060.

Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, and Xiaojian Wu. Mem-α: Learning memory construction via reinforcement learning. arXiv preprint arXiv:2509.25911, 2025. https://arxiv.org/abs/2509. 25911.

Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. BloombergGPT: A large language model for finance. arXiv preprint arXiv:2303.17564, 2023. https://arxiv.org/abs/2303.17564.

Xixi Wu, Kuan Li, Yida Zhao, Liwen Zhang, Litu Ou, Huifeng Yin, Zhongwang Zhang, Xinmiao Yu, Dingchu Zhang, Yong Jiang, et al. Resum: Unlocking long-horizon search intelligence via context summarization. arXiv preprint arXiv:2509.13313, 2025. https://arxiv.org/abs/2509.13313.

Yijia Xiao, Rujun Han, Yanfei Chen, Zifeng Wang, Ke Jiang, Zhongying CuiZhu, Vishy Tirumalashetty, Wei Wang, Burak Gokturk, Tomas Pfister, and Chen-Yu Lee. Financeharness: Autonomous financial deep research framework. arXiv preprint arXiv:2607.27853, 2026. https://arxiv.org/abs/2607.27853.

Xiaomi MiMo Team. MiMo-V2.5-Pro. Official model card, 2026. https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro.

Jian Xie, Tianhe Lin, Zilu Wang, Yuting Ning, Yuekun Yao, Tianci Xue, Zhehao Zhang, Zhongyang Li, Kai Zhang, Yufan Wu, Shijie Chen, Boyu Gou, Mingzhe Han, Yifei Wang, Vint Lee, Xinpeng Wei, Xiangjun Wang, Yu Su, and Huan Sun. QUEST: Training frontier deep research agents with fully synthetic tasks. arXiv preprint arXiv:2605.24218, 2026. https://arxiv.org/abs/2605.24218.

Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, and Jimin Huang. PIXIU: A large language model, instruction data and evaluation benchmark for finance. arXiv preprint arXiv:2306.05443, 2023. https://arxiv.org/abs/2306.05443.

Yinuo Xu, Shuo Lu, Jianjie Cheng, Meng Wang, Qianlong Xie, Xingxing Wang, Ran He, and Jian Liang. How to train your deep research agent? prompt, reward, and policy optimization in Search-R1. arXiv preprint arXiv:2602.19526, 2026. https://arxiv.org/abs/2602.19526.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. TIES-Merging: Resolving interference when merging models. In Advances in Neural Information Processing Systems, 2023. https://arxiv.org/abs/2306.01708.

Lingyong Yan, Can Xu, Yukun Zhao, Wenxuan Li, Qingyang Chen, Jiulong Wu, Wenli Song, Xiangnan Li, Weixian Shi, Yiqun Chen, et al. Dumate-deepresearch: An auditable multi-agent system with recursive search and rubric-grounded reasoning. arXiv preprint arXiv:2606.07299, 2026. https://arxiv.org/abs/2606.07299.

Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jeff Z. Pan, Hinrich Sch"utze, Volker Tresp, and Yunpu Ma. Memory-r1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. arXiv preprint arXiv:2508.19828, 2025. https: //arxiv.org/abs/2508.19828.

Hongyang Yang, Xiao-Yang Liu, and Christina Dan Wang. FinGPT: Open-source financial large language models. arXiv preprint arXiv:2306.06031, 2023. https://arxiv.org/abs/2306.06031.

Zhibang Yang, Xinke Jiang, Yuzhen Xiao, Ruizhe Zhang, Yue Fang, XinFei Wan, Zhengxing Song, Yuxuan Liu, Yuheng Huang, Xu Chu, Junfeng Zhao, and Yasha Wang. ScaffoldAgent: Utility-guided dynamic outline optimization for open-ended deep research. arXiv preprint arXiv:2606.20122, 2026. https://arxiv.org/abs/2606.20122.

Rui Ye, Zhongwang Zhang, Kuan Li, Huifeng Yin, Zhengwei Tao, Yida Zhao, Liangcai Su, Liwen Zhang, Zile Qiao, Xinyu Wang, Pengjun Xie, Fei Huang, Siheng Chen, Jingren Zhou, and Yong Jiang. Agentfold: Long-horizon web agents with proactive context management. arXiv preprint arXiv:2510.24699, 2025. https://arxiv.org/abs/2510.24699.

Chengyue Yu, Siyuan Lu, Chenyi Zhuang, Dong Wang, Qintong Wu, Zongyue Li, Runsheng Gan, Chunfeng Wang, Siqi Hou, Gaochi Huang, Wenlong Yan, Lifeng Hong, Aohui Xue, Yanfeng Wang, Jinjie Gu, David Tsai, and Tao Lin. AWorld: Orchestrating the training recipe for agentic AI. arXiv preprint arXiv:2508.20404, 2025. https://arxiv.org/ abs/2508.20404.

Wei Yu, Suxing Liu, Minjie Yu, Jiahao Wang, Zhijian Zheng, Haocheng Deng, and Bing Li. MetaResearcher: Scaling deep research via self-reflective reinforcement learning in adversarial virtual environments. arXiv preprint arXiv:2606.19893, 2026a. https://arxiv.org/abs/2606.19893.

Yi Yu, Liuyi Yao, Yuexiang Xie, Qingquan Tan, Jiaqi Feng, Yaliang Li, and Libing Wu. Agentic memory: Learning unified long-term and short-term memory management for large language model agents. arXiv preprint arXiv:2601.01885, 2026b. https://arxiv.org/abs/2601.01885.

Qianhao Yuan, Jie Lou, Zichao Li, Jiawei Chen, Yaojie Lu, Hongyu Lin, Le Sun, Debing Zhang, and Xianpei Han. Memsearcher: Training LLMs to reason, search and manage memory via end-to-end reinforcement learning. arXiv preprint arXiv:2511.02805, 2025. https://arxiv.org/abs/2511.02805.

Z.ai. GLM-5.2. Official model card, 2026. https://huggingface.co/zai-org/GLM-5.2.

Xuan Zhang, Wenxuan Zhang, See-Kiong Ng, and Yang Deng. Self-evolving world models for LLM agent planning. arXiv preprint arXiv:2606.30639, 2026a. https://arxiv.org/abs/2606.30639.

Xuan Zhang, Zhijian Zhou, Lingfeng Qiao, Yulei Qin, Ke Li, Xing Sun, Xiaoyu Tan, Chao Qu, and Yuan Qi. Internalizing the future: A unified agentic training paradigm for world model planning. arXiv preprint arXiv:2606.27483, 2026b. https://arxiv.org/abs/2606.27483.

Zhen Zhang, Kaiqiang Song, Xun Wang, Yebowen Hu, Weixiang Yan, Chenyang Zhao, Henry Peng Zou, Haoyun Deng, Sathish Reddy Indurthi, Shujian Liu, Simin Ma, Xiaoyang Wang, Xin Eric Wang, and Song Wang. CM2: Reinforcement learning with checklist rewards for multi-turn and multi-step agentic tool use. arXiv preprint arXiv:2602.12268, 2026c. https://arxiv.org/abs/2602.12268.

Zhen Zhang, Liangcai Su, Zhuo Chen, Xiang Lin, Haotian Xu, Simon Shaolei Du, Kaiyu Yang, Bo An, Lidong Bing, and Xinyu Wang. Argus: Evidence assembly for scalable deep research agents. arXiv preprint arXiv:2605.16217, 2026d. https://arxiv.org/abs/2605.16217.

Weikang Zhao, Xili Wang, Chengdi Ma, Lingbin Kong, Zhaohua Yang, Mingxiang Tuo, Xiaowei Shi, Yitao Zhai, and Xunliang Cai. MUA-RL: Multi-turn user-interacting agent reinforcement learning for agentic tool use. arXiv preprint arXiv:2508.18669, 2025. https://arxiv.org/abs/2508.18669.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025. https://arxiv.org/abs/2507.18071.

Yijin Zhou, Linqian Zeng, Xiaoya Lu, Wenyuan Xie, Dongrui Liu, Junchi Yan, and Jing Shao. Exploring agentic tool-calling decisions via uncertainty-aligned reinforcement learning. arXiv preprint arXiv:2606.06976, 2026. https: //arxiv.org/abs/2606.06976.

Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and Tat-Seng Chua. TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in finance. arXiv preprint arXiv:2105.07624, 2021. https://arxiv.org/abs/2105.07624.

Fengbin Zhu, Xiang Yao Ng, Ziyang Liu, Chang Liu, Xianwei Zeng, Chao Wang, Tianhui Tan, Xuan Yao, Pengyang Shao, Min Xu, et al. Findeepresearch: Evaluating deep research agents in rigorous financial analysis. arXiv preprint arXiv:2510.13936, 2025. https://arxiv.org/abs/2510.13936.

Han Zhu, Chengkun Cai, Yuanfeng Song, Xing Chen, Sirui Han, and Yike Guo. Self-evolving deep research via joint generation and evaluation. arXiv preprint arXiv:2606.04507, 2026a. https://arxiv.org/abs/2606.04507.

Minghang Zhu, Chuyang Wei, Junhao Xu, Yilin Cheng, Zhumin Chen, and Jiyan He. DEEPRUBRIC: Evidence-tree rubric supervision for efficient reinforcement learning of deep research agents. arXiv preprint arXiv:2606.17029, 2026b. https://arxiv.org/abs/2606.17029.

Pengyu Zhu, Lijun Li, Longju Yang, Sen Su, and Jing Shao. Is deep research reliable? misleading knowledge induces false conclusions. arXiv preprint arXiv:2607.20891, 2026c. https://arxiv.org/abs/2607.20891.

## A Benchmark Verification Protocols

This appendix documents how model outputs are converted into benchmark scores. We preserve each benchmark’s released evaluator, reference answers, or per-example rubrics rather than introducing a shared judge across datasets. The prompts below omit only lengthy in-prompt demonstrations where noted; the corresponding released prompts are used unchanged during evaluation.

## A.1 BizFinBench

BizFinBench is scored with the official BizFinBench v2 evaluation scripts. The evaluator selects the released comparison function associated with each sample’s task\_type, including anomaly tracing, event-logic reasoning, financial data description, multi-turn perception, quantitative computation, report analysis, stock-price prediction, and sentiment analysis. Our interface extracts the modelvisible final answer and performs only the formatting required by the relevant official script, such as wrapping a scalar answer, JSON array, boxed value, or interval. The normalized prediction and the original sample metadata are then passed to the official evaluator, and a returned score of at least 1.0 is counted as correct. This protocol is deterministic and uses no LLM judge prompt.

## A.2 FinanceBench

FinanceBench is evaluated using its released reference answers and official answer-matching procedure. The evaluator receives the model’s final answer after answer extraction; no additional task-solving context or external evidence is supplied by the evaluator. We report the resulting question-level accuracy.

## A.3 FinSearchComp T2 and T3

For both FinSearchComp T2 and T3, we use the official judge\_system\_prompt and judge\_prompt\_- template stored with each released example. The two tracks use the same judge template. The judge compares the extracted student answer with the reference answer and any accompanying scoring points, then returns a binary answer\_score. Multi-part answers receive credit only when all required elements are correct; numerically equivalent forms are accepted, and an omitted unit is accepted when the question already specifies that unit. Empty or error-valued answers receive zero. The official prompt also contains four demonstrations, which are retained during evaluation but omitted here for space.

## Chinese system prompt used by the evaluator.

你是一个金融问题答案的判断和评分者。给定一道题目、标准答案（可能附有评分要点）和学生答案，先识别学生答案中的最终答案，再与标准答案及评分要点比较。只有全部必要要点均正确时得<sub>1</sub> 分，否则得<sub>0</sub> 分；空答案或异常报错得<sub>0</sub> 分。数值的等价格式视为一致；当题目已经指定单位时，学生答案可以省略单位。不要自行回答或解决题目。输出简短的评分依据，并以<sub>JSON</sub> 字段answer\_score 返回<sub>0</sub> 或<sub>1</sub>。

## English rendering.

You are a judge and scorer of answers to financial questions. Given a question, a reference answer that may include scoring points, and a student answer, first identify the student’s final answer and then compare it with the reference answer and scoring points. Award 1 only if every required element is correct; otherwise award 0. Empty or error-valued answers receive 0. Numerically equivalent formats are treated as equal, and a unit may be omitted when it is specified by the question. Do not solve the question yourself. Return a brief rationale and a JSON object

whose answer\_score is 0 or 1.   
Official user template.   
<<sup>题目</sup>><sup>：</sup> {prompt}   
<<sup>标准答案</sup>><sup>：</sup> {response\_reference}   
<<sup>学生答案</sup>><sup>：</sup> {response}

## A.4 FinanceAgentBench v1.1 and v2

FinanceAgentBench is scored from the correctness rubrics released with each example. An LLM judge receives the question, expert reference answer, model prediction, and ordered list of correctness criteria. It determines whether the prediction satisfies each criterion and separately checks whether any confident factual claim directly contradicts the expert reference. Formatting, ordering, and writing style do not affect a match; reasonable rounding is accepted when units and orders of magnitude agree. A sample is counted as correct only when every correctness criterion is satisfied and no contradiction is detected. If an example contains no correctness criterion, the evaluator falls back to exact, substring, and numeric-overlap answer matching. The same protocol is used for v1.1 and v2.

## Judge system prompt.

You are a strict but fair judge for the Finance Agent Benchmark. You will be given a financial question, an expert reference answer, the agent’s final answer, and a list of rubric criteria. Decide, for each criterion, whether the final answer satisfies it, and separately decide whether the answer contradicts the expert reference on any specific fact. Do not solve the question or use outside knowledge. Return only valid JSON containing the ordered criterion-level decisions, a contradiction decision, and a short overall rationale.

## Judge user template.

[Question] {question}   
[Expert reference answer] {reference}   
[Agent’s final answer] {prediction}   
[Rubric criteria] {rubric\_block}

Illustrative rubric example. The following synthetic example shows the complete criterion-level decision process; it is not an item from the benchmark test set.

[Question]   
Using the company’s FY2024 filing, report its revenue, year-over-year revenue growth, and operating margin.   
[Expert reference answer]   
FY2024 revenue was \$10.2 billion, representing 8.5% year-over-year growth, and operating margin was 14.3%.   
[Agent’s final answer]   
The company reported FY2024 revenue of \$10.2 billion, up 8.5% year over year, with an operating margin of 14.0%.   
[Rubric criteria]   
0. States that FY2024 revenue was \$10.2 billion.   
1. States that FY2024 revenue increased 8.5% year over year.   
2. States that FY2024 operating margin was 14.3%.

```jsonl
[Judge output]
{
"criteria_results": [
{"index": 0, "matched": true, "rationale": "Revenue matches."},
{"index": 1, "matched": true, "rationale": "Growth rate matches."},
{"index": 2, "matched": false, "rationale": "14.0% does not match 14.3%."}
],
"contradiction": {"detected": true, "rationale": "The operating-margin figure conflicts with the
reference."},
"overall_rationale": "Two of three criteria are satisfied, but the operating margin is incorrect."
}
```

Here, the criterion-level rubric score is $2 / 3 ,$ but the benchmark’s binary sample score is 0 because not all correctness criteria are satisfied and the prediction contains a factual contradiction.

## A.5 RFC-Bench Task 2

We evaluate only RFC-Bench Task 2, which asks the model to classify the principal manipulation in a source-grounded counterfactual financial paragraph. This benchmark uses no LLM judge. The evaluator extracts one of four labels—numerical, flipping, sentiment, or causal—from the model-visible final answer, normalizes a small set of label aliases, and performs exact matching against the reference perturbation type. Ambiguous or invalid outputs are counted as incorrect. The reported RFC-Bench result is therefore Task-2 classification accuracy rather than an aggregate over the full RFC-Bench suite.

## Task and output prompt.

RFC-BENCH Task 2: source-grounded counterfactual financial misinformation type classification. Given the source article or page content and a manipulated financial news paragraph, identify which manipulation type was applied. Use only the provided source page and manipulated paragraph; do not browse, retrieve, or use outside evidence. Choose the single label that best captures the main manipulation mechanism: numerical for altered quantitative information, flipping for reversed direction or polarity, sentiment for amplified or weakened evaluative tone, or causal for a distorted cause, driver, attribution, or consequence. Output exactly one of these four labels and no additional text.

## B Additional Case Study

The following case complements the TSMC forecast and BKR acquisition trajectories discussed in Section 6.4. It focuses on retrieval recovery and the reuse of previously collected evidence during a multi-period financial calculation.

AMD Guidance. Figure 10 follows a 13-step trajectory that sizes AMD’s revenue-guidance ranges across four quarters. The initial accession guess fails, after which the agent broadens the SEC search, locates the relevant EX-99.1 exhibits, and records the midpoint and range endpoints for each quarter. The ledger allows the agent to reopen the Q3 source without repeating the full search. The final answer is supported by four primary filings and reports range widths of 10.53%, 8.96%, 8.00%, and 8.45% of midpoint, respectively.

Sizing AMD's Quarterly Revenue Guidance  
![](images/b1242e7db572e793cc47993875edf972a93268d74004814369c703933fdd49a1.jpg)  
Figure 10 AMD guidance. A failed document lookup is replaced by broader SEC discovery, exhibit-level extraction, and a replayable calculation over four quarterly guidance records.