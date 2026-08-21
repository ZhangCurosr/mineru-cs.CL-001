# POLICYGUIDE: From Guarding One Action to Guiding the Whole Workflow for Policy-Compliant LLM Agents

Seongjae Kang<sup>1</sup> Taehyung Yu<sup>1</sup> Sung Ju Hwang<sup>1,2</sup>

<sup>1</sup>KAIST <sup>2</sup>DeepAuto.ai

{tjdwo2744, taehyung.yu, sjhwang}@kaist.ac.kr

## Abstract

Customer-service LLM agents must follow organizational policy when acting on a user’s behalf. Compliance failures arise from either forbidden actions, such as granting an ineligible change, or omitted procedural requirements, such as identification or confirmation. Runtime safeguards can intervene on risky actions, but action-local checks do not guide an agent through a multi-step procedure. Workflowfollowing systems support prescribed process execution, but primarily target workflow completion rather than safeguarding agent behavior. POLICYGUIDE instead compiles each domain policy into a workflow graph and invokes a proactive verifier at user-turn boundaries. From persisted graph state, the verifier reconciles open requests and returns step-specific remediation along a policy-compliant path. Across the τ<sup>2</sup>-bench airline, retail, and telecom domains with a GPT 5.4 agent and verifier, POL-ICYGUIDE raises mean PASS<sup>4</sup> from 0.42 to 0.62, with the largest gain on telecom (0.19 to 0.61), the most workflow-structured domain. The same workflows transfer to Claude Sonnet 4.6 and Gemini 2.5 Pro agents. Complementary evaluations find the lowest observed attacksuccess rate under adversarial users and the strongest procedural compliance in an authordesigned workflow-level validation.

## 1 Introduction

LLM agents are beginning to support customerservice work, including booking flights, modifying orders, and changing account plans through tools on user accounts. These systems typically pair a general-purpose reasoning-andacting loop (Yao et al., 2023) with a frontier model such as GPT 5.4, Claude Sonnet 4.6, or Gemini 2.5 Pro (OpenAI, 2026; Anthropic, 2026; Comanici et al., 2025). Because these models are large and closed-weight, domain-specific fine-tuning is often unavailable or impractical; runtime safeguards offer an integration point that does not require retraining. τ-bench (Yao et al., 2024) and τ<sup>2</sup>-bench (Barres et al., 2025) evaluate this setting against natural-language policies in Airline, Retail, and Telecom.

![](images/4b89fbc7e170d36af7f999bcde53a972022bd951291cf3afb9ebf1e3567874a3.jpg)  
Figure 1: Workflow systems execute prescribed procedures, while external safeguards monitor agent behavior. POLICYGUIDE combines both roles.

Policy compliance depends on both the selected action and the procedure used to reach it. An agent may grant an ineligible change, or it may skip or misorder identification, eligibility checks, and confirmation. Such procedural failures can produce a forbidden outcome or leave an otherwise permissible action unsupported. Our sourcepolicy analysis (Appendices B and B.5) finds that procedural requirements are pervasive (67.4% in airline, ∼100% in retail, and 98.0% in telecom), whereas ordered workflow requirements concentrate in telecom (54.0%, versus 4.7% in airline and 3.6% in retail). Flat prerequisites can often be checked when the agent proposes a guarded action, such as a mutating tool call. Ordered requirements also constrain earlier dialogue and tool-use actions. For example, telecom troubleshooting follows diagnose–instruct–verify sequences that may contain no agent-side mutation for an action guard to intercept.

These two failure modes motivate complementary capabilities (Figure 1). Safeguarding monitors agent behavior and intervenes on risky actions (Zwerdling et al., 2025; Chen et al., 2025; Xiang et al., 2025), whereas workflow enforcement steers execution through required steps (Ye et al., 2025; Wu et al., 2024; Shi et al., 2025b). The two literatures thus target different primary objectives: safe agent behavior and faithful workflow completion. PolicyGuard (Kang et al., 2026) provides the closest connection by incorporating procedural remediation into a mutating-call safeguard, but remains action-triggered and cannot cover earlier deviations outside its guarded action class.

![](images/ec5a6ea97d6ce5f4fa46ef9046ab08861747b4592d0242c8052f56df98db2be8.jpg)  
Figure 2: One task under three enforcement regimes. (a) An action guard checks only the final mutating call, so skipped procedure is discovered late and returned as a block. (b) A workflow/SOP agent drives the procedure, but its checks are designed for faithful workflow execution rather than as a safeguard against policy-violating behavior by a general-purpose agent. (c) POLICYGUIDE runs as an external, advisory guide: at user-turn boundaries it tracks graph position across turns and stops at the first unsatisfied node. If the agent attempts a mutating tool call before completing the workflow, the runtime returns remediation for the unmet step. It recommends the mutation only after the required workflow steps are grounded.

We propose POLICYGUIDE, an external runtime guide for policy-compliant agents (Figure 2). POL-ICYGUIDE compiles each domain policy into a workflow graph. At user-turn boundaries, a proactive verifier traverses the graph from its persisted position, reconciles all open requests, and returns focused remediation for the first unmet step. Persisted state lets the verifier apply workflow safeguards throughout the interaction while coordinating multiple requests. The result is an agentagnostic external overlay that pairs the same procedural representation with different agents. Across the τ<sup>2</sup>-bench airline, retail, and telecom domains with a GPT 5.4 agent and verifier, POLICYGUIDE raises mean $\mathrm { P A S S ^ { 4 } }$ across domains from 0.42 (unguided) to 0.62 (§4.2), with the largest gain on telecom (0.19 to 0.61), the domain whose policy is most workflow-structured. We additionally evaluate diagnostic workflow variants and a matched FlowAgent workflow-controller baseline on telecom (§4.3–4.4). The same workflows transfer to Claude Sonnet 4.6 and Gemini 2.5 Pro agents (§4.5). We further evaluate adversarial robustness with CRAFT red-teaming and workflow compliance with an author-designed Telecom trace audit (§4.6 and §4.7).

## Contributions.

• We characterize policy compliance as a joint safeguarding and workflow-enforcement problem: agents must avoid impermissible actions while completing required procedural steps, including those occurring before or outside a guarded action class.

• We introduce POLICYGUIDE, an external proactive verifier that compiles policies into workflow graphs, tracks multiple open requests across turns, and guides the agent through the required steps.

• We demonstrate 20-point mean PASS<sup>4</sup> gains and cross-agent transfer. Complementary analyses show robustness to CRAFT red-team attacks and stronger workflow compliance.

## 2 Background and Related Work

POLICYGUIDE connects runtime safeguards, which monitor agent behavior, with workflowguided systems, which execute prescribed procedures. It gives persisted workflow state a safeguarding role over agent behavior rather than making workflow control the agent architecture; separating verifier from actor additionally enables reuse across agent runtimes (Figure 1).

## 2.1 τ<sup>2</sup>-bench and policy-adherent agents

$\tau ^ { 2 }$ -bench (Barres et al., 2025), building on τ- bench (Yao et al., 2024), benchmarks policyadherent LLM agents in customer-service domains with a natural-language policy, read-only tools, and mutating tools. We use the airline, retail, and telecom domains: each task is either policy-violation (the agent must refuse) or mutation (the agent must act correctly), and success requires both the final database state and the natural-language assertions to hold. Telecom adds dual control, where some required actions are user-side tools the agent cannot call, making workflow order especially visible. Nearby benchmarks target complementary questions: CRMArena-Pro studies confidentiality compliance rather than ordered procedures (Huang et al., 2025); IntellAgent generates diagnostic tests from policy graphs (Levi and Kadar, 2025); Near-Miss audits failures post hoc (Rabinovich et al., 2026); AgentRewardBench evaluates trajectory judges (Lù et al., 2025); and CRAFT supplies adversarial users rather than a benign workflowcompletion benchmark (Nakash et al., 2025).

## 2.2 Runtime safeguards: action verification

Runtime safeguards are usually action-scoped. ToolGuard compiles tool-level guards, and Solver-Aided checks call constraints with solver support (Winston et al., 2026); both see only the call under check, so process-level requirements are unreachable. PCAS monitors event traces (Palumbo et al., 2026), while ShieldAgent wraps agents with structural safety checks (Chen et al., 2025); both track order but reduce dialogue semantics to predicates or keyword matching. GuardAgent is a single-turn admission controller (Xiang et al., 2025), Tool-Safe classifies unsafe tool use (Mou et al., 2026), and AgentSpec specifies tool-agent safety properties (Wang et al., 2026); these target broader tool-risk settings rather than multi-turn business procedures. AGrail adapts checks online (Luo et al., 2025), Conseca synthesizes just-in-time policies (Tsai and Bagdasarian, 2025), and Progent enforces least privilege over tool arguments (Shi et al., 2025a), but still act locally around risky actions. PolicyGuard (Kang et al., 2026) is the closest action-guard baseline: it reads the full conversation, checks a mutating call against a per-tool checklist, and returns PASS/BLOCK with remediation. This makes it much more dialogue-aware than argumentonly guards, but it remains action-scoped: it does not persist position in a policy workflow or proactively guide the agent through the missing steps before a mutating call is attempted. POLICYGUIDE instead verifies workflow state across turns.

## 2.3 Workflow- and SOP-guided agents

A parallel line encodes procedures as traversable graphs or state machines. SOP-Agent (Ye et al., 2025) compiles a standard operating procedure into a decision graph that restricts actions at each node. StateFlow (Wu et al., 2024) represents a task as a finite-state machine whose states hold prompts and tool calls. SMoT maintains explicit task state (Liu et al., 2023); MetaGPT and ProAgent organize agents around procedural roles or plans (Hong et al., 2024; Ye et al., 2023). JourneyBench studies dynamic prompting in our domain (Balaji et al., 2026); FLAP enforces flows through constrained decoding (Roy et al., 2024); and FlowBench finds that even strong models struggle to follow supplied workflows reliably (Xiao et al., 2024).

These systems primarily study faithful workflow execution rather than safeguarding against policyviolating agent behavior. FlowAgent is the closest qualification (Shi et al., 2025b): its pre- and postdecision controllers guide execution and can reject invalid transitions. Its focus, however, is compliant and flexible workflow execution under out-ofworkflow requests; it is not framed or evaluated as a safeguard against policy-violating agent behavior. POLICYGUIDE instead gives persisted workflow state a safeguarding role: a separate verifier monitors the interaction, returns remediation for unmet steps, and is evaluated with both benign and manipulative users. This separation also permits the same workflow to pair with different agents, a practical benefit rather than the main conceptual distinction.

## 3 Method

POLICYGUIDE combines an offline policy workflow with an external runtime verifier that guides a general-purpose LLM agent (Figure 3). The workflow represents the procedures required by a domain policy, while code persists the verifier’s workflow state and delivers next-step remediation to the agent. Conceptually, POLICYGUIDE is a reference-monitor-inspired runtime safeguard (Anderson, 1972; Schneider, 2000): it observes the interaction and can intervene before policy-sensitive actions, but the evaluated configuration steers execution rather than claiming classical mandatory enforcement. Its repeated judgment over a growing interaction trace is related to runtime verification (Leucker and Schallhart, 2009; Bauer et al., 2011), while its explicit request state is related to dialogue-state tracking (Williams et al., 2013; Henderson et al., 2014).

![](images/3f8c9713edbf3dd03e68a6ccb049685147481006b767aed5dd54c8f72559b832.jpg)

![](images/ce81f5843d34fb5b5ce13681e9224cd413dbaf9d23154ff2686c78a0140ec935.jpg)  
Figure 3: POLICYGUIDE separates offline policy authoring from online enforcement. Offline, the policy and tool registry are compiled, repaired, and validated into a reusable workflow bundle. Online, each verifier call consumes the conversation, grounded tool results, persisted request state, and workflow bundle; it reconciles requests, traverses the graph to the first unmet requirement, and returns remediation, updated state, and mutation-gate status for the next agent turn.

## 3.1 Theoretical motivation

An action-triggered verifier mediates only the actions that invoke it, such as proposed mutating tool calls. This is sufficient only when every reachable first deviation occurs at such an action. Policy workflows, however, can constrain other agent actions, including evidence gathering, user-facing instructions, branch selection, and completion decisions. These deviations matter even when the eventual mutation is permissible, because a later action check cannot undo an already-committed procedural violation. Appendix A formalizes this distinction through intervention coverage. Theorem 1 shows that an ideal binding verifier preserves procedural validity exactly when its firing schedule covers every reachable first deviation. Corollary 1 shows that an ideal workflow-level schedule satisfies this condition, whereas an action-triggered schedule does so only when every first deviation itself triggers the check.

## 3.2 Policy workflow representation

A workflow is the graph of the policy-compliant interaction. In the three generated domains, the main graph begins with a shared intake, identification, and classification path, then enters a requestspecific subflow. Nodes name actors and actions; edges name transitions. Shared procedures such as identification are reused across subflows, while domain-specific subflows express decision gates or diagnostic chains.

Node types are entry/exit (structure), agent\_action (non-tool agent action), user\_input (user response), tool\_call (readonly tool), tool\_authorization (mutating-tool authorization), decision (branch), and subflow (subflow invocation). Each node specification names its actor and expected action and states an explicit satisfying condition that the runtime verifier judges against the interaction (§3.4); subflows are inlined at load time, so the runtime traverses one flat graph per domain. A mutating tool call is enabled at its authorization node and verified from the corresponding tool result.

```latex
Algorithm 1: POLICYGUIDE verifier
Input : policy π, tools T, workflow G, history H,
and state S
$\widehat { \mathcal { R } } \gets$ requests in H reconciled with the tracked
requests in S
foreach open request $\boldsymbol { r } \in \widehat { \mathcal { R } }$ do
p ← entry of G if r is new; otherwise its position
in S
while p is nonterminal do
if the requirement at p is not satisfied by H
then
break
p ← successor along the outgoing edge in G
that matches H
if p is terminal then
d<sub>r</sub> ← ∅
else
d<sub>r</sub> ← action required at p
d ← merge $\{ d _ { r } : r \in \widehat { \mathcal { R } } \}$
return (d, Rb)
```

## 3.3 Offline workflow generation

The workflows are generated offline by a multistage pipeline and frozen once per domain for all workflow-based conditions (Figure 3, top). Stage 1 extracts tool specifications and mutating tools, excluding user-device actions. Stage 2 derives request types, shared procedures, ordered subflows, and a coverage audit; Stage 3 reviews the plan. Stage 4 generates and schema-validates subflows (one repair retry and branch review), and Stage 5 connects the intake spine, classifier, and subflows. Stage 6 validates schema conformance, tool inventory, mutating-tool authorization coverage, graph composition, edge arity, and reachability; reviews policy-to-graph mappings; and prunes unused subflows. Appendix G reproduces the prompts, Appendix H shows examples, and Appendix E specifies these checks and reports the remaining semantic-audit scope.

## 3.4 Online policy-guided runtime

Algorithm 1 summarizes the runtime (Figure 3, bottom). Each firing is a single verifier generation

$$
{ V } _ { \phi } ( \pi , { \mathcal T } , G , H , S ) = ( d , \widehat { \mathcal { R } } ) ,
$$

where $V _ { \phi }$ is the verifier $\operatorname { L L M } ; \pi , { \mathcal { T } } , G , H ,$ and S are the raw policy, tool specifications, frozen workflow graph, interaction history, and code-owned request state; and d and Rb are the merged remediation and the updated request records the runtime persists.

Firing and interface. The verifier fires before the agent responds to each user turn, and once more after a mutating tool call not authorized by the current workflow state is intercepted; skipped tool-result turns are folded into the next firing’s conversation delta, so each call judges the complete trajectory. Its prompt is a cached static prefix (π, T , the rendered graph, judging rules, and output contract) plus the conversation and the latest state record; it returns a free-text audit and one structured record per open request—node walk with cited evidence, position, status, mutating-tool authorization, selection memory, and remediation—plus a global transfer flag (Appendices D.3 and G). It runs at temperature 0, model-paired with the agent.

Reconcile and traverse. The verifier reconciles the open requests against S (continuing requests keep their recorded positions; new ones open at the graph entry; abandoned or duplicate ones are dropped or merged), then walks each from its recorded node, judging every node’s satisfying condition against H: facts and eligibility count only when confirmed by tool results, while the user’s own choices and consent count from their messages. The walk stops at the first unsatisfied node, whose required action becomes the remediation; one generation can advance several nodes, and terminal nodes mark a request done.

State and delivery. Code, rather than the model’s conversational memory, owns state persistence. It rejects unknown node IDs, filters authorization outputs against the enumerated mutating-tool inventory, reconstructs the currently enabled tool set, and persists each request’s position and memory. The merged remediation is injected as a guidance message before the agent acts.

Intervention. In the evaluated advisory mode, the first mutating tool call not authorized by the current workflow state within each user-turn region is intercepted before execution and triggers a corrective verifier firing. The one-shot gate then disarms for an immediate retry, preventing the advisory mechanism from deadlocking execution. Other workflow-governed actions are steered through remediation rather than hard-gated.

<table><tr><td colspan="4"></td><td colspan="3">Airline (50)</td><td colspan="3">Retail (114)</td><td colspan="3">Telecom (114)</td></tr><tr><td colspan="2">System</td><td>Verifier</td><td>Overall</td><td>PV</td><td>Mut</td><td>Overall</td><td>PV</td><td>Mut</td><td>Overall</td><td>PV</td><td></td><td>Mut</td></tr><tr><td rowspan="4">PAS1</td><td>ReAct</td><td></td><td>0.640</td><td>0.865</td><td>0.433</td><td>0.800</td><td>0.900</td><td></td><td>0.791</td><td>0.384</td><td>0.721</td><td>0.180</td></tr><tr><td>ToolGuard</td><td>static code</td><td>0.575</td><td>0.969</td><td>0.212</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PolicyGuard</td><td>GPT 5.4</td><td>0.710</td><td>1.000</td><td>0.442</td><td>0.645</td><td>0.975</td><td>0.613</td><td></td><td>0.406</td><td>0.733</td><td>0.208</td></tr><tr><td>POLICYGUIDE</td><td>GPT 5.4</td><td>0.775</td><td>0.979</td><td>0.587</td><td>0.809</td><td>0.975</td><td>0.793</td><td></td><td>0.866</td><td>0.895</td><td>0.849</td></tr><tr><td rowspan="4">PAS</td><td>ReAct</td><td></td><td>0.460</td><td>0.750</td><td>0.192</td><td>0.596</td><td>0.700</td><td>0.587</td><td></td><td>0.193</td><td>0.442</td><td>0.042</td></tr><tr><td>ToolGuard</td><td>static code</td><td>0.520</td><td>0.875</td><td>0.192</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PolicyGuard</td><td>GPT 5.4</td><td>0.580</td><td>1.000</td><td>0.192</td><td>0.360</td><td>0.900</td><td>0.308</td><td></td><td>0.202</td><td>0.488</td><td>0.028</td></tr><tr><td>POLICYGUIDE</td><td>GPT 5.4</td><td>0.620</td><td>0.917</td><td>0.346</td><td>0.614</td><td>0.900</td><td>0.587</td><td></td><td>0.614</td><td>0.721</td><td>0.549</td></tr></table>

Table 1: Main results on the base splits (GPT 5.4 agent, n=4; airline $5 0 ,$ retail/telecom 114 tasks). Cells report $\mathrm { P a s s ^ { 1 } }$ and $\mathrm { P a s s ^ { 4 } }$ overall and on the PV/Mut slices. The verifier is absent for ReAct, static code for ToolGuard, and GPT 5.4 for PolicyGuard and POLICYGUIDE.

![](images/9cd63648fbb798dba20b0cff31925108bd14c901572955f76afa9ea6d7bc7c97.jpg)  
Figure 4: $\mathrm { P a s s } ^ { k }$ vs. k (reliability; higher is better) on the base split of each domain, GPT 5.4, all cells $n { = } 4$

## 3.5 Variants and ablations

POLICYGUIDE rests on two separable ingredients: what the policy is compiled into (the graph versus the raw policy text) and who tracks progress (an external verifier versus the acting agent). Each variant strips one. POLICYGUIDE-RAW keeps the verifier model, firing schedule, carried state, and remediation channel but substitutes the raw policy for G, so no graph position persists—isolating the compiled graph. POLICYGUIDE-SELF places the frozen graph in the actor’s system prompt but removes the external verifier, code-owned state, per-turn remediation, and corrective intercept—isolating external tracking (§4.3).

## 4 Experiments

## 4.1 Setup

We evaluate on the τ<sup>2</sup>-bench Airline (50 tasks; 24 PV / 26 Mut), Retail (114; 10 PV / 104 Mut), and Telecom (114; 43 PV / 71 Mut) base splits. PV tasks require the agent to prevent a policy-violating mutation; Mut tasks require it to complete a permitted mutation under the policy prerequisites. Diagnostic variants and FlowAgent use the benchmark-provided held-out test splits for Retail (40; 4 PV / 36 Mut) and Telecom (40; 21 PV / 19 Mut): the fixed IDs come from split\_tasks.json, not author sampling. Telecom test is more PV-heavy than base (52.5% versus 37.7%), so we report both slices. We evaluate model-paired actor–verifier configurations using GPT 5.4, Claude Sonnet 4.6, and Gemini 2.5 Pro, with the verifier drawn from the actor’s model family; the frozen user simulator is GPT 4.1. The domain-wide comparison uses GPT 5.4. To keep policy representation consistent, we use GPT 5.4 to author one workflow per domain and reuse each frozen workflow across systems and agent families. This isolates runtime and executor differences from

<table><tr><td>Domain</td><td>Metric</td><td>ReAct</td><td>POLICYGUIDE SELF</td><td>POLICYGUIDE RAW</td><td>POLICYGUIDE</td></tr><tr><td rowspan="3">Airline</td><td>Overall</td><td>0.460</td><td>0.480</td><td>0.520</td><td>0.620</td></tr><tr><td>PV</td><td>0.750</td><td>0.833</td><td>0.875</td><td>0.917</td></tr><tr><td>Mut</td><td>0.192</td><td>0.154</td><td>0.192</td><td>0.346</td></tr><tr><td rowspan="3">Retail</td><td>Overall</td><td>0.575</td><td>0.350</td><td>0.575</td><td>0.725</td></tr><tr><td>PV</td><td>0.750</td><td>0.750</td><td>0.750</td><td>1.000</td></tr><tr><td>Mut</td><td>0.556</td><td>0.306</td><td>0.556</td><td>0.694</td></tr><tr><td rowspan="3">Telecom</td><td>Overall</td><td>0.250</td><td>0.325</td><td>0.350</td><td>0.675</td></tr><tr><td>PV</td><td>0.429</td><td>0.571</td><td>0.619</td><td>0.667</td></tr><tr><td>Mut</td><td>0.053</td><td>0.053</td><td>0.053</td><td>0.684</td></tr></table>

Table 2: Workflow ablations (GPT 5.4 agent; Airline base split, Retail and Telecom benchmark test splits of 40 tasks). All cells report Pass<sup>4</sup>.
<table><tr><td>System</td><td>Runtime control</td><td> $\mathrm { P a s s ^ { 4 } }$ </td></tr><tr><td>ReAct</td><td>actor only</td><td>0.250</td></tr><tr><td>PolicyGuard</td><td>action-local check</td><td>0.325</td></tr><tr><td>FlowAgent</td><td>PDL + API control</td><td>0.350</td></tr><tr><td>POLICYGUIDE</td><td>external graph verifier 0.675</td><td></td></tr></table>

Table 3: Matched workflow-controller comparison on the 40-task Telecom benchmark test split. All values are $\mathrm { P a s s ^ { 4 } }$

workflow re-authoring.

The main comparison contrasts ReAct (no guard), PolicyGuard, and POLICYGUIDE on the same task IDs and GPT 5.4 substrate; ToolGuard is included on Airline, where its released code guards apply. All main cells use $n { = } 4$ . We report Pass<sup>1</sup> and $\mathrm { P a s s ^ { 4 } }$ ; unless explicitly labeled $\mathrm { P a s s ^ { 1 } }$ , PV and Mut denote $\mathrm { P a s s ^ { 4 } }$ on the corresponding task slice. For a task with c successful trials, $\begin{array} { r } { \mathrm { P a s s } ^ { k } = \binom { c } { k } / \binom { n } { k } } \end{array}$ averaged across tasks. We also include FlowAgent (Shi et al., 2025b) as a matched workflowcontroller baseline on Telecom (§4.4). The benchmark’s standard evaluators score final database state and natural-language task assertions rather than complete temporal conformance of intermediate actions. $\mathrm { P a s s } ^ { k }$ therefore measures reliable policy-constrained task outcomes, not direct tracelevel procedural validity. We supplement it with an author-designed Telecom event-order rubric (§4.7) and report the guard-derived Call-NMR audit separately (Appendix F).

## 4.2 Main results

POLICYGUIDE achieves the highest overall $\mathrm { P a s s ^ { 4 } }$ in all three domains (Table 1; Figure 4); the lead persists as k increases, and pooled paired tests favor it over both baselines (Appendix C). Gains are largest on Telecom’s long diagnose–instruct–verify chains, consistent with persisted graph position mattering most for ordered procedures rather than one final action. The improvement spans both PV and Mut, rather than trading completion for stricter blocking.

<table><tr><td>Agent</td><td>Metric</td><td>ReAct</td><td>PolicyGuard</td><td>POLICYGUIDE</td></tr><tr><td rowspan="3">GPT 5.4</td><td>Overall</td><td>0.460</td><td>0.580</td><td>0.620</td></tr><tr><td>PV</td><td>0.750</td><td>1.000</td><td>0.917</td></tr><tr><td>Mut</td><td>0.192</td><td>0.192</td><td>0.346</td></tr><tr><td rowspan="3">Claude Sonnet 4.6</td><td>Overall</td><td>0.720</td><td>0.780</td><td>0.780</td></tr><tr><td>PV</td><td>0.958</td><td>1.000</td><td>1.000</td></tr><tr><td>Mut</td><td>0.500</td><td>0.577</td><td>0.577</td></tr><tr><td rowspan="3">Gemini 2.5 Pro</td><td>Overall</td><td>0.480</td><td>0.600</td><td>0.680</td></tr><tr><td>PV</td><td>0.750</td><td>1.000</td><td>0.917</td></tr><tr><td>Mut</td><td>0.231</td><td>0.231</td><td>0.462</td></tr></table>

Table 4: Agent-family generalization on Airline (50 tasks, $n { = } 4 ;$ verifier model paired to the agent). All metrics are Pass<sup>4</sup>. The GPT 5.4-authored workflow graph is reused without re-authoring.

![](images/17b5e7810d3307890c848a754d76de70045fd95f68d7a009739e1cecfe0e2210.jpg)  
Figure 5: $\mathrm { P a s s } ^ { k }$ for Claude Sonnet 4.6 and Gemini 2.5 Pro agents on airline (verifier = agent), with n=4 for every system.

Retail separates guidance from blocking: POL-ICYGUIDE preserves ReAct’s Mut performance while improving PV, whereas PolicyGuard’s PV gain coincides with lower Mut. The overall difference is not significant.

## 4.3 Diagnostic workflow variants

Actor-only workflow access. POLICYGUIDE-SELF gives the frozen graph to the actor but removes the external verifier, persisted state, remediation, and mutation intercept. Its Mut $\mathrm { P a s s ^ { 4 } }$ does not exceed ReAct in any domain, showing that access to the workflow does not by itself ensure reliable execution. Because several runtime components are removed together, this comparison tests the external stack as a bundle rather than isolating state persistence alone.

Compiled structure under external tracking. POLICYGUIDE-RAW retains the verifier schedule and remediation channel but replaces the graph with raw policy text. Relative to this matched guide,

![](images/fdf5140d62c5f4aecd4393c7b9eeae027ec5600f75cf09a0cae135f84fb43064.jpg)  
Figure 6: CRAFT red-team attack-success rate on airline (20 attack tasks, GPT 5.4, n=4); lower is safer.

POLICYGUIDE improves overall $\mathrm { P a s s ^ { 4 } }$ by 0.100, 0.150, and 0.325 on Airline, Retail, and Telecom. The larger Telecom gap is consistent with explicit graph position helping the verifier resume long, ordered diagnostic chains. These ablations are therefore diagnostic rather than a complete factorial decomposition.

## 4.4 Matched workflow-controller comparison

Table 3 adds the closest workflow-aware runtime comparison. For representation matching, we deterministically compile the same frozen graph into PDL, with no LLM authoring. FlowAgent places the raw policy and PDL inside the actor and applies its released API-dependency and duplicate-call controllers; POLICYGUIDE instead tracks graph state in an external persisted verifier. ReAct and PolicyGuard provide actor-only and action-local references.

## 4.5 Generalization across agent families

The GPT 5.4-authored Airline graph is reused unchanged with Claude Sonnet 4.6 and Gemini 2.5 Pro (Table 4; Figure 5), separating executor transfer from workflow re-authoring. The gains over unguided execution support executor-side transfer across all three model families. For Gemini 2.5 Pro, POLICYGUIDE improves Mut $\mathrm { P a s s ^ { 4 } }$ from 0.231 under either baseline to 0.462, while its lower PV than PolicyGuard (0.917 versus 1.000) indicates a completion benefit rather than stricter final-action checking. Transfer across workflow-author models remains untested.

## 4.6 Adversarial robustness

Under CRAFT (Nakash et al., 2025), persuasive users inject false eligibility premises to induce forbidden mutations. POLICYGUIDE has the lowest ASR@k at every k (Figure 6); its per-trial ASR is

<table><tr><td>System</td><td></td><td>Step-TCR Trace-TCR Process-valid rate</td><td></td></tr><tr><td>ReAct</td><td>86.4</td><td>35.4</td><td>17.5</td></tr><tr><td>PolicyGuard</td><td>85.7</td><td>23.9</td><td>13.1</td></tr><tr><td>POLICYGUIDE</td><td>94.5</td><td>63.4</td><td>56.2</td></tr></table>

Table 5: Author-designed Telecom ordered trace compliance $( \% ; n { = } 4 )$ . Step- and Trace-TCR condition on outcome-passing traces.

0.087, versus 0.125 for PolicyGuard and 0.200 for ReAct, preventing 91.3% of tested attacks while improving benign completion. These results show that POLICYGUIDE is robust to CRAFT red-team attacks. Its lower ASR is consistent with requiring tool-derived evidence, so unsupported user claims cannot satisfy workflow prerequisites.

## 4.7 Procedural trace compliance

Because final-state Pass ignores intermediate order, the authors manually designed a task-conditioned Telecom rubric from the raw policy and support manual. It checks identification, diagnosis before intervention, consent, correction order, and final verification. On outcome-passing traces, Step-TCR is the fraction of applicable checks satisfied and Trace-TCR the fraction satisfying all checks. The process-valid rate is the fraction of all task–run pairs passing both the Tau2 outcome and the rubric, pooled across four runs; it is our sole end-to-end measure, not a $\mathrm { P a s s } ^ { k }$ statistic. This best-effort workflow-level extension of prerequisite analysis is distinct from Call-NMR (Rabinovich et al., 2026; Kang et al., 2026), whose Telecom adaptation appears in Appendix F.

POLICYGUIDE attains the highest process-valid rate (56.2%, versus 17.5% for ReAct and 13.1% for PolicyGuard; Table 5) and leads both conditional diagnostics. We report the latter only to characterize successful traces.

## 5 Conclusion

POLICYGUIDE replaces action-local checks with an external guide that traverses a compiled policy graph, persists progress, and returns targeted remediation. Across three τ<sup>2</sup>-bench domains it achieves the best overall $\mathrm { P a s s ^ { 4 } }$ , transfers across agent families, and performs strongest on procedural Telecom. Ablations and an author-designed ordered-trace audit support external graph tracking and treating the procedure—not only the final action—as the unit of policy adherence.

## Limitations

Evaluation scope. We evaluate three English τ<sup>2</sup>-bench customer-service domains (Barres et al., 2025) with a frozen user simulator and four trials per multi-trial cell. These domains vary in procedural structure, but do not represent other policy regimes, languages, or live users. Retail has only 10 PV tasks and the overall gain over ReAct is not significant. The ToolGuard runtime baseline is Airline-only because its released guards target that domain (Zwerdling et al., 2025); our generated Telecom guards are used only as a frozen NMR oracle. Paired tests are limited to systems with pertask outputs. Because $\tau ^ { 2 } .$ -bench does not supply a general ordered-trace oracle, our Telecom trace metric is an exploratory, author-designed operationalization of selected task-relevant requirements observable in serialized traces. It uses deterministic event ordering and text matching, is conditioned by gold task actions, and has no second-annotator agreement estimate; it therefore does not establish exhaustive natural-language policy compliance. Call-NMR partially audits prior reads in Airline and Retail, while its adapted Telecom oracle saturates and is reported only as a non-identification result (Appendix F).

Benchmark coverage. Adjacent benchmarks test related but different questions. CRMArena-Pro (Huang et al., 2025) emphasizes business-task capability and confidentiality awareness; IntellAgent (Levi and Kadar, 2025) generates diagnostic conversations; Near-Miss (Rabinovich et al., 2026) audits completed trajectories; and AgentReward-Bench (Lù et al., 2025) evaluates trajectory judges. CRAFT (Nakash et al., 2025) supplies adversarial users for our robustness audit, not the benign workflow-completion distribution. We report only its clean, release-aligned 20-task Airline split. Although the CRAFT paper also evaluates Retail, the released Retail task lists, cached strategies, and evaluator in our checkout do not reproduce one consistent paper-faithful 30-task set; no official Telecom set exists. Our result therefore shows that POLICYGUIDE prevents most tested persuasive Airline attacks, which is sufficient to check that its completion gains do not sacrifice resistance, but does not establish cross-domain or adaptiveattack robustness. None is therefore a drop-in test of online workflow guidance, and transfer would require new workflows and task-specific outcome measures.

Workflow generation and faithfulness. Using one frozen GPT 5.4-authored workflow per domain is a deliberate control: every system and agent family receives the same policy representation, so the comparison isolates runtime and executor differences rather than re-authoring. This achieves the study’s fair-comparison objective, but does not establish author-side generalization across models or seeds. We address workflow faithfulness separately by manually verifying each frozen graph against its source policy and tool specifications (Appendix E).

Trigger and cost. The reported runtime fires at user-turn boundaries and after its one-shot corrective intercept, rather than before every policyrelevant agent action. Corollary 2 characterizes the corresponding coverage condition and shows why deviations between these intervention points remain outside the unconditional guarantee. Broader intervention coverage would require more verifier calls. Guide calls cost approximately \$0.40 per conversation; smaller models and sparser invocation can reduce, but not eliminate, this overhead.

Probabilistic enforcement. Like other LLMbased agent safeguards (Chen et al., 2025; Xiang et al., 2025; Kang et al., 2026), each node judg ment is probabilistic, so compliance is empirical rather than guaranteed. Theorem 1 assumes a faithful workflow, an ideal binding verifier, and coverage before every reachable first deviation (Appendix A); the advisory runtime does not satisfy these conditions unconditionally. Verifier exceptions are fail-open, so deployments requiring hard guarantees need an additional deterministic monitor for the formally expressible policy subset.

## Ethics Statement

POLICYGUIDE is a probabilistic aid for policy adherence, not a guarantee, and should not be the sole control for high-stakes actions. Because the verifier reads the conversation, tool results, and persisted workflow state, privacy, access control, retention, and auditing requirements must extend to verifier calls and logs. All experiments use synthetic τ<sup>2</sup>-bench tasks and simulated users (Barres et al., 2025); no real customer data or external actions are involved. Generated workflows may reproduce source-policy errors or introduce unsupported restrictions, so deployment requires review by policy owners, monitoring, and a safe fallback. We will release the prompts, workflow schemas, and analysis artifacts needed for reproducibility.

## References

James P. Anderson. 1972. Computer security technology planning study. Technical Report ESD-TR-73- 51, Vol. I, Electronic Systems Division, Air Force Systems Command, Hanscom AFB, Bedford, MA. DTIC accession no. AD-758206.

Anthropic. 2026. Claude sonnet 4.6 system card. Anthropic, https://anthropic.com/ claude-sonnet-4-6-system-card.

Sumanth Balaji, Piyush Mishra, Aashraya Sachdeva, and Suraj Agrawal. 2026. Beyond ivr: Benchmarking customer support llm agents for businessadherence. arXiv preprint arXiv:2601.00596.

Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. 2025. τ<sup>2</sup>-bench: Evaluating conversational agents in a dual-control environment. arXiv preprint arXiv:2506.07982.

Andreas Bauer, Martin Leucker, and Christian Schallhart. 2011. Runtime verification for LTL and TLTL. ACM Transactions on Software Engineering and Methodology, 20(4):14:1–14:64.

Zhaorun Chen, Mintong Kang, and Bo Li. 2025. Shieldagent: Shielding agents via verifiable safety policy reasoning. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML).

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Matthew Henderson, Blaise Thomson, and Steve Young. 2014. Word-based dialog state tracking with recurrent neural networks. In Proceedings of the 15th Annual Meeting ofthe Special Interest Group on Discourse and Dialogue (SIGDIAL), pages 292–299. Association for Computational Linguistics.

Sirui Hong, Mingchen Zhuge, Jiaqi Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. 2024. MetaGPT: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations (ICLR).

Kung-Hsiang Huang, Akshara Prabhakar, Onkar Thorat, Divyansh Agarwal, Prafulla Kumar Choubey, Yixin Mao, Silvio Savarese, Caiming Xiong, and Chien-Sheng Wu. 2025. Crmarena-pro: Holistic assessment of llm agents across diverse business scenarios and interactions. arXiv preprint arXiv:2505.18878. Also published in Transactions on Machine Learning Research (TMLR), 2026.

Seongjae Kang, Taehyung Yu, and Sung Ju Hwang. 2026. Policyguard: A dialogue-grounded sub-agent verifier for policy adherence in llm agents. arXiv preprint arXiv:2606.29225.

Martin Leucker and Christian Schallhart. 2009. A brief account of runtime verification. Journal ofLogic and Algebraic Programming, 78(5):293–303.

Elad Levi and Ilan Kadar. 2025. Intellagent: A multiagent framework for evaluating conversational ai systems. arXiv preprint arXiv:2501.11067.

Jia Liu, Jie Shuai, and Xiyao Li. 2023. State machine of thoughts: Leveraging past reasoning trajectories for enhancing problem solving. arXiv preprint arXiv:2312.17445.

Xing Han Lù, Amirhossein Kazemnejad, Nicholas Meade, Arkil Patel, Dongchan Shin, Alejandra Zambrano, Karolina Stanczak, Peter Shaw, Christopher J.´ Pal, and Siva Reddy. 2025. Agentrewardbench: Evaluating automatic evaluations of web agent trajectories. arXiv preprint arXiv:2504.08942.

Weidi Luo, Shenghong Dai, Xiaogeng Liu, Suman Banerjee, Huan Sun, Muhao Chen, and Chaowei Xiao. 2025. AGrail: A lifelong agent guardrail with effective and adaptive safety detection. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 8104–8139, Vienna, Austria. Association for Computational Linguistics.

Yutao Mou, Zhangchi Xue, Lijun Li, Peiyang Liu, Shikun Zhang, Wei Ye, and Jing Shao. 2026. Toolsafe: Enhancing tool invocation safety of llm-based agents via proactive step-level guardrail and feedback. Preprint, arXiv:2601.10156. ArXiv preprint arXiv:2601.10156.

Itay Nakash, George Kour, Koren Lazar, Matan Vetzler, Guy Uziel, and Ateret Anaby-Tavor. 2025. Effective red-teaming of policy-adherent agents. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 2250–2268, Suzhou, China. Association for Computational Linguistics.

OpenAI. 2026. Introducing GPT-5.4. OpenAI Blog, https://openai.com/index/ introducing-gpt-5-4/. Released March 5, 2026.

Nils Palumbo, Sarthak Choudhary, Jihye Choi, Prasad Chalasani, and Somesh Jha. 2026. Policy compiler for secure agentic systems. Preprint, arXiv:2602.16708. ArXiv preprint arXiv:2602.16708.

Ella Rabinovich, David Boaz, Naama Zwerdling, and Ateret Anaby-Tavor. 2026. Near-miss: Latent policy failure detection in agentic workflows. arXiv preprint arXiv:2603.29665.

Shamik Roy, Sailik Sengupta, Daniele Bonadiman, Saab Mansour, and Arshit Gupta. 2024. FLAP: Flow-adhering planning with constrained decoding in LLMs. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 517–539, Mexico City, Mexico. Association for Computational Linguistics.

Fred B. Schneider. 2000. Enforceable security policies. ACM Transactions on Information and System Security, 3(1):30–50.

Tianneng Shi, Jingxuan He, Zhun Wang, Hongwei Li, Linyu Wu, Wenbo Guo, and Dawn Song. 2025a. Progent: Securing ai agents with privilege control. arXiv preprint arXiv:2504.11703.

Yuchen Shi, Siqi Cai, Zihan Xu, Yulei Qin, Gang Li, Hang Shao, Jiawei Chen, Deqing Yang, Ke Li, and Xing Sun. 2025b. Flowagent: Achieving compliance and flexibility for workflow agents. arXiv preprint arXiv:2502.14345.

Lillian Tsai and Eugene Bagdasarian. 2025. Contextual agent security: A policy for every purpose. In Proceedings ofthe 20th Workshop on Hot Topics in Operating Systems (HotOS 2025).

Haoyu Wang, Christopher M. Poskitt, and Jun Sun. 2026. Agentspec: Customizable runtime enforcement for safe and reliable llm agents. In Proceedings of the 2026 IEEE/ACM 48th International Conference on Software Engineering (ICSE ’26).

Jason D. Williams, Antoine Raux, Deepak Ramachandran, and Alan W. Black. 2013. The dialog state tracking challenge. In Proceedings ofthe SIGDIAL 2013 Conference, pages 404–413. Association for Computational Linguistics.

Cailin Winston, Claris Winston, and René Just. 2026. Solver-aided verification of policy compliance in tool-augmented llm agents. arXiv preprint arXiv:2603.20449.

Yiran Wu, Tianwei Yue, Shaokun Zhang, Chi Wang, and Qingyun Wu. 2024. Stateflow: Enhancing llm task-solving through state-driven workflows. In Conference on Language Modeling (COLM).

Zhen Xiang, Linzhi Zheng, Yanjie Li, Junyuan Hong, Qinbin Li, Han Xie, Jiawei Zhang, Zidi Xiong, Chulin Xie, Carl Yang, Dawn Song, and Bo Li. 2025. Guardagent: Safeguard llm agents via knowledgeenabled reasoning. arXiv preprint arXiv:2406.09187.

Ruixuan Xiao, Wentao Ma, Ke Wang, Yuchuan Wu, Junbo Zhao, Haobo Wang, Fei Huang, and Yongbin Li. 2024. Flowbench: Revisiting and benchmarking workflow-guided planning for llm-based agents. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 10883–10900.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2024. τ-bench: A benchmark for toolagent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR).

Anbang Ye, Qianran Ma, Jia Chen, Muqi Li, Tong Li, Fujiao Liu, Siqi Mai, Meichen Lu, Haitao Bao, and Yang You. 2025. Sop-agent: Empower general purpose ai agent with domain-specific sops. arXiv preprint arXiv:2501.09316.

Yining Ye, Xin Cong, Shizuo Tian, Jiannan Cao, Hao Wang, Yujia Qin, Yaxi Lu, Heyang Yu, Huadong Wang, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2023. Proagent: From robotic process automation to agentic process automation. arXiv preprint arXiv:2311.10751.

Naama Zwerdling, David Boaz, Ella Rabinovich, Guy Uziel, David Amid, and Ateret Anaby-Tavor. 2025. Towards enforcing company policy adherence in agentic workflows. arXiv preprint arXiv:2507.16459.

## A Theoretical Analysis

Policy workflows constrain a broader set of agent actions than the class mediated by an actiontriggered verifier. We formalize when a firing schedule can preserve procedural validity over this broader workflow.

## A.1 Intervention coverage

We represent each interaction as a sequence of policy-relevant events. A workflow G defines a nonempty language ${ \mathcal { L } } ( G )$ of compliant complete sequences. We call a partial sequence valid when it can still be completed to such a sequence, and write

$$
P _ { G } = { \mathrm { P r e f } } ( { \mathcal { L } } ( G ) ) = \{ \tau \mid \exists \rho : \tau \rho \in { \mathcal { L } } ( G ) \} .\tag{1}
$$

Let $\Sigma _ { \mathrm { a g } }$ denote policy-relevant agent actions, including user-facing messages, instructions, and tool calls, and let $A \subseteq \Sigma _ { \mathrm { a g } }$ be the designated action class mediated by an action-triggered verifier. For mutating-call guards, A is the set of agent-issued mutating tool calls.

A reachable first deviation is a pair $( \tau , e )$ where $\tau \in P _ { G }$ can arise during an interaction, $e \in \Sigma _ { \mathrm { a g } }$ can be the next agent event, and τe /∈ $P _ { G }$ . Let $D _ { G }$ be the set of such deviations. A firing schedule $S$ covers $( \tau , e ) \ \in \ D _ { G }$ if it invokes the verifier after observing τ and before e is committed; let $C _ { S } ( G ) \subseteq D _ { G }$ denote its intervention coverage. We compare:

• the action-triggered schedule $S _ { A }$ , which fires exactly when the proposed event belongs to A; and

• the ideal workflow-level schedule $S _ { \mathrm { w f } }$ , which fires before every policy-relevant agent action is committed.

The theorem compares these schedules under the same ideal verifier: when consulted, it permits an event if and only if the resulting trace remains in $P _ { G }$ , and a rejected event cannot commit. User responses and tool results may update the trace, but do not themselves violate an agent obligation. An uncovered agent event may commit without a verifier verdict.

Theorem 1 (Complete intervention coverage). Under the assumptions above, a firing schedule S preserves procedural validity $( P _ { G }$ membership) throughout every execution if and only if

$$
C _ { S } ( G ) = D _ { G } .\tag{2}
$$

That is, the verifier must cover every reachablefirst deviation.

Proof. For sufficiency, start from the empty valid prefix. If the next observation is not an agent event, it preserves $P _ { G }$ by assumption. If the agent proposes an event that stays in $P _ { G }$ , committing it preserves validity. Otherwise the proposal is in $D _ { G }$ and belongs to $C _ { S } ( G )$ by Equation $2 ;$ the ideal verifier rejects it before it commits. Induction over the interaction therefore keeps every prefix in $P _ { G }$

For necessity, suppose $( \tau , e ) \in D _ { G }$ is not in $C _ { S } ( G )$ . The valid prefix τ can arise during an interaction, and the ideal verifier permits all covered events that lead to it. At $( \tau , e )$ the schedule does not fire, so e can commit and produce $\tau e \notin P G$ . Thus $S$ cannot guarantee procedural validity throughout every execution. □

Corollary 1 (Workflow-level versus action-triggered coverage). The workflow-level schedule covers every member of $D _ { G }$ . The action-triggered schedule covers exactly those members whose action e belongs to A. Therefore workflow-levelfiring preserves procedural validity for every workflow, whereas action-triggeredfiring does so ifand only if every reachable first deviation belongs to A. If $D _ { G }$ contains a point with e /∈ A, workflow-level firing provides a guarantee that action-triggered firing cannot provide, even when both use the same ideal verifier.

Proof. The workflow-level schedule fires before every agent-controlled action, so it covers all of $D _ { G }$ . The action-triggered schedule fires exactly for actions in A. The claims then follow directly from Theorem 1. □

Corollary 2 (Evaluated boundary schedule). Let $S _ { \mathrm { e v a l } }$ fire after every user turn and after the runtime intercepts an unauthorized mutating call. Let $B _ { G } \subseteq D _ { G }$ contain the reachable first deviations $( \tau , e )$ for which one of these firings occurs after the valid prefix τ and before the next agent event e is committed. Under the same ideal, bindingverifier assumptions as Theorem $I , S _ { \mathrm { e v a l } }$ preserves procedural validity if and only if $B _ { G } = D _ { G }$ . In particular, if a first deviation can occur after an intervening agent event and before the next scheduledfiring, the evaluated schedule does not provide an unconditional guarantee.

Proof. By construction, $C _ { S _ { \mathrm { e v a l } } } ( G ) = B _ { G }$ . The claim follows directly from Theorem 1. □

Corollary 2 characterizes the coverage condition for the evaluated cadence; it does not assert that the deployed guide satisfies the theorem’s bindingverifier assumption. Its non-mutating remediation is advisory, and the corrective mutation gate is oneshot, so the experiments measure risk reduction under this practical schedule rather than a formal guarantee.

The timing in Theorem 1 is essential. Because $P _ { G }$ is prefix-closed, once τe /∈ $P _ { G }$ , no later extension can make τe a valid prefix: if some τeρ belonged to $P _ { G }$ , then its prefix τe would also belong to $P _ { G }$ . A later action-triggered verifier may block a subsequent guarded action after reading the history, but it cannot prevent or undo the earlier procedural violation.

## B Policy structure analysis

This appendix supports the paper’s central claim: workflow enforcement matters most when the policy is itselfa workflow, as in telecom. We extend PolicyGuard’s argument-/process-level classification with a third, workflow-level class, apply it to every atomic requirement of all three source policies, and examine how both process- and workflow-level requirements relate to the observed gains. Processlevel requirements motivate context-aware guidance generally, whereas workflow-level requirements create a particular need for ordered graph traversal and persistent progress tracking. Line references refer to the policy documents released with $\tau ^ { 2 }$ -bench: the 136-line retail policy and, for telecom, the 158-line account policy and the 205-line technical-support manual.

## B.1 Operational definitions

Following PolicyGuard (Kang et al., 2026), we classify the source policy document, not any generated artefact of the systems under test. A requirement is argument-level (A) if verifiable from the mutating call’s arguments plus deterministic computation— the class precompiled guards express natively—and process-level (P) if verification must read the user– agent dialogue (D) and/or a prior read-only tool result (T). We split PolicyGuard’s process-level class by adding workflow-level (W): a requirement that additionally mandates an action ordered after the outcome of another required action (a state effect, a user response to a prior step, or a post-act verification), so discharging it out of order is itself a violation. The three classes are mutually exclusive:

W rows read dialogue and tool results like P rows, but P is reserved for theflat remainder, dischargeable at the action point from evidence gathered in any order (status checks, elicit–confirm–act chains). A and P+W therefore remain comparable with PolicyGuard. Requirements are extracted by hand and grouped by each document’s own section headers; descriptive sentences and API meta-rules are excluded. Table 6 gives the partition; the catalogs follow.

<table><tr><td>Domain</td><td>A</td><td>P</td><td>W</td><td>Total</td><td>%P+W</td><td>% W</td></tr><tr><td>Airline</td><td>14</td><td>27</td><td>2</td><td>43</td><td>67.4%</td><td>4.7%</td></tr><tr><td>Retail</td><td>0</td><td>27</td><td>1</td><td>28</td><td>~100%</td><td>3.6%</td></tr><tr><td>Telecom (main)</td><td>1</td><td>21</td><td>7</td><td>29</td><td>96.6%</td><td>24.1%</td></tr><tr><td>Telecom (manual)</td><td>0</td><td>1</td><td>20</td><td>21</td><td>100%</td><td>95.2%</td></tr><tr><td>Telecom (both)</td><td>1</td><td>22</td><td>27</td><td>50</td><td>98.0%</td><td>54.0%</td></tr></table>

Table 6: Argument- (A), process- (P), and workflowlevel (W) partition of the source policies (W splits PolicyGuard’s process-level class; P+W equals it).

## B.2 Airline catalog (summary)

The airline classification is inherited from PolicyGuard (Kang et al., 2026) (full catalog there); we add the workflow-level split. The 167-line policy yields 43 requirements, 14 A / 27 P / 2 W. Argument-level mass sits in booking and modification schema rules (cabin uniformity, passenger limits, payment-method counts); process-level mass splits between dialogue obligations (explicit confirmation, the insurance offer) and tool-read eligibility gates (baggage allowances, flown-segment checks, the disjunctive cancellation and compensation conditions). The two W rows are the transfer pair and the delayed-flight certificate mandated after a change or cancellation—otherwise airline is flat gates discharged at the mutation.

## B.3 Retail catalog

The 28 retail requirements (Table 7) partition as 0 A / 27 P / 1 W (13 D, 14 T, 1 D+T). Every candidate A-row ranges over environment state rather than argument values: the status gates read the order-status enum only get\_order\_details surfaces, the balance and variant constraints compare against profile and catalog reads, and identity must be re-derived even when the user supplies an id. The contrast with airline is mechanical: airline’s book\_reservation passes the whole reservation as call arguments, so schema constraints are argument-checkable, whereas retail and telecom mutations are thin id-referencing calls whose constrained values are surfaced only by read calls.

<table><tr><td>ID</td><td>Line</td><td>Requirement (paraphrased)</td><td>Type</td></tr><tr><td colspan="4">Global rules</td></tr><tr><td>G1</td><td>10</td><td>Authenticate identity by locating the user id via email or name+zip—even when the user already provides the id</td><td>P (D+T)</td></tr><tr><td>G2</td><td>14</td><td>One user per conversation; deny any request about another user</td><td>P (D)</td></tr><tr><td>G3</td><td>16</td><td>List action details + obtain explicit “yes&quot; before any DB-updating action</td><td>P (D)</td></tr><tr><td>G4</td><td>18</td><td>No fabricated information/knowledge/procedures; no subjective recommendations</td><td>P (D)</td></tr><tr><td>G5</td><td>20</td><td>At most one tool call per turn (not paired with a user-facing reply)</td><td>P (D)</td></tr><tr><td>G6</td><td>22</td><td>Deny user requests that are against the policy</td><td>P (D)</td></tr><tr><td>G7</td><td>24</td><td>Transfer iff unhandleable: call transfer_to_human_agents, then the literal handoff message</td><td>W (D)</td></tr><tr><td colspan="4">Generic action rules</td></tr><tr><td>N1</td><td></td><td>Act only on orders with status pending or delivered</td><td>P (T)</td></tr><tr><td>N2</td><td>82 84</td><td>Exchange / modify-items tools callable only once per order</td><td>P (T)</td></tr><tr><td>N3</td><td>84</td><td>Collect all items to change into one list before making the call</td><td>P (D)</td></tr><tr><td colspan="4">Cancel pending order</td></tr><tr><td>C1</td><td>88</td><td>Order status must be pending; check it before taking the action</td><td>P (T)</td></tr><tr><td>C2</td><td>90</td><td>User confirms order id + reason ∈ {no longer needed&#x27;, ordered by mistake&#x27;}; no other reason</td><td>P (D)</td></tr><tr><td colspan="4">Modify pending order</td></tr><tr><td>M1</td><td>96</td><td>Order status must be pending; check it before taking the action</td><td>P (T)</td></tr><tr><td>M2</td><td>98</td><td>Only shipping address, payment method, or item options may be modified—nothing</td><td>P (D)</td></tr><tr><td>M3</td><td>102</td><td>else New payment = a single method, different from the original</td><td>P (T)</td></tr><tr><td>M4</td><td>104</td><td>If the new payment is a gift card, its balance must cover the total amount</td><td>P (T)</td></tr><tr><td>M5</td><td>110</td><td>Modify-items is one-shot (order becomes unmodifiable): remind + confirm all items</td><td>P (D)</td></tr><tr><td>M6</td><td>112</td><td>first Each new item must be available</td><td>P (T)</td></tr><tr><td>M7</td><td>112</td><td>New item = same product, different option (no product-type change)</td><td>P (T)</td></tr><tr><td>M8</td><td>114</td><td>User provides a payment method for the price difference</td><td>P (D)</td></tr><tr><td>M9</td><td>114</td><td>If that payment is à gift card, its balance must cover the price difference</td><td>P (T)</td></tr><tr><td colspan="4">Return delivered order</td></tr><tr><td>R1</td><td>118</td><td>Order status must be delivered; check it before taking the action</td><td>P (T)</td></tr><tr><td>R2</td><td>120</td><td>User confirms order id + the list of items to be returned</td><td>P (D)</td></tr><tr><td>R3</td><td>122-124</td><td>Refund method provided; must be the original payment method or an existing gift</td><td>P (T)</td></tr><tr><td colspan="4">card Exchange delivered order</td></tr><tr><td>E1</td><td>130</td><td>Order status must be delivered; check it before taking the action</td><td>P (T)</td></tr><tr><td>E2</td><td>130</td><td>Remind + confirm the user has provided all items to exchange (one-shot)</td><td>P (D)</td></tr><tr><td>E3</td><td>132</td><td>Each new item = same product, different option, and available</td><td>P (T)</td></tr><tr><td>E4</td><td>134</td><td>Payment for the price difference; if a gift card, balance must cover the difference</td><td>P (T)</td></tr></table>

Table 7: Hand-classified atomic requirements of the $\tau ^ { 2 } .$ -bench Retail policy document (28 requirements, 0 A / 27 P / 1 W; subtypes D-only 13, T-only 14, D+T 1). Line refers to retail/policy.md as released with τ<sup>2</sup>-bench; Type A = argument-level, P = process-level (flat), W = workflow-level (order-bound; Appendix B.1), with D = dialogue-dependent, T = requires a prior read-only tool call.

Structurally, however, retail is the flattest domain: each mutation is guarded by an orderfree conjunction—status ∧ content ∧ payment ∧ confirmation—with prerequisite depth 1, no branching, and no verify-after-act; its only W row is the transfer pair. This is exactly the regime a conversation-aware pass/block verifier already covers, and why retail is where POLICYGUIDE’s graph adds the least.

## B.4 Telecom catalog

The 29 account-policy requirements (Table 8) partition as 1 A / 21 P / 7 W; the lone A row is the refuel-amount bound. The W rows concentrate in the overdue-payment sequence: send the payment request only for a confirmed-Overdue bill, call make\_payment only after the user accepts the request it created, and always verify the bill became PAID—prerequisite chaining that ends in a verify-after-act obligation a pre-execution verifier structurally cannot enforce. Suspension adds crossprocedure dependence (lift only after the overdue bills are paid, unless the contract has ended) and a post-act duty only the user can perform (reboot the device).

<table><tr><td>ID</td><td>Line</td><td>Requirement (paraphrased)</td><td>Type</td></tr><tr><td colspan="4">Global rules</td></tr><tr><td>G1</td><td>7</td><td>No fabricated information/knowledge/procedures; no subjective recommendations</td><td>P (D)</td></tr><tr><td>G2</td><td>9</td><td>At most one tool call per turn (not paired with a user-facing reply)</td><td>P (D)</td></tr><tr><td>G3</td><td>11</td><td>Deny user requests that are against the policy</td><td>P (D)</td></tr><tr><td>G4</td><td>13</td><td>Transfer iff unhandleable: call transfer_to_human_agents, then the literal handoff message</td><td>W (D)</td></tr><tr><td>G5</td><td>15</td><td>Try your best to resolve the issue before transferring</td><td>W (D)</td></tr><tr><td colspan="4">Customer lookup</td></tr><tr><td>L1</td><td>94-97</td><td>Identify the customer via phone number, customer ID, or full name + date of birth</td><td>P (D+T)</td></tr><tr><td>L2</td><td>99</td><td>For name lookup, date of birth is required for verification</td><td>P (D)</td></tr><tr><td colspan="4">Overdue bill payment (ordered procedure)</td></tr><tr><td>01</td><td>105,117</td><td>① Check the bill status is Overdue before acting (the API does not check it)</td><td>P (T)</td></tr><tr><td>02</td><td>106</td><td>② Check the bill amount due</td><td>P (T)</td></tr><tr><td>03</td><td>107-108</td><td>③ Send the payment request (→ AWAITING PAYMENT); gated on O1</td><td>P (T)</td></tr><tr><td>04</td><td>109-110</td><td>④ Inform the user to check their payment requests</td><td>W (D)</td></tr><tr><td>05</td><td>111</td><td>⑤ Only after the user accepts, call make_payment</td><td>W (D+T)</td></tr><tr><td>06</td><td>113</td><td>⑥ Always verify the bill became PAID before telling the user</td><td>W (T)</td></tr><tr><td>07</td><td>116</td><td>At most one bill in AWAITING PAYMENT at a time</td><td>P (T)</td></tr><tr><td colspan="4">Line suspension</td></tr><tr><td>S1</td><td>125</td><td>Lift a suspension only after all overdue bills are paid</td><td>W (T)</td></tr><tr><td>S2</td><td>126</td><td>Do not lift if the contract end date is past—even if all bills are paid</td><td>P (T)</td></tr><tr><td>S3</td><td>128</td><td>After resuming, instruct the user to reboot the device</td><td>W (D)</td></tr><tr><td colspan="4">Data refueling (ordered procedure)</td></tr><tr><td>F1</td><td>134</td><td>Refuel amount ≤ 2 GB</td><td>A</td></tr><tr><td>F2</td><td>136</td><td>① Ask how much data the user wants to refuel</td><td>P (D)</td></tr><tr><td>F3</td><td>137</td><td>② Confirm the price</td><td>P (D)</td></tr><tr><td>F4</td><td>138</td><td>③ Apply the refuel to the line associated with the user&#x27;s phone number</td><td>P (D+T)</td></tr><tr><td colspan="4">Change plan (ordered procedure)</td></tr><tr><td>P1</td><td>144</td><td>① Establish which line the plan change is for</td><td>P (D)</td></tr><tr><td>P2</td><td>145</td><td>② Gather the available plans</td><td>P (T)</td></tr><tr><td>P3</td><td>146</td><td>③ Ask the user to select one</td><td>P (D)</td></tr><tr><td>P4</td><td>147</td><td>④ Calculate the price of the new plan</td><td>P (T)</td></tr><tr><td>P5</td><td>148</td><td>⑤ Confirm the price</td><td>P (D)</td></tr><tr><td>P6</td><td>149</td><td>⑥ Apply the plan to the line associated with the user&#x27;s phone number</td><td>P (D+T)</td></tr><tr><td colspan="4">Data roaming</td></tr><tr><td>RM1</td><td>155</td><td>If the user is travelling abroad, check whether the line is roaming-enabled</td><td>P (T)</td></tr><tr><td>RM2</td><td>155</td><td>If not enabled, enable it at no cost</td><td>P (T)</td></tr></table>

Table 8: Hand-classified atomic requirements of the τ<sup>2</sup>-bench Telecom main\_policy.md (29 requirements, 1 A / 21 P / 7 W). Line refers to the document as released with τ<sup>2</sup>-bench; ⃝n marks a step in an ordered procedure (“To do so you need to follow these steps”). Types as in Table 7.

The 205-line technical-support manual is a troubleshooting manual rather than a rulebook: 21 requirements (Table 9), 1 P / 20 W. Its defining property is dual control: fixes execute on the user’s device, so the agent must instruct the user, await their report, and re-verify. Every rule instantiates diagnose → conditional fix → re-verify, and the chapters impose the prerequisite chain Service ⊂ Data ⊂ MMS—a literal decision tree with mandated traversal order, on which a task may contain no agent-side mutating call to intercept at all.

## B.5 Policy structure and observed gains

Every domain is majority process-level, consistent with the benefit of context-aware guidance over naive acting. Yet retail—the most processlevel domain—gains least, indicating that the P+W fraction alone does not explain the cross-domain variation. Workflow-level requirements are much more concentrated in telecom: 2/43 on airline and 1/28 on retail versus 27/50 on telecom, including 20/21 in the manual alone (Table 6). The rule is applied uniformly: telecom’s refuel and changeplan recipes are elicit–confirm–apply chains and remain P, the same shallow shape as retail; its W mass lives in the overdue-payment state machine, the suspension procedure, and above all the diagnostic manual. POLICYGUIDE tracks position in a workflow graph across turns, so it has the most to exploit exactly where W requirements concentrate. Consistent with this account, the compiled-graph gain over the matched raw-policy guide is larger on telecom (+0.325) than on airline (+0.100) or retail (+0.150; §4.3). Thus processlevel requirements help explain the general value of context-aware guidance, while the concentration of workflow-level requirements helps explain why POLICYGUIDE’s explicit graph and state tracking are especially useful in telecom.

<table><tr><td>ID</td><td>Line</td><td>Requirement (diagnose → conditional fix → verify)</td><td>Type</td></tr><tr><td colspan="4">Cellular service (ll. 55–99)</td></tr><tr><td>TSS1</td><td>69-72</td><td>Diagnose service via check_status_bar</td><td>P (T)</td></tr><tr><td>TSS2</td><td>74-78</td><td>If Airplane Mode ON → guide toggle_airplane_mode OFF</td><td>W(D+T)</td></tr><tr><td>TSS3</td><td>79-87</td><td>Check SIM: Missing → reseat; Locked → escalate; Active → ok (three-way branch)</td><td>W (D+T)</td></tr><tr><td>TSS4</td><td>88-92</td><td>If APN incorrect → guide reset_apn_settings, then reboot_device</td><td>W (D+T)</td></tr><tr><td>TSS5</td><td>93-99</td><td>If line suspended → handle per main policy, then verify service restored</td><td>W (T)</td></tr><tr><td colspan="4">Mobile data (ll. 100–163)</td></tr><tr><td>TSD0</td><td>106-108</td><td>Prerequisite: the user must first have cellular service</td><td>W (T)</td></tr><tr><td>TSD1</td><td>122-127</td><td>Diagnose via run_speed_test</td><td>W (T)</td></tr><tr><td>TSD2</td><td>129-131</td><td>Airplane Mode (as in the Service chapter)</td><td>W (D+T)</td></tr><tr><td>TSD3</td><td>132-135</td><td>If mobile data disabled → guide toggle_data ON</td><td>W (D+T)</td></tr><tr><td>TSD4</td><td>136-141</td><td>If roaming abroad &amp; data off → guide toggle_roaming + verify the line is</td><td>W (D+T)</td></tr><tr><td>TSD5</td><td>142-145</td><td>roaming-enabled If Data Saver ON → guide toggle_data_saver_mode OFF</td><td>W (D+T)</td></tr><tr><td>TSD6</td><td>146-150</td><td>If VPN ON &amp; performance poor → guide disconnect_vpn</td><td>W (D+T)</td></tr><tr><td>TSD7</td><td>151-158</td><td>If usage exceeds the plan limit → offer change-plan or refuel</td><td>W (T)</td></tr><tr><td>TSD8</td><td>159-163</td><td>If network mode 2G/3G → guide set_network_mode_preference</td><td>W (D+T)</td></tr><tr><td colspan="4">MMS (ll. 164–205)</td></tr><tr><td>TSM0</td><td>170-173</td><td>Prerequisite: the user must have cellular service and mobile data</td><td>W (T)</td></tr><tr><td>TSM1</td><td>181-183</td><td>Diagnose via can_send_mms</td><td>W (T)</td></tr><tr><td>TSM2</td><td>185-188</td><td>Ensure basic service + data connectivity first</td><td>W (T)</td></tr><tr><td>TSM3</td><td>189-193</td><td>If on 2G → guide set_network_mode_preference to 3G+</td><td>W (D+T)</td></tr><tr><td>TSM4</td><td>194-199</td><td>If MMSC URL unset → guide reset_apn_settings, then reboot_device</td><td>W (D+T)</td></tr><tr><td>TSM5</td><td>200-203</td><td>If Wi-Fi Calling ON → guide toggle_wifi_calling OFF</td><td>W (D+T)</td></tr><tr><td>TSM6</td><td>204-205</td><td>If the messaging app lacks storage/SMS permissions → guide grant_app_permission</td><td>W (D+T)</td></tr></table>

Table 9: Hand-classified atomic requirements of the $\tau ^ { 2 } .$ -bench Telecom tech\_support\_manual.md (21 requirements, 1 P / 20 W). Line refers to the document as released with $\tau ^ { 2 }$ -bench. Every rule is a diagnostic-gated (T) user-guidance (D) step; the three chapters form the prerequisite chain Service ⊂ Data ⊂ MMS; every row except TSS1 (the entry diagnostic) is workflow-level.

## C Reliability and significance

Pass<sup>k</sup> breakdown. Table 10 gives the exact values underlying Figure 4. Table 11 reports Pass<sup>1</sup> separately for each trial. These tables use the same base splits as Table 1: Airline base-50 and Retail/Telecom base-114.

Paired significance. For each task, $\mathrm { P a s s ^ { 4 } }$ is one iff all four trials succeed. We compare systems on common tasks and obtain 95% confidence intervals by paired bootstrap (10,000 task-level resamples).

<table><tr><td>Domain</td><td>System</td><td> $\mathrm { P ^ { 1 } }$ </td><td> $\mathrm { P ^ { 2 } }$ </td><td> $\mathrm { P ^ { 3 } }$ </td><td> $\mathrm { P ^ { 4 } }$ </td><td> $\mathrm { P ^ { 4 } / P ^ { 1 } }$ </td></tr><tr><td rowspan="4">Airline</td><td>ReAct</td><td>0.640</td><td>0.530</td><td>0.485</td><td>0.460</td><td>0.72</td></tr><tr><td>ToolGuard</td><td>0.575</td><td>0.553</td><td>0.535</td><td>0.520</td><td>0.90</td></tr><tr><td>PolicyGuard</td><td>0.710</td><td>0.630</td><td>0.595</td><td>0.580</td><td>0.82</td></tr><tr><td>POLICYGUIDE</td><td>0.775</td><td>0.707</td><td>0.660</td><td>0.620</td><td>0.80</td></tr><tr><td rowspan="3">Retail</td><td>ReAct</td><td>0.800</td><td>0.700</td><td>0.638</td><td>0.596</td><td>0.75</td></tr><tr><td>PolicyGuard</td><td>0.645</td><td>0.506</td><td>0.421</td><td>0.360</td><td>0.56</td></tr><tr><td>POLICYGUIDE</td><td>0.809</td><td>0.715</td><td>0.654</td><td>0.614</td><td>0.76</td></tr><tr><td rowspan="3">Telecom</td><td>ReAct</td><td>0.384</td><td>0.273</td><td>0.226</td><td>0.193</td><td>0.50</td></tr><tr><td>PolicyGuard</td><td>0.406</td><td>0.292</td><td>0.237</td><td>0.202</td><td>0.50</td></tr><tr><td>POLICYGUIDE</td><td>0.866</td><td>0.763</td><td>0.682</td><td>0.614</td><td>0.71</td></tr></table>

Table 10: $\mathrm { P a s s } ^ { k }$ breakdown for the base-split results in Table 1 and Figure 4 (GPT 5.4, n=4). $\mathrm { \bar { P } ^ { 4 } / P ^ { 1 } }$ is the consistency ratio.

Across domain strata, we use

$$
Z = \frac { \sum _ { d } a _ { d } - \sum _ { d } b _ { d } } { \sqrt { \sum _ { d } a _ { d } + \sum _ { d } b _ { d } } } ,
$$

where $a _ { d }$ counts POLICYGUIDE-only $\mathrm { P a s s ^ { 4 } }$ successes and $b _ { d }$ the reverse. Table 12 shows that POLICYGUIDE improves pooled Pass<sup>4</sup> over ReAct $( p < 1 0 ^ { - 8 } )$ and PolicyGuard $( p < 1 0 ^ { - 1 2 } )$ , pooling all three domains. Domain-level effect sizes and intervals appear in Table 13. Because $\Delta \mathrm { P ^ { 4 } }$ is a signed difference rather than a probability, negative limits are valid; an interval crossing zero indicates that the domain-level difference is not statistically distinguishable from zero.

<table><tr><td>Domain</td><td>System</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>pstd</td></tr><tr><td rowspan="4">Airline</td><td>ReAct</td><td>0.620</td><td>0.620</td><td>0.640</td><td>0.680</td><td>0.024</td></tr><tr><td>ToolGuard</td><td>0.560</td><td>0.580</td><td>0.580</td><td>0.580</td><td>0.009</td></tr><tr><td>PolicyGuard</td><td>0.700</td><td>0.740</td><td>0.700</td><td>0.700</td><td>0.017</td></tr><tr><td>POLICYGUIDE</td><td>0.800</td><td>0.720</td><td>0.820</td><td>0.760</td><td>0.038</td></tr><tr><td rowspan="3">Retail</td><td>ReAct</td><td>0.807</td><td>0.746</td><td>0.842</td><td>0.807</td><td>0.035</td></tr><tr><td>PolicyGuard</td><td>0.649</td><td>0.623</td><td>0.632</td><td>0.675</td><td>0.020</td></tr><tr><td>POLICYGUIDE</td><td>0.816</td><td>0.798</td><td>0.789</td><td>0.833</td><td>0.017</td></tr><tr><td rowspan="3">Telecom</td><td>ReAct</td><td>0.342</td><td>0.377</td><td>0.404</td><td>0.412</td><td>0.027</td></tr><tr><td>PolicyGuard</td><td>0.465</td><td>0.386</td><td>0.360</td><td>0.412</td><td>0.039</td></tr><tr><td>POLICYGUIDE</td><td>0.860</td><td>0.860</td><td>0.842</td><td>0.904</td><td>0.023</td></tr></table>

Table 11: Pass<sup>1</sup> in each of the four trials on the base splits. pstd is the population standard deviation across trial-level values.
<table><tr><td>Opponent</td><td>D</td><td></td><td></td><td>∑a∑b ndisc</td><td>Z</td><td>p</td></tr><tr><td>ReAct</td><td>3</td><td>77</td><td>19</td><td></td><td> $9 6 \ + 5 . 9 2$ </td><td> $< 1 0 ^ { - 8 }$ </td></tr><tr><td>PolicyGuard</td><td>3</td><td>97</td><td>19</td><td></td><td> $1 1 6 + 7 . 2 4$ </td><td> $< 1 0 ^ { - 1 2 }$ </td></tr></table>

Table 12: Pooled stratified McNemar tests on per-task Pass<sup>4</sup>. D is the number of domain strata; a counts POLICYGUIDE-only passes and b the reverse, summed across strata.

## D Cost analysis

POLICYGUIDE adds verifier inference to the underlying agent. We therefore report guide-side model usage separately from the actor and user simulator, and examine how prompt caching and the firing policy limit this overhead.

## D.1 Guide-side usage

Table 14 summarizes the GPT 5.4 configuration. The guide costs \$0.34–\$0.56 per task and fires 7.4– 11.5 times per task; Telecom is higher because its diagnostic workflows require longer interactions. Although 85.8–88.1% of prompt tokens are cached, each call generates about 2.2–2.5k output tokens for workflow traversal, evidence checks, and remediation. Output generation consequently accounts for an estimated 67.5–71.0% of guide spend.

Thus, caching substantially reduces repeated input processing, but does not eliminate the marginal cost of the verifier: its structured audit is much longer than a binary policy verdict. Reducing guide output length is therefore the main remaining costoptimization opportunity.

<table><tr><td>Domain</td><td>Opponent</td><td>n</td><td>∆P⁴ [95% CI]</td></tr><tr><td rowspan="3">Airline</td><td>ReAct</td><td>50</td><td rowspan="3"> $+ 0 . 1 6 0 \left[ + 0 . 0 2 0 , + 0 . 3 0 0 \right]$   $+ 0 . 1 0 0 [ - 0 . 0 4 0 , + 0 . 2 6 0 ]$ </td></tr><tr><td>ToolGuard</td><td>50</td></tr><tr><td>PolicyGuard</td><td>50</td></tr><tr><td rowspan="2">Retail</td><td>ReAct</td><td>114</td><td> $+ 0 . 0 1 8 \left[ - 0 . 0 7 0 , + 0 . 1 0 5 \right]$ </td></tr><tr><td>PolicyGuard</td><td>114</td><td>+0.254  $[ + 0 . 1 4 9 , + 0 . 3 6 0 ]$ </td></tr><tr><td rowspan="2">Telecom</td><td>ReAct</td><td>114</td><td> $+ 0 . 4 2 1 \ [ + 0 . 3 1 6 , + 0 . 5 2 6 ]$ </td></tr><tr><td>PolicyGuard 114</td><td></td><td> $+ 0 . 4 1 2 \left[ + 0 . 2 9 8 , + 0 . 5 2 6 \right]$ </td></tr></table>

Table 13: Per-domain paired-bootstrap differences in Pass<sup>4</sup> on the base splits (10,000 task-level resamples). Positive values favor POLICYGUIDE.
<table><tr><td>Domain</td><td>Calls/ task</td><td>Prompt tok./call</td><td>Cached input</td><td>Output tok./call</td><td>total $</td><td>Guide Guide $/task</td></tr><tr><td>Airline</td><td>7.56</td><td>32,360</td><td>88.1%</td><td>2,478</td><td>20.10</td><td>0.40</td></tr><tr><td>Retail</td><td>7.42</td><td>22,803</td><td>85.8%</td><td>2,179</td><td>13.67</td><td>0.34</td></tr><tr><td>Telecom</td><td>11.47</td><td>28,518</td><td>86.5%</td><td>2,186</td><td>22.29</td><td>0.56</td></tr></table>

Table 14: Guide-side model usage for the GPT 5.4 configuration (50 Airline and 40 Retail/Telecom tasks). Costs exclude the actor and user simulator.

## D.2 Wall-clock time

Table 15 reports mean end-to-end task time. POLI-CYGUIDE requires 5.45–5.78× the observed wallclock time of ReAct, reflecting the additional verifier generations at successive turns.

<table><tr><td>Domain</td><td>ReAct (s/task)</td><td>POLICYGUIDE (s/task)</td><td>Ratio</td></tr><tr><td>Airline</td><td>36.4</td><td>210.1</td><td>5.78×</td></tr><tr><td>Retail</td><td>34.6</td><td>193.6</td><td>5.60×</td></tr><tr><td>Telecom</td><td>45.5</td><td>247.6</td><td>5.45×</td></tr></table>

Table 15: Mean end-to-end wall-clock time per task.

The measurement covers the complete simulated conversation, including actor, verifier, usersimulator, and tool execution, rather than isolated verifier latency.

## D.3 Cost-aware execution

The verifier prompt places the policy, workflow graph, tool specifications, judging rules, and output contract in a byte-stable prefix. Only the evolving conversation and latest request state vary across calls, allowing the repeated enforcement context to benefit from prefix caching.

Each firing uses one verifier generation for all open requests and may advance across several satisfied workflow nodes. The verifier fires before responses to user turns, while intervening tool results are incorporated at the next firing; an intercepted unauthorized action triggers an additional check. Consequently, the number of guide calls scales with relevant agent turns rather than with individual workflow nodes or tool observations.

## E Workflow verification

After generation, the authors manually verified each frozen workflow against the source policy and tool specifications. We reviewed the represented request types, the ordering of policy prerequisites, the authorization and subsequent verification of mutating actions, and the policy or tool-contract basis of graph constraints. This was a verification step: we did not manually edit the generated workflows used in the experiments.

Programmatic validation is also integrated into workflow generation. Each generated file is schema-validated, and the assembled graph is checked for subflow composition, valid tool references and decision branches, reachability, and mutating-action authorization coverage. The resulting findings are supplied to the pipeline’s automated review stage. We reran these checks on the exact frozen workflows; Table 16 reports the results.

<table><tr><td>Domain</td><td></td><td>Nodes Auth. nodes</td><td>Validator flags</td></tr><tr><td>Airline</td><td>158</td><td>11</td><td>0</td></tr><tr><td>Retail</td><td>104</td><td>7</td><td>0</td></tr><tr><td>Telecom</td><td>127</td><td>5</td><td>1</td></tr></table>

Table 16: Programmatic validation rerun on the frozen workflow graphs.

The Telecom flag concerns disable\_roaming, which is exposed by the environment but has no authorizing workflow path. Manual review confirmed that the source policy specifies enabling roaming but does not authorize the agent to disable it. Its absence therefore does not omit a source-policy procedure; the workflow correctly leaves this action outside its authorized policy scope.

## F Call-level near-miss audit

A near miss (Rabinovich et al., 2026) is a mutation in an outcome-passing task that lacks a policy prerequisite. Following the runtime-view convention used by PolicyGuard, Call-NMR (Rabinovich et al., 2026; Kang et al., 2026) is the fraction of successfully executed mutating calls in passing Mut trajectories that lack at least one earlier read required by a frozen ToolGuard guard. Blocked attempts, tool errors, and calls without a successful response are excluded. The same domain oracle is applied to every system.

<table><tr><td>Call-NMR (%; ↓)</td><td>Airline</td><td>Retail</td><td>Telecom†</td></tr><tr><td>ReAct</td><td>25.4</td><td>47.6</td><td>0.0</td></tr><tr><td>PolicyGuard</td><td>32.5</td><td>34.8</td><td>0.0</td></tr><tr><td>POLICYGUIDE</td><td>15.6</td><td>34.7</td><td>0.0</td></tr></table>

Table 17: Call-NMR on passing Mut trajectories (n=4): percentage of successfully executed agent mutations missing a frozen guard-derived read prerequisite. <sup>†</sup>Telecom is an adapted, agent-side diagnostic whose read oracle saturates; its zeros do not establish equal procedural quality.

On Airline, POLICYGUIDE has the lowest observed rate (15.6%, versus 25.4% for ReAct and 32.5% for PolicyGuard). On Retail, POL-ICYGUIDE and PolicyGuard are effectively tied (34.7% and 34.8%); POLICYGUIDE nevertheless supports more outcome-passing Mut trajectories (116 versus 80). Thus Call-NMR audits prior-read coverage conditional on success, not task coverage or complete procedural validity.

Telecom adaptation. The original guard-derived audit is not defined for Telecom. We adapt its call-level convention using a frozen GPT 5.4 guard tree generated from the tagged concatenation of Telecom’s raw agent policy and technical-support manual. Argument- and response-aware matching requires reads to resolve the same customer, line, or bill as the mutation. Because Telecom is dualcontrol, the denominator covers only agent-side carrier mutations; user/device actions such as toggling data, resetting APN settings, and rebooting are outside the agent-call oracle.

The adapted read oracle yields 0.0% for all three primary systems (Table 17). This is a ceiling effect, not evidence that they are procedurally equivalent: the oracle cannot express conversational evidence such as travel status, selected refuel amount, and price confirmation, nor ordering among user/device actions. This non-identification motivates our workflow-level expansion of prerequisite analysis in Section 4.7.

## G Prompts

This appendix reproduces the load-bearing prompts of both halves of the system as prompt-card figures: the runtime guide prompt (Figures 7–9, teal cards) and the workflow-generation pipeline prompts (Figures 10–11, slate cards). Unicode punctuation is transliterated to ASCII, and the verifier’s next-step instruction is named remediation consistently; otherwise the text is verbatim. In these prompt excerpts, “turn” names a verifier invocation and the hard-gate language states the template’s authorization contract. The reported advisory configuration invokes the verifier at user-turn boundaries, intercepts the first unauthorized mutating call after a user message, and then permits an immediate retry after corrective guidance (§3.4).

The guide’s system message is assembled once per domain by substituting three placeholders— {policy\_doc} (the raw policy), {graph\_doc} (the composed graph rendered as a topology section plus per-node specs), and {tools\_doc} (the mutating/read-only tool partition plus the domain’s closed value vocabularies)—into a fixed template, with the per-turn task instruction appended at the end so the whole thing is one cached static prefix (Appendix D.3). Figure 7 shows the protocol, perturn trigger, and output contract; Figures 8 and 9 the judging rules referenced by the main text’s verifier description (§3.4). The generation pipeline’s shared system prompt carries the schema contract (the eight node types with required fields, edge semantics, id rules) plus the authoring essentials of Figure 10; the plan and adversarial-review stage prompts are in Figure 11. The remaining stages (plan review, per-subflow generation, subflow path review, main wiring) restate subsets of the same contract scoped to their output file.

## H Example workflow graphs

This appendix visualizes one generated-schema workflow per domain, composed flat as the guide’s cached prefix renders it. Subflow composition prefixes node ids (identify\_user.load\_profile, book\_flow.authorize\_book), so the guide addresses every node of every inlined subflow by a stable path-like id. The three figures illustrate the intended schema: every depicted solid edge carries the same label (when: satisfied), depicted mutating tools use a tool\_authorization node (trapezoid) with stated prerequisites upstream, and an authorize→verify pair represents the post-call success check. These visualizations do not establish complete mutating-tool coverage for every frozen artifact; Appendix E reports the manual verification and programmatic checks. Node colors match the main-text node vocabulary: entry/exit , agent\_action user\_input , tool\_call , tool\_authorization , decision , and subflow .

Airline (Figure 12). The shared main spine (intake → identify → classify) and the full transactional book\_reservation path: trip and passenger collection, a mandatory search whose TOOL\_RESULT grounds the flight arguments, the insurance disclosure, and the shared summary-andconfirm exchange, all upstream of the gate. The classifier routes each reconciled request into its request-type subflow; general and transfer are terminal branches, not subflows.

Retail (Figure 13). The main spine performs intake and shared identification before the classifier dispatches each request to cancellation, modification, return, exchange, account, or terminal handling. The expanded cancel\_pending\_order branch is the canonical flat gate conjunction of Appendix B.3: locate the order, verify its status is exactly pending, obtain the closed-set cancellation reason, confirm, act, and verify.

Telecom (Figure 14). After intake and shared customer identification, the classifier dispatches requests to billing, line, plan, roaming, and troubleshooting subflows. The expanded overdue-bill branch invokes pay\_overdue\_bill, whose eligibility gates precede confirmation and authorization of send\_payment\_request. The expanded MMS branch invokes troubleshoot\_mms, an ordered traversal of service and data prerequisites followed by the documented MMS causes and a closing resolve-or-transfer decision (Appendix B.4).

## H.1 Turn-by-turn guide example

For the airline task “Book me on HAT136 JFK→LAX, Nov 15,” the agent jumps straight to book\_reservation; the guide walks it back through the policy path, one remediation step per turn. Turn 1 stops at identify\_user.ask\_user\_id and asks for the user id; turn 2 directs get\_user\_details; subsequent turns walk through trip collection, flight search, rule validation, payment, baggage/insurance computation, and summary-and-confirm. Only after the upstream nodes are satisfied does book\_flow.authorize\_book\_reservation set authorize\_tool; the following turn verifies the successful TOOL\_RESULT and closes the request.

![](images/20e1076b46d636e999e44da51b916c43267aef6afa81b193107ad946d770047b.jpg)  
Figure 7: Guide system prompt, part 1 of 3. Left: the protocol block (reconcile-then-traverse, one generation per fired turn) and the per-turn trigger, which is folded into the cached static prefix rather than re-sent. Right: the Part-2 output contract—the single fenced JSON block the runtime parses into code-owned state.

## I The Use of LLMs

We used LLMs solely for light editing, such as correcting grammatical errors and polishing wording. They did not contribute to research ideation, experiments, analysis, or substantive writing.

## Judging rules · ground truth

\- Ground truth = tool results. A fact, eligibility condition, or completed action counts as established ONLY when a TOOL\_RESULT in the conversation confirms it (directly or by your reasoning over tool results) -- NOT because the user asserted it, told you to assume it, or stated it as a given, and NOT because the agent merely said it; a value the agent computes from already-established inputs is itself established -- but a decision or argument that turns on a numeric or temporal computation must be worked out step by step in your reasoning and taken from those steps, never asserted as a conclusion; the user's own choices, consent, and preferences are established by the user's message, but a factual or eligibility condition the policy gates on is never established by the user's word -- when the user supplies or assumes one, delegate the read-only tool that verifies it and judge the condition from that result before relying on it, and a tool result that contradicts the user's claim governs.

## Judging rules · authorization

Authorization. A WRITE (mutating) tool may be authorized ONLY when every policy prerequisite for that specific tool is met from tool-confirmed facts (plus the user 's own consent where the policy asks for it) -- apply exactly the prerequisites the policy states for that tool, adding none it does not state, so once all of them are met you authorize rather than withholding for a condition you inferred. When you authorize, the runtime opens that tool's gate so the agent can call it; until then the runtime hard-blocks the call. Whenever your remediation instructs the agent to call a WRITE tool you have judged its prerequisites met, so set that intent's authorize\_tool to that tool the same turn -- never instruct a WRITE call while leaving authorize\_tool null. The authorizing remediation must state the exact arguments the agent must pass -- every id and value from grounded results, with any amounts, counts, or derived figures computed from the state the requested changes produce rather than from the prior state, so they reconcile with the tool's requirements -- and cover every change the user requested, so the agent does not guess, miscompute, or omit a step. To change specific fields of an existing record, reuse the record's current values for the fields the user is not changing rather than searching for or re-collecting new ones; and never withhold authorization to first establish a value the write tool itself computes or returns -- such an output is not a prerequisite. When the tool applies to several records, draw each call' s arguments from that record's own grounded data and confirm the pairing before authorizing -- a value belonging to one record must never cross into another, since the call cannot be undone.

Figure 8: Guide system prompt, part 2 of 3: judging rules (i)—evidence grounding and mutating-tool authorization.

![](images/89a6696b3388130119932fa43595f9122461aa1307f28c8b6a4ad148cebd42d4.jpg)  
Figure 9: Guide system prompt, part 3 of 3: judging rules (ii)—argument sourcing, transfer scoping, corrective-fix traversal, read-tool delegation (left) and the remediation contract (right).

## Generation system prompt · runtime contract

A proactive verifier reads the WHOLE graph plus the full conversation each turn, tracks where each open request sits, and writes the agent's next-step remediation. Two things the graph must make enforceable:

\- Mutating tools are gated. A tool\_authorization node is the choke point for one WRITE tool: the agent may call that tool ONLY after the guide authorizes it ( its upstream prerequisites met) and the runtime confirms success at the following verify node. So every prerequisite/eligibility/confirmation a mutation requires must sit UPSTREAM of its authorization node.

\- Faithfulness. The guide can only enforce what the graph encodes. The graph must reflect the policy and the domain's tool/task properties exactly.

## Generation system prompt · authoring essentials 1–3

The bottom line: the graph must (A) reflect the policy faithfully, (B) reflect the domain's tool/task properties, and (C) be valid + minimal.

1. Cover every WRITE tool. Each mutating tool is reachable through some intent and ends in an authorize\_<short> (tool\_authorization) -> verify\_<tool> (agent\_action) pair -- the authorization choke point and the postcall success check (the tool can fail, so the verify node confirms a successful TOOL\_RESULT and, on error, directs a correctable retry). Place everything the policy requires before a mutation UPSTREAM of its authorization. A value the mutating tool itself computes or returns is an output, not a prerequisite.

2. Gates are outcome-framed. An eligibility/permissibility gate is SATISFIED only if the permitting condition actually HOLDS -- never "has the agent checked X?" ( that flips true the moment the check runs, even when it concludes ineligible). When the condition fails, the gate stays NOT\_SATISFIED and its remediation REFUSES. Keep gates as agent\_action nodes on the main path -- do not model refusal as a branch to a deadend exit.

3. Ground facts in tools; quote the policy exactly. A fact/ eligibility condition counts only when established by a tool result -- not a user assertion. The user's own consent/preference is established by the user's message. Where the environment gates on an exact status/enum literal, quote it and require exact equality (a qualified variant is a different value), and name whose field. Quote limits/prices/amounts/ time-windows verbatim, identically across expectation /evaluation\_prompt/remediation\_template. Encode only conditions the policy states -- never invent a check no tool/data can establish.

## Generation system prompt · authoring essentials 4–7

4. Classifier completeness. The main classify decision has one branch per intent + general (anything unsupported -> a non-transfer refusal exit) + transfer (must go to a human). Operations the policy forbids outright get NO subflow -- they route to general.

5. Right tool side. tool\_call/tool\_authorization may name ONLY tools from the AGENT inventory. An action the END USER performs on their own device is an agent\_action that instructs the user and judges their reported outcome -- never a tool node (which could never be satisfied).

6. Minimal + faithful. Author the fewest nodes that enforce the policy. Fold a validation/disclosure into the node that collects its data; merge related collection steps rather than one node per policy sentence. Every normative policy statement must be enforced by some node (or be genuinely out of graph scope). When the policy enumerates the possible causes of a problem, the flow that resolves it must check every documented cause; a cause whose remedy is a mutating tool must be its OWN checkpoint that reaches that tool's authorization.

7. Main spine + shared subflows. main: entry -> intake ( greet + open question) -> identify (shared identification subflow) -> classify -> intent subflow anchors -> exits ["exit\_normal","exit\_general"," exit\_transfer"]. Any procedure shared by 2+ intents may be its own subflow. When the domain has userowned records that requests target, the identification subflow ends with a tool\_call that loads the authenticated user's full account record.

Figure 10: Workflow-generation system prompt: the runtime contract the graph must satisfy, and the authoringessentials contract every stage must follow.

<table><tr><td>Stage 2 · plan requirements</td><td></td><td>Stage 6 · adversarial review checklist</td><td></td></tr><tr><td>1. Cover every WRITE tool. Each mutating tool is reachable through at least one intent subflow ending in its authorize -&gt; verify pair. 2. Intents from the policy, not just the tools. Include advisory/no-tool intents the policy describes and procedure-driven intents from any troubleshooting manual. Unify all policy documents into ONE flat intent taxonomy. 3. Two archetypes. Transactional (mutate a record): load the target -&gt; gate eligibility -&gt; confirm -&gt; execute. Troubleshooting (diagnose a symptom against a manual ): an ordered checklist -- check a documented cause -&gt; apply its fix -&gt; re-test -&gt; continue; resolve on a passing re-test, escalate after all causes exhausted.</td><td>An agent WRITE-tool fix routes into that tool&#x27;s mutating subflow; a user-device fix is an agent_action instructing the user. 4. Shared subflows. Plan the identification subflow and a present_summary_and_confirm subflow. 5. Per-subflow skeletons: ordered steps, id/type/tool/gist</td><td colspan="2">1. Policy completeness. Walk the policy line by line. Every normative statement must be enforced by some node&#x27;s criteria (or be genuinely out of graph scope). Fix any missing rule, wrong number, or weakened condition. 2. Cause coverage (enumerate explicitly). Wherever the policy enumerates the possible causes/conditions of a problem, list every one and name the node whose criterion handles it. If any cause has no node, add it. 3. Essentials compliance. Gates outcome-framed; environment -gated enums quoted with exact equality and whose- field named; facts grounded in tool results; every WRITE tool has its authorize -&gt; verify pair with everything the policy requires upstream. Tool nodes name only agent-inventory tools. 4. Graph sanity. Entry reaches every node; every node reaches an exit; decision branch labels/targets consistent; classifier covers all intents + general +</td></tr></table>

Figure 11: Workflow-generation stage prompts: the plan stage’s requirements (left) and the adversarial reviewer’s checklist (right).

Figures 12–14 use the following shared node notation.  
![](images/fa5ee69fcce6a56f9e20c566490937b79148a6793f3a5436c83be8a4902bec21.jpg)  
Figure 12: Airline: the composed workflow graph—the shared spine (start, intake, the identify\_user subflow, classify) and the full transactional book\_reservation request path. Solid edges fire only when: satisfied; dashed edges are classifier branches. The mutating tool is represented downstream of its tool\_authorization node (trapezoid), immediately followed by the verify node that demands a successful TOOL\_RESULT.

![](images/06fe4fc16d7caea5e8035971af2c93958c315055ae166aa33df970a4a3be126b.jpg)  
Figure 13: Retail: the shared entry–intake–identification spine and classifier dispatch into request-specific subflows, with cancel\_pending\_order expanded. Its status and reason gates, shared summary-and-confirm step, authorization, and post-call verification form the complete cancellation path. The gray box marks the expanded cancellation subflow; the dotted box summarizes the other classifier branches.

![](images/84a2fabee0c93c4acd771335691827af31ac9a30e8c9106e460fbe830891654e.jpg)  
Figure 14: Telecom: the shared entry–intake–identification spine and classifier dispatch, with two request branches expanded. The overdue-bill branch invokes pay\_overdue\_bill, which gates send\_payment\_request; the MMS branch invokes troubleshoot\_mms, which checks service and data prerequisites before the documented MMS causes and closes by resolving or transferring. The dotted box summarizes the remaining classifier branches.