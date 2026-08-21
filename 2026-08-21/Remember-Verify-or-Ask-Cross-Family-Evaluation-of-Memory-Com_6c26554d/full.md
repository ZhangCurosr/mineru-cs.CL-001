# Remember, Verify, or Ask? Cross-Family Evaluation of Memory Commitment in LLM Agents

Baichuan Li

Department of Operations Research and Engineering Management

Southern Methodist University

Dallas, USA

baichuanl@smu.edu

Junyi Yao Zihao Zheng

Department of Computer Science & Engineering

Washington University in St. Louis

St. Louis, USA

j.yao@wustl.edu z.zihaogary@wustl.edu

Abstract—Persistent memory can personalize an LLM agent, but an incorrect durable update can silently distort future behavior. We study the memory–clarification boundary: whether interaction-derived information should be persisted, used only in the current context, re-verified, or clarified with the user. MCB contains 140 primary scenarios, split into 70 development and 70 held-out items, plus a separate 70-item contrast set. It evaluates both action labels and structured tool-call selection. Two non-authors independently label the 70 held-out primary and 70 contrast items (97.1% agreement, Cohen’s κ = 0.962); a blind third resolves four disagreements, replacing eight author labels by non-author majority. Across Claude and Qwen, models verify changing facts more reliably than they ask users to resolve ambiguity. Bare Qwen asks on 0/12 clarification items while verifying 12/18 freshness items. Few-shot prompting raises accuracy from 0.557 to 0.771 (paired ∆ = +0.214, Holm-adjusted exact McNemar $p _ { \mathrm { H } } ~ = ~ 0 . 0 0 2 )$ , yet clarification recall remains 0.333. The policy prompt reduces erroneous persistence from 0.243 to 0.100 $\begin{array} { r } { ( p _ { \mathrm { H } } = 0 . 0 3 8 ) . } \end{array}$ , although its accuracy gain is not significant. Label–tool agreement is 57% for each Claude model and 23% for Qwen; Qwen accuracy falls from 0.557 to 0.343 $( p _ { \mathrm { H } } = 0 . 0 4 7 )$ . Memory evaluation must test both stated decisions and tool-call choices.

Index Terms—LLM agents, long-term memory, clarification, benchmark, tool use, reproducibility

## I. INTRODUCTION

LLM agents increasingly retain interaction history to personalize later behavior and support long-horizon tasks [1], [2]. Memory formation, however, is not uniformly beneficial. A temporary request should not become a standing preference; a service status may become stale; one tool failure may be noise; and an underspecified correction may require a question before it is generalized. The critical capability is therefore not only recall, but commitment: deciding what may safely influence future behavior.

We call this decision the memory–clarification boundary. Given an acquired candidate update and a later reuse context, an agent chooses among four operationally distinct actions: persist, ephemeral use, verify against the world, or clarify with the user. Verification and clarification are not interchangeable: the world is the source of truth for changing facts, whereas the user is the source of truth for intent and scope.

We introduce the Memory–Clarification Boundary benchmark (MCB). It contains 140 primary scenarios, deterministically split into 70 development and 70 held-out items, plus a separate 70-item contrast set for robustness analysis. MCB tests both stated choices and structured tool-call selection. Four reference systems, two pinned Claude models, and a local Qwen3.5-9B model are evaluated on the same 70 independently adjudicated held-out items. Every model is tested with a bare prompt, an explicit five-rule policy, and four development-set demonstrations. Shared items permit exact paired tests instead of conclusions drawn from overlapping independent intervals.

We test whether the observed failure generalizes beyond a single model provider. We also separate two claims often conflated as “prompt sensitivity”: a prompt can improve total accuracy, or can reshape a safety-relevant behavior such as erroneous persistence without changing total accuracy. Qwen exhibits both after multiplicity correction. Few-shot prompting improves accuracy; the policy prompt reduces over-memory. Neither eliminates under-asking.

Our contributions are: (1) an auditable benchmark for memory commitment with non-author-adjudicated evaluation labels, anti-shortcut items, and behavior-specific metrics; (2) a two-family experiment covering Claude and Qwen under matched interventions; (3) a tool-call-selection variant that requires a concrete memory entry, local-use note, source query, or user question rather than an action label; and (4) a reproducible artifact containing data, blind annotations, prompts, runners, item-level predictions, model metadata, tests, and paired analyses.

## II. RELATED WORK

LongMemEval, LoCoMo, MemoryBank, and MemBench evaluate long-term conversational memory, knowledge updates, retrieval, and abstention [3], [4], [5], [6]. Recent benchmarks add real-dialogue memory lifecycles and penalties for obsolete-memory reuse [7], [8]. The closest work moves beyond recall: PerMemBench learns a binary session-level storage gate [9]; Memory-R1 learns ADD/UPDATE/DELETE/NOOP operations [10]; and Mem2ActBench tests whether retrieved memory grounds later tool parameters [11]. MCB is complementary: it jointly distinguishes durable storage, local use, world verification, and user clarification at candidatecommitment time.

Table I compares explicit evaluation targets; a dash means “not a central scored capability,” not that a dataset can never contain such an event. Unlike retrieval benchmarks, MCB supplies the candidate update and isolates the commitment decision. Unlike binary storage gating, it separates two sources of uncertainty: the world and the user.

Interactive benchmarks such as τ-bench and $\tau ^ { 2 } .$ -bench show that static answers can diverge from tool-mediated agent behavior [13], [14]. MCB-Act applies this insight at the memory boundary by replacing action labels with structured tool-call choices and retaining their arguments for audit.

Clarification is an information-gathering action, not merely a class label. CLAMBER documents broad failures to identify and clarify ambiguous needs; related work uses uncertainty or expected value to decide when to ask [12], [15], [16]. MCB instantiates a source-of-truth distinction: clarify queries the user, verify queries the world, and ephemeral limits commitment.

## III. BENCHMARK AND METHOD

## A. Task and Labels

Each item has an acquire context, a candidate update, and a later reuse context. The gold action is assigned using released rules:

• persist: explicitly durable preferences, authorized policies, or stable facts;

• ephemeral: information scoped to one artifact, session, or time window;

• verify: changing world state or a single noisy signal that must be rechecked;

• clarify: unresolved referent, durability, scope, conflict, or second-hand preference for which the user is authoritative.

When persist and a weaker action remain tied, the rules prefer the weaker commitment. This encodes an asymmetric cost: an unnecessary question is visible and recoverable, while a wrong durable update can remain silent.

The benchmark covers stable and episodic preferences, freshness-sensitive facts, one-off corrections, policy constraints, ambiguous updates, and noisy failures (20 items each). After audited test labels are applied, its action distribution is 38 persist, 40 ephemeral, 33 verify, and 29 clarify. Each item has a rationale. Every scenario category contains multiple actions, so the category is not the label. Eight lexical traps place words such as “always” or “today” in contexts where their usual heuristic action is wrong.

Items are deterministically split 70/70 by sorted identifier within category. The author-labeled development split supplies the four demonstrations; all reported results use independently adjudicated held-out gold. Two non-authors independently labeled the 70 primary-test and 70 contrast items while blinded to author labels, categories, rationales, model outputs, and each other. Their full-set agreement was 0.971 $\begin{array} { r } { ( \kappa = 0 . 9 6 2 ) ; } \end{array}$ agreement was 0.943 $( \kappa = 0 . 9 2 3 )$ on primary and 1.000 on contrast. A blind third non-author broke four primary ties. Non-author agreement/majority changed eight primary labels and no contrast labels; all metrics were then recomputed. Models never receive category or rationale. Majority Action predicts the most frequent audited test action (ephemeral); the category-majority oracle additionally uses hidden category. Both inspect test labels and are distribution-only leakage diagnostics, not deployable baselines.

## B. Metrics and Statistics

We report accuracy with a 2,000-resample percentilebootstrap 95% interval, macro-F1, and class-sensitive measures. Over-memory (OM) is the fraction of all items wrongly predicted persist. Under-memory is the miss rate on goldpersist items. Clarification (Clar.) and verification (Ver.) are recalls on their respective gold items. Invalid outputs remain an always-wrong fifth state rather than being mapped to a favorable class.

All systems share the same test items. Headline differences therefore use paired-bootstrap intervals and an exact two-sided McNemar test on discordant correctness. For behavior rates we apply the same exact paired test to item-level events (e.g., erroneous persistence). We control familywise error with Holm correction separately across six primary prompt comparisons for accuracy, the corresponding six OM comparisons, three tool-call comparisons for accuracy, and two contrast-validation comparisons for accuracy. Main inferential claims report p<sub>H</sub>; other exact tests are descriptive and unadjusted. We call adjusted $p < 0 . 0 5$ significant. Per-category cells contain only 10 items and are not used for firm ranking claims.

## C. Label and Tool-Call Evaluation

Label-mode prompts request JSON containing one of the four actions. Bare defines the actions only. Policy adds five commitment rules, including the weaker-action tie-breaker. Few-shot adds one development example per action.

MCB-Act removes the label vocabulary. The model must emit one structured call: memory\_write, use\_now, check\_source, or ask\_user. Each takes a required string argument, checked by deterministic minimum-content and relevance rules. The selected tool maps to an action for scoring, and the payload is retained for audit. Thus MCB-Act evaluates tool-call selection, not downstream tool execution. It uses only the bare condition to isolate whether translating an unassisted stated choice into a concrete call changes behavior; policyconditioned tool calls are left to future work.

## IV. EXPERIMENTAL SETUP

Reference systems are Always-Persist, Majority Action, a temporal/scope keyword heuristic, and the category-majority oracle. Claude Haiku 4.5 (claude-haiku-4-5-20251001) and Claude Sonnet 4.6 (claude-sonnet-4-6) are called independently per item with tools disabled in label mode. Served identifiers are logged.

The cross-family model is the post-trained Qwen3.5-9B model [17], run locally through Ollama 0.24.0 [18]. We use a Q4 K M quantization (9.7B parameters), model digest

TABLE I  
POSITIONING BY EXPLICIT SCORED TARGET. “STRUCTURED ACTION” INCLUDES TOOL SELECTION OR PARAMETERIZED MEMORY OPERATIONS; MCB-ACT SCORES TOOL-CALL SELECTION BUT DOES NOT EXECUTE DOWNSTREAM EFFECTS.
<table><tr><td>Benchmark</td><td>Recall/reuse</td><td>Storage gate</td><td>Ask user</td><td>Check world</td><td>Structured action</td></tr><tr><td>LongMemEval / LoCoMo [3], [4]</td><td>√</td><td>一</td><td></td><td></td><td>一</td></tr><tr><td>MemBench [6]</td><td>√</td><td></td><td></td><td></td><td>一</td></tr><tr><td>PerMemBench [9]</td><td>√</td><td>√</td><td>一</td><td></td><td>一</td></tr><tr><td>CLAMBER [12]</td><td></td><td>一</td><td>√</td><td></td><td></td></tr><tr><td>Mem2ActBench [11]</td><td>√</td><td></td><td></td><td></td><td>√</td></tr><tr><td>MCB (ours)</td><td>一</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

TABLE II  
ILLUSTRATIVE MCB DECISIONS. EACH LABEL REFLECTS AUTHORITY OR COMMITMENT SCOPE RATHER THAN A LEXICAL TOPIC.
<table><tr><td>Acquire / candidate update</td><td>Later reuse</td><td>Gold</td><td>Boundary cue</td></tr><tr><td>“I always prefer dark mode.&quot;</td><td>Configure a new workspace</td><td>Persist</td><td>Explicitly durable and reusable preference</td></tr><tr><td>“Use APA for this report.&quot;</td><td>Prepare an unrelated report</td><td>Ephemeral</td><td>Scope is limited to one artifact</td></tr><tr><td>&quot;Restaurant X is open tonight.&quot;</td><td>Rely on it one month later</td><td>Verify</td><td>The world state can change</td></tr><tr><td>“Make it like last time.&quot;</td><td>Several prior artifacts fit</td><td>Clarify</td><td>Only the user can resolve the referent</td></tr></table>

6488c96fa5fa...eda893ea7, temperature 0, seed 13, disabled thinking, and output limits of 96 tokens in label mode and 128 in act mode. The native API makes these settings explicit. Each condition makes 70 isolated calls; all 280 Qwen outputs parse successfully. Exact prompts, complete digest, runtime metadata, raw outputs, and token counts accompany each prediction.

## V. RESULTS AND DISCUSSION

## A. Cross-Family Under-Asking

Table III shows a common asymmetry. In Claude labelmode runs, verification recall is 0.889–1.000 while clarification recall is 0.500–0.750. Qwen makes the distinction sharper: bare Qwen verifies 12/18 freshness items but clarifies 0/12 ambiguous items. Its clarification cases are instead mapped to persist (7), verify (4), or ephemeral (1). This is a sourceof-truth confusion: the model may recognize uncertainty yet consult the world rather than the user who alone can resolve intent. It otherwise silently commits an interpretation.

This is cross-family evidence for under-asking, but not for identical error policies. Bare Qwen over-persists on 0.243 of all items versus 0.029 for bare Haiku (descriptive paired event $p _ { \mathrm { r a w } } < 0 . 0 0 1 )$ . The useful generalization is therefore narrow: both families under-ask; the alternative action they choose is model-dependent.

## B. Prompt and Policy Sensitivity

Table IV separates total accuracy from behavioral change. Qwen few-shot improves accuracy by 0.214 $( p _ { \mathrm { { H } } } = \mathrm { { 0 } } . 0 0 2 )$ fixes 16 bare errors while breaking one correct answer, and increases clarification recall from 0 to 0.333. Nevertheless it still misses 8/12 clarification opportunities, so examples mitigate but do not remove under-asking.

Qwen policy improves accuracy by only 0.071, which is not statistically separable from zero $( p _ { \mathrm { { H } } } = 0 . 5 3 9 )$ . Its policy is nonetheless behaviorally different: erroneous persistence falls by 0.143, from 17/70 to 7/70. On paired item-level OM events, the policy removes 11 bare errors and introduces one $( p _ { \mathrm { { H } } } = 0 . 0 3 8 )$ . It moves uncertainty primarily to verification (recall 0.667 to 0.944), not to the user (clarification 0 to 0.083). A headline accuracy alone would miss this safety-relevant intervention effect.

## C. Exploratory Contrast Validation

To assess robustness beyond the primary 70-item test, we froze a separate 70-item contrast extension before inference. It contains 35 evidence-flip pairs (10 items/category), including seven cue-conflicting traps; the development set and demonstrations are unchanged. Both non-author annotators independently reproduced all 70 contrast labels $( \kappa = 1 . 0 0 0 )$ On the combined 140 Qwen items, bare, policy, and fewshot accuracy is 0.614, 0.757, and 0.843. Relative to bare, policy gains 0.143 $( p _ { \mathrm { { H } } } < 0 . 0 0 1 )$ and few-shot gains 0.229 $( p _ { \mathrm { H } } < 0 . 0 0 1 )$ . Clarification recall remains the weakest class $( 0 . 0 7 4 / 0 . 4 0 7 / 0 . 5 1 9 )$ . Because these rule-authored templates align closely with the explicit policy rules, we retain the extension as a controlled sensitivity check rather than claim naturalistic external validity.

The Claude pattern is also model-dependent. Haiku’s policy and few-shot gains survive the six-test correction $( p _ { \mathrm { H } } = 0 . 0 0 2 $ and 0.047); Sonnet’s do not. Across three models, the benchmark therefore measures a prompt-conditioned commitment policy rather than a fixed model trait.

## D. Stated Choices Do Not Reliably Predict Tool-Call Choices

MCB-Act addresses the concern that selecting a label may not predict tool-call behavior. For both Claude models, raw action agreement between bare label mode and tool-call mode is 0.571: 30/70 decisions change when labels become tools. Sonnet accuracy drops from 0.814 to 0.529 $( \Delta = - 0 . 2 8 6$ $p _ { \mathrm { H } } ~ < ~ 0 . 0 0 1 )$ ; Haiku’s drop from 0.629 to 0.514 is not significant $( p _ { \mathrm { H } } = 0 . 0 5 7 )$ . Qwen supplies the cross-family test: agreement is only 0.229 and accuracy drops from 0.557 to

TABLE III  
HELD-OUT MCB RESULTS ON NON-AUTHOR-ADJUDICATED GOLD (n = 70). ACCURACY INCLUDES A BOOTSTRAP 95% CI. OM IS ERRONEOUS PERSISTENCE OVER ALL ITEMS; CLAR. AND VER. ARE RECALLS ON 12 AND 18 RELEVANT GOLD ITEMS. ALL INVALID-OUTPUT RATES ARE ZERO. ACT ROWS EMIT ONE TOOL-CALL OBJECT WITHOUT A POLICY PROMPT.
<table><tr><td>System</td><td>Condition</td><td>Accuracy [95% CI]</td><td>Macro-F1</td><td>OM↓</td><td>Clar.↑</td><td>Ver.↑</td></tr><tr><td>Always Persist</td><td>一</td><td>.257 [.157,.371]</td><td>.102</td><td>.743</td><td>.000</td><td>.000</td></tr><tr><td>Majority Action</td><td></td><td>.314 [.214,.429]</td><td>.120</td><td>.000</td><td>.000</td><td>.000</td></tr><tr><td>Keyword</td><td></td><td>.257 [.157,.371]</td><td>.228</td><td>.071</td><td>.833</td><td>.056</td></tr><tr><td>Category oracle</td><td></td><td>.800 [.700,.900]</td><td>.792</td><td>.071</td><td>.667</td><td>.889</td></tr><tr><td>Claude Haiku 4.5</td><td>bare</td><td>.629 [.514,.743]</td><td>.615</td><td>.029</td><td>.500</td><td>.944</td></tr><tr><td></td><td>policy</td><td>.857 [.771,.929]</td><td>.846</td><td>.014</td><td>.750</td><td>.944</td></tr><tr><td>Claude Sonnet 4.6</td><td>few-shot</td><td>.757 [.643,.857]</td><td>.732</td><td>.057</td><td>.500</td><td>.944</td></tr><tr><td></td><td>bare</td><td>.814 [.714,.900]</td><td>.790</td><td>.057</td><td>.500</td><td>.889</td></tr><tr><td></td><td>policy</td><td>.843 [.757,.929]</td><td>.831</td><td>.014</td><td>.667</td><td>1.000</td></tr><tr><td>Qwen3.5-9B</td><td>few-shot</td><td>.814 [.729,.900]</td><td>.796</td><td>.043</td><td>.583</td><td>.944</td></tr><tr><td></td><td>bare</td><td>.557 [.443,.671]</td><td>.450</td><td>.243</td><td>.000</td><td>.667</td></tr><tr><td></td><td>policy</td><td>.629 [.514,.743]</td><td>.542</td><td>.100</td><td>.083</td><td>.944</td></tr><tr><td></td><td>few-shot</td><td>.771 [.671,.871]</td><td>.726</td><td>.129</td><td>.333</td><td>.833</td></tr><tr><td>Claude Haiku 4.5</td><td>act</td><td>.514 [.400,.629]</td><td>.456</td><td>.143</td><td>.583</td><td>.778</td></tr><tr><td>Claude Sonnet 4.6</td><td>act</td><td>.529 [.414,.643]</td><td>.531</td><td>.100</td><td>.500</td><td>.556</td></tr><tr><td>Qwen3.5-9B</td><td>act</td><td>.343 [.243,.457]</td><td>.236</td><td>.057</td><td>.083</td><td>.056</td></tr></table>

TABLE IV

SELECTED EXACT PAIRED COMPARISONS. ∆ IS ACCURACY UNLESS MARKED OM; INTERVALS ARE PAIRED-BOOTSTRAP 95% CIS; p<sub>H</sub> IS HOLM-ADJUSTED WITHIN THE FAMILIES DEFINED IN SEC. III-B.
<table><tr><td>Comparison</td><td>∆ [95% CI]</td><td>PH</td></tr><tr><td>Haiku policy-bare</td><td>+.229 [.114,.343]</td><td>.002</td></tr><tr><td>Haiku few-shot-bare</td><td>+.129 [.043,.214]</td><td>.047</td></tr><tr><td>Sonnet policy-bare</td><td>+.029 [-.086,.143]</td><td>1.000</td></tr><tr><td>Sonnet few-shot-bare</td><td>+.000 [-.100,.100]</td><td>1.000</td></tr><tr><td>Qwen policy-bare</td><td>+.071 [-.014,.157]</td><td>.539</td></tr><tr><td>Qwen few-shot-bare</td><td>+.214 [.114,.329]</td><td>.002</td></tr><tr><td>Qwen policy-bare (OM)</td><td>-.143</td><td>.038</td></tr><tr><td>Sonnet act-bare</td><td>-.286 [-.414,-.143]</td><td>&lt;.001</td></tr><tr><td>Qwen act-bare</td><td>-.214 [-.386,-.029]</td><td>.047</td></tr></table>

0.343 $( \Delta = - 0 . 2 1 4 , p _ { \mathrm { H } } = 0 . 0 4 7 )$ . It calls use\_now on 54/70 items, reducing over-memory but collapsing verification recall from 0.667 to 0.056. Thus tool-call selection changes all three models, but with model-specific action biases. Every emitted argument passes the deterministic well-formedness rules, including all clarification questions on gold-clarify items. The bottleneck is tool choice, not malformed arguments.

## E. What the Benchmark Does and Does Not Establish

Always-Persist obtains 0.257 accuracy and 0.743 OM, establishing that unconditional durable commitment conflicts with most item-level decisions. Majority Action reaches 0.314 but has only 0.120 macro-F1, exposing the weakness of a frequency-only rule. Neither shows that every system retaining raw history will fail: tiering, expiration, and retrieval filters can implement a weaker effective commitment. Similarly, the category oracle’s 0.800 shows that scenario type carries substantial signal; bare models do not reliably exceed it. We therefore emphasize paired intervention effects and classspecific errors, not a system leaderboard.

MCB-Act is a more behaviorally concrete test than naming a label, but it records rather than executes the selected store, source, or user-facing operation. No simulated user answers the question, no verified source changes, and no downstream task score is observed. The results establish a label–toolselection gap, not a quantified improvement in end-to-end utility.

## VI. LIMITATIONS, ETHICS, AND REPRODUCIBILITY

The primary 70-item cross-family test yields wide intervals; 10-item category cells and eight traps support qualitative analysis only. Evaluation labels now follow a completed blind nonauthor audit, but scenarios and the labeling rules were authorwritten. The perfectly reproduced contrast labels may reflect close rule–template alignment, so that extension remains a controlled sensitivity check. Development demonstrations retain author labels. Qwen is one 9B quantized checkpoint; quantization, serving stack, and disabled thinking are part of the evaluated system. Act mode uses only the bare intervention and one checkpoint per family. All scenarios are synthetic and in English.

The benchmark contains no personal records. Its intended use is to reduce privacy and personalization failures from inappropriate durable memory. The artifact includes 140 primary items, 70 contrast-validation items, all frozen annotations and majority decisions, original and audited labels, prompts, runners, tests, model outputs, exact metadata, and scripts that regenerate every table and paired test. Deterministic components run without model access; stored predictions permit statistical reproduction. Generative AI tools were used as experimental subjects and for implementation/language assistance; numerical claims in this paper are generated from the released item-level outputs and deterministic analysis code.

## VII. CONCLUSION

Memory commitment is a distinct agent capability: deciding whether to remember, limit, verify, or ask. Cross-family label experiments confirm severe under-asking and show that prompts can change both accuracy and safety-relevant action distributions. Cross-family tool-call selection further reveals that stated decisions do not reliably survive translation into action choices, although the resulting bias is model-dependent. Evaluation of persistent-memory agents should therefore report clarification and over-memory explicitly, use paired tests, and elicit structured behavior rather than labels alone.

## REFERENCES

[1] C. Packer, S. Wooders, K. Lin, V. Fang, S. G. Patil, I. Stoica, and J. E. Gonzalez, “MemGPT: Towards LLMs as operating systems,” 2023, arXiv:2310.08560.

[2] P. Chhikara, D. Khant, S. Aryan, T. Singh, and D. Yadav, “Mem0: Building production-ready AI agents with scalable long-term memory,” 2025, arXiv:2504.19413.

[3] D. Wu, H. Wang, W. Yu, Y. Zhang, K.-W. Chang, and D. Yu, “LongMemEval: Benchmarking chat assistants on long-term interactive memory,” in International Conference on Learning Representations (ICLR), 2025.

[4] A. Maharana, D.-H. Lee, S. Tulyakov, M. Bansal, F. Barbieri, and Y. Fang, “Evaluating very long-term conversational memory of LLM agents,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL). Association for Computational Linguistics, 2024, pp. 13 851–13 870.

[5] W. Zhong, L. Guo, Q. Gao, H. Ye, and Y. Wang, “MemoryBank: Enhancing large language models with long-term memory,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 17, 2024, pp. 19 724–19 731.

[6] H. Tan, Z. Zhang, C. Ma, X. Chen, Q. Dai, and Z. Dong, “MemBench: Towards more comprehensive evaluation on the memory of LLM-based agents,” in Findings of the Association for Computational Linguistics: ACL 2025. Association for Computational Linguistics, 2025, pp. 19 336–19 352.

[7] J. Xiao, X. Yu, C. Wang, W. Zheng, X. Lin, K. Liu, H. Ding, Y. Zhang, W. Wang, F. Feng, and X. He, “AlpsBench: An LLM personalization benchmark for real-dialogue memorization and preference alignment,” 2026, arXiv:2603.26680.

[8] M. N. Uddin, K. Shubham, E. Blanco, C. Baral, and G. Wang, “From recall to forgetting: Benchmarking long-term memory for personalized agents,” in Findings of the Association for Computational Linguistics: ACL 2026. Association for Computational Linguistics, 2026, arXiv:2604.20006.

[9] Y. In, W. Kim, S. Park, K. Yoon, and C. Park, “Personalize-thenstore: Benchmarking and learning personalized memory for long-horizon agents,” 2026, arXiv:2605.25535.

[10] S. Yan, X. Yang, Z. Huang, E. Nie, Z. Ding, Z. Li, X. Ma, H. Schutze,¨ V. Tresp, and Y. Ma, “Memory-R1: Enhancing large language model agents to manage and utilize memories via reinforcement learning,” 2025, arXiv:2508.19828.

[11] Y. Shen, K. Li, W. Zhou, and S. Hu, “Mem2ActBench: A benchmark for evaluating long-term memory utilization in task-oriented autonomous agents,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics. San Diego, California, USA: Association for Computational Linguistics, 2026, pp. 8173–8190.

[12] T. Zhang, P. Qin, Y. Deng, C. Huang, W. Lei, J. Liu, D. Jin, H. Liang, and T.-S. Chua, “CLAMBER: A benchmark of identifying and clarifying ambiguous information needs in large language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2024, pp. 10 746–10 766.

[13] S. Yao, N. Shinn, P. Razavi, and K. Narasimhan, “τ-bench: A benchmark for tool-agent-user interaction in real-world domains,” 2024, arXiv:2406.12045.

[14] V. Barres, H. Dong, S. Ray, X. Si, and K. Narasimhan, “τ<sup>2</sup>-bench: Evaluating conversational agents in a dual-control environment,” 2025, arXiv:2506.07982.

[15] A. Testoni and R. Fernandez, “Asking the right question at the right time:´ Human and model uncertainty guidance to ask clarification questions,” in Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL). Association for Computational Linguistics, 2024, pp. 258–275.

[16] M. J. Q. Zhang and E. Choi, “Clarify when necessary: Resolving ambiguity through interaction with LMs,” in Findings of the Association for Computational Linguistics: NAACL. Association for Computational Linguistics, 2025, pp. 5541–5558.

[17] Qwen Team, “Qwen3.5-9B model card,” Hugging Face model repository, 2026, accessed: 2026-08-13. [Online]. Available: https: //huggingface.co/Qwen/Qwen3.5-9B

[18] Ollama, “Ollama: Run large language models locally,” Software, 2026. [Online]. Available: https://ollama.com