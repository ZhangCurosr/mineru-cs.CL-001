# LATENT ON-POLICY SELF-DISTILLATION

Guibin Zhang<sup>1,∗</sup>, Jiayang Lyu<sup>2,∗</sup>, Ran Sun<sup>2</sup>, Xinlei Yu<sup>1</sup>,

Haoyu Zhao<sup>1</sup>, Qibing Ren<sup>3†</sup>, Shuicheng Yan<sup>1†</sup>

<sup>1</sup>National University of Singapore <sup>2</sup>Beijing University of Posts and Telecommunications

<sup>3</sup>Shanghai Jiao Tong University <sup>†</sup> Corresponding <sup>∗</sup> Equal Contribution

§ GitHub: github.com/bingreeky/LOPD

Hugging Face: Qwen3-8B-LOPD Olmo3-7B-LOPD

## ABSTRACT

Enabling agents to learn from experience and internalize it into their policy has become a central problem in self-evolving AI. On-policy self-distillation (OPSD) offers an effective pathway by using a privileged self-teacher to provide dense supervision on the student’s own trajectories; however, existing methods still rely heavily on designer-specified privileged artifacts (e.g., answers, feedback, skills, or trajectories), limiting the end-to-end learnability and scalability required for continual self-improvement. In this work, we introduce Latent On-Policy Self-Distillation (LOPD), which, rather than proposing another hand-crafted OPSD variant with a newly prescribed form of privileged context, makes the teacher’s privileged context itself learnable end-to-end from experience. Technically, LOPD retrieves relevant experiences and composes them into continuous latent tokens that condition a self-teacher, while the student generates trajectories from the task and interaction history and receives dense token-level supervision at every visited prefix. We further introduce a privileged-margin objective to stabilize and regulate the learning of latent context. Empirically, LOPD demonstrates (I) strong performance, outperforming RLVR and representative OPSD methods including OPSD, SDPO, and Skill-SD across both agentic tool use and code generation; and (II) high learning efficiency, surpassing GRPO and Skill-SD with less than 30% of their rollout budget. Ablation studies further provide direct evidence that making privileged context learnable is necessary for realizing these gains. Together, these results position LOPD as a step toward a more scalable and self-directed paradigm for agent evolution.

## 1 INTRODUCTION

How should an agent learn from experience? A natural answer is to expose the model to prior successful trajectories, corrective feedback, or expert demonstrations, and then train it to internalize the behaviors they reveal. This intuition has made on-policy distillation (OPD) a central paradigm for experience learning in large language models: the student first samples its own trajectory, and a teacher then provides dense token-level supervision on the states the student actually visits (Song & Zheng, 2026; Li et al., 2026b). Compared with off-policy imitation, OPD reduces the mismatch between training and inference; compared with reinforcement learning with verifiable rewards (RLVR), it converts sparse outcome feedback into fine-grained distributional signals over the student’s own rollout (Hubotter et al.¨ , 2026; Yang et al., 2026). In this view, OPD is not merely a compression technique for a stronger model, but a mechanism for turning external experience into persistent policy improvement.

On-policy self-distillation (OPSD) has recently emerged as an especially appealing form of this paradigm because it removes the dependence on a separate, stronger teacher. Instead, the teacher and the student are instantiated from the same model under different contexts: the student acts from the task and interaction history, while the teacher additionally receives privileged information such as verified reasoning traces, final answers, environment feedback, or successful prior rollouts (Zhao et al., 2026; Hubotter et al.¨ , 2026; Penaloza et al., 2026). The privileged context makes the teacher distribution more informative than the student’s, enabling dense feedback without querying an external model. This design shifts the central question of OPSD from which teacher is strong enough? to what privileged context should be given to the self-teacher? The answer matters: the context must expose useful experience, make token-level supervision sharper, and yet remain compatible with a student that will not see that context at inference time.

![](images/aa7f4507a2a9373446df25fb43aeef630b6392b4f34fe7995c5539eac71d1d5e.jpg)  
Figure 1: From fixed to learnable privileged context. Existing OPSD methods condition the selfteacher on designer-specified answers, skills, or trajectories. LOPD instead learns latent context from prior experience and distills its dense supervision into the same on-policy student. Insights/tips illustrate compatible sources; LOPD instantiates the substrate using trajectories only.

Existing substrates for privileged experience remain imperfect in a more fundamental sense. Although prior work instantiates privileged context in many forms—verified answers, reasoning traces, environment feedback, contrastive evidence, or action-only trajectories (Zhao et al., 2026; Hubotter¨ et al., 2026; Yu et al., 2026c; Penaloza et al., 2026; Yang et al., 2026)—these contexts are typically hand-designed or rule-extracted transformations of experience. They decide a priori what the teacher should see: an oracle answer, a textual reflection, a successful rollout, a retrieved document, or another discrete artifact. Such substrates can be useful, but they also constrain self-distillation to the information format chosen by the designer, rather than allowing the teacher to learn which aspects of prior experience are actually useful for supervising the student’s current trajectory. This motivates a substrate-level question for OPSD:

## Research Question

Can privileged context itselfbe learned end-to-endfrom experience, so that the self-teacher, rather than a designer-specified rule, determines what experiential knowledge to retain and how to encode itfor dense on-policy supervision?

To address this challenge, we propose Latent On-Policy Self-Distillation (LOPD), a framework that realizes privileged context as a learnable latent substrate. Its training pipeline otherwise follows standard OPSD: the student first rolls out trajectories from the task and interaction history, and the teacher re-evaluates every visited prefix to provide dense token-level distributions, against which the student is optimized through reverse-KL distillation. The key difference is that the teacher is conditioned not on a pre-defined privileged artifact, but on learnable latent context instantiated by a composer that transforms retrieved experiences into compact continuous tokens. Because this context is differentiable, the same training process that improves the student also teaches the composer which aspects of experience to retain and how to organize them into effective teacher supervision. To ensure that this learning makes the teacher more informative rather than merely easier for the student to match, we introduce a privileged-margin constraint that requires the teacher to maintain a verifiable log-probability advantage over the student. After training, only the student is retained: inference requires no experience database, retrieval module, composer, or latent context. In this way, LOPD turns experience from a hand-crafted input artifact into an end-to-end learnable supervision substrate whose benefits are internalized by the student policy.

## Positioning

This work does not aim to introduce yet another elaborate OPSD variant by hand-designing a new form of privileged context. Its central claim is at the level of the experience substrate: if self-evolving AI is to scale, neither the useful content of experience nor its representation should be fixed by designer heuristics. A scalable system should learn end-to-end what evidence to preserve and how to transform it into supervision. We study OPSD as a concrete mechanism for realizing this broader principle.

Our contributions are summarized as follows:

• Context Re-formulation. We reformulate privileged context from pre-defined, hand-designed textual artifacts into an end-to-end learnable latent substrate, allowing the teacher to automatically extract task-relevant supervision signals from prior experience.

• Proposed Solution. We develop LOPD, which composes retrieved experiences into continuous latent context for the self-teacher and jointly optimizes the privileged context and student through on-policy distillation. A privileged-margin constraint keeps the learned context informative and prevents the teacher from collapsing toward the student.

• Experimental Validation. LOPD consistently surpasses RLVR and representative OPSD methods over three model backbones and seven benchmarks, and outperforms GRPO and Skill-SD with less than 30% of their rollout budget. Further ablation studies directly establish the necessity of jointly learning the privileged context.

## 2 RELATED WORK

On-Policy (Self-)Distillation. On-policy distillation trains a student on trajectories sampled from its own policy while querying a teacher for dense token-level supervision on the visited states (Song & Zheng, 2026; Li et al., 2026b). Recent work has applied this recipe broadly: reasoning and efficient post-training (Wu et al., 2026a; Jin et al., 2026; Fu et al., 2026), long-context modeling (Zhang et al., 2026a), context and knowledge distillation (Ye et al., 2026; Lazaridis et al., 2026), GUI grounding (Zhang et al., 2026b), and multimodal domains such as video grounding (Li et al., 2026a), speech alignment (Cao et al., 2026), and visual reasoning (Yuan et al., 2026; Liu et al., 2026). Within this landscape, on-policy self-distillation (OPSD) removes the external teacher by instantiating teacher and student from the same model under different contexts: the student acts under the deployable input, while the teacher receives privileged context (Zhao et al., 2026; Hubotter¨ et al., 2026; Penaloza et al., 2026; Yang et al., 2026; Yu et al., 2026c). This line is best understood by the form of privileged context it supplies: verified answers and reasoning traces (Zhao et al., 2026; Yang et al., 2026), rich environment feedback or self-revision signals (Hubotter et al.¨ , 2026; He et al., 2026; Zhang et al., 2026c), contrastive or peer-rollout evidence (Yu et al., 2026a; Pan et al., 2026), action-only frontier-agent trajectories (Penaloza et al., 2026), and retrieved or evidence-guided context (Ye et al., 2026; Lazaridis et al., 2026). These designs demonstrate that privileged context is the key interface through which OPSD converts experience into supervision, but the context itself is usually pre-defined as a textual or discrete artifact; in contrast, LOPD makes the privileged context a learnable latent substrate jointly optimized with distillation.

Latent Computation. Latent computation uses continuous latent tokens/embeddings or hidden states as the carrier of LLM computation rather than natural language (Yu et al., 2026b; Zhu et al., 2025). In reasoning, latent tokens can expand the model’s internal compute budget (Deng et al., 2026; Hao et al., 2025; Amos et al., 2026), letting the model perform deeper deliberation before producing an answer. In memory, latent states can serve as compact carriers of procedural (Zhang et al., 2025; Yu et al., 2026d), factual (Wang et al., 2024; Feng et al., 2026), and experiential (Hou et al., 2026; Wu et al., 2026b) information, preserving reusable experience without exposing long textual traces in the prompt. In planning, latent computation can represent intermediate plans, subgoals, or world-state summaries that guide downstream actions while remaining flexible and differentiable. Our method is conceptually close to latent memory, but uses it in a different role: the latent state is not an inference-time augmentation, but a learnable privileged context through which an on-policy teacher supervises a student that does not observe this context.

## 3 METHOD

## 3.1 PROBLEM SETUP: PRIVILEGED CONTEXT IN OPSD

We consider a multi-turn agent that receives a task x and interacts with an environment through observations and actions. We call the acting policy the student. At turn t, the student observes $\pmb { s } _ { t } = ( x , \pmb { o } _ { \leq t } , \pmb { a } _ { < t } )$ , where $\mathbf { \delta } _ { \mathbf { o } _ { \leq } t }$ denotes observations so far and $\mathbf { \delta } \mathbf { a } _ { < t }$ denotes previous actions, and samples actions to form a trajectory:

$$
\begin{array} { r } { \pmb { a } _ { t } \sim \pi _ { \theta } ^ { S } ( \cdot \mid \pmb { s } _ { t } ) , \qquad \pmb { \tau } = ( \pmb { s } _ { 1 } , \pmb { a } _ { 1 } , \dots , \pmb { s } _ { \mid \pmb { \tau } \mid } , \pmb { a } _ { \mid \pmb { \tau } \mid } ) \sim \pi _ { \theta } ^ { S } ( \cdot \mid \pmb { x } ) , } \end{array}\tag{1}
$$

where $\mathbf { } \mathbf { } a _ { t } = ( a _ { t , 1 } , \ldots , a _ { t , L _ { t } } )$ is the tokenized action at turn $t , L _ { t }$ is its length, and $| \tau |$ is the number of turns. OPSD constructs a teacher by giving the same model additional privileged context. Abstractly,

![](images/9e14204c880020d0f96946986ca606002b7060cf5df554b76e0f7c1f3c12b423.jpg)  
Figure 2: Overview of LOPD. The student generates on-policy trajectories from the task and interaction history. A fixed-backbone teacher re-evaluates the same prefixes with latent context composed from retrieved experiences, and reverse-KL distillation matches their top-M-plus-tail distributions.

this context is obtained from some experience source E by a fixed transformation:

$$
\begin{array} { r } { \boldsymbol { c } _ { \mathrm { f i x } } = \Phi _ { \mathrm { f i x } } ( \boldsymbol { x } , \mathcal { E } ) , \qquad \pi _ { \boldsymbol { \theta } } ^ { T } ( \cdot \mid \boldsymbol { s } _ { t } , c _ { \mathrm { f i x } } ) = \pi _ { \boldsymbol { \theta } } ( \cdot \mid \left[ \boldsymbol { s } _ { t } ; c _ { \mathrm { f i x } } \right] ) , } \end{array}\tag{2}
$$

where $\mathcal { E }$ may contain answers, traces, feedback, demonstrations, or retrieved trajectories, and $\Phi _ { \mathrm { f i x } }$ is a designer-specified rule that decides what artifact the teacher sees. Given a student trajectory, OPSD matches teacher and student distributions on the same visited prefixes:

$$
\mathcal { L } _ { \mathrm { O P S D } } ( \theta ) = \mathbb { E } _ { \tau \sim \pi _ { \theta } ^ { S } } \left[ \sum _ { t = 1 } ^ { | \tau | } \sum _ { n = 1 } ^ { L _ { t } } D \big ( \mathrm { s g } \big [ \pi _ { \theta } ^ { T } ( \cdot \mid s _ { t } , c _ { \mathrm { f i x } } , a _ { t , < n } ) \big ] \big \| \pi _ { \theta } ^ { S } ( \cdot \mid s _ { t } , a _ { t , < n } ) \big ) \right] ,\tag{3}
$$

where $a _ { t , < n }$ is the action prefix before token n, D is a token-level divergence, and $\mathrm { s g } [ \cdot ]$ denotes stop-gradient. This setup makes the limitation explicit: the supervision quality is bounded by a pre-defined context constructor. We instead parameterize the constructor:

$$
\pmb { c } _ { \phi } = \Phi _ { \phi } ( x , \pmb { \mathcal { E } } ) = \langle \boldsymbol { e } _ { 1 } \rangle \oplus \langle \boldsymbol { e } _ { 2 } \rangle \oplus \dots \oplus \langle \boldsymbol { e } _ { K } \rangle ,\tag{4}
$$

where $\Phi _ { \phi }$ is a learnable composer, $\left. e _ { i } \right.$ is a continuous latent token, K is the number of latent tokens per retrieved experience, and ⊕ denotes context concatenation. The goal of LOPD is to make privileged context itself learnable while preserving the defining asymmetry of OPSD: the teacher additionally observes $\mathbf { \Delta } _ { c _ { \phi } . }$ , whereas the student continues to condition only on $\mathbf { \Delta } _ { \mathbf { \mathcal { S } } _ { t } }$

## 3.2 LEARNABLE LATENT PRIVILEGED CONTEXT

LOPD instantiates $\Phi _ { \phi }$ as a latent-context composer. We deliberately begin with a minimal experience pipeline. Offline, we retain successful rollouts in an experience bank, with each entry storing its task description and a compact action–result trace that omits verbose observations. For a current task x, a dense retriever embeds the query and each stored task–trajectory pair, then returns the top-J entries under cosine similarity; these entries form $\mathcal { E } = \{ m _ { j } \} _ { j = 1 } ^ { J }$ . Full construction and retrieval details are provided in Appendix A.3.

## Raw Experience as a Minimal Substrate

Our experience bank and similarity retriever are intentionally simple. Richer sources such as off-theshelf agent skills, reusable codebooks, or learned retrievers can be incorporated without changing the framework. The point is not to prescribe another experience format, but to show that useful privileged context can be learned directly from minimally processed trajectories rather than hand-crafted rules.

Given x and the retrieved set $\mathcal { E } ,$ the composer first encodes each item into hidden states:

$$
\mathbf { H } _ { j } = \operatorname { E n c } _ { \psi } ( x , m _ { j } ) \in \mathbb { R } ^ { T _ { j } \times d } ,\tag{5}
$$

where $m _ { j }$ is the j-th retrieved trajectory, $\operatorname { E n c } _ { \psi }$ is the encoder that maps the task–experience pair into hidden states, $T _ { j }$ is the encoded sequence length, and d is the hidden dimension. The variable-length states are compressed into a fixed number of latent tokens by a lightweight latent compressor. In our implementation, this compressor is QFormer-style cross-attention with learned queries, although other latent-token generators can be used under the same abstraction:

$$
\mathbf { E } _ { j } = \operatorname { C o m p } _ { \chi } ( \mathbf { H } _ { j } ; \mathbf { Q } _ { \chi } ) = [ \langle e _ { j , 1 } \rangle , \dots , \langle e _ { j , K } \rangle ] \in \mathbb { R } ^ { K \times d } ,\tag{6}
$$

where $\mathrm { C o m p } _ { \chi }$ is the compressor, $\mathbf { Q } _ { \boldsymbol { \chi } } \in \mathbb { R } ^ { K \times d }$ is a learned query bank, and $\mathbf { E } _ { j }$ is the latent representation of experience item $m _ { j }$ . We use $\phi : = ( \psi , \chi )$ to denote all trainable composer parameters, comprising the encoder LoRA parameters ψ and compressor parameters $\chi ;$ the encoder backbone itself remains frozen. Note that we do not treat the particular attention block as a contribution; its role is simply to produce a compact, differentiable context (details in Appendix A.1). The teacher receives the latent tokens as ordinary context positions:

$$
c _ { \phi } ( x , \mathcal { E } ) = \bigoplus _ { j = 1 } ^ { J } ( \langle e _ { j , 1 } \rangle \oplus \cdots \oplus \langle e _ { j , K } \rangle ) , \qquad \pi _ { \theta , \phi } ^ { T } ( \cdot \mid s , c _ { \phi } ) = \pi _ { \bar { \theta } } ( \cdot \mid [ s ; c _ { \phi } ] ) ,\tag{7}
$$

where s is an interaction state defined above and $c _ { \phi }$ is the learned privileged context. Each $\langle e _ { j , k } \rangle$ is implemented as a continuous embedding and behaves like a special token in the teacher’s context window. The composer is cold-started on successful trajectories with the backbone frozen, yielding $\phi _ { 0 }$ . During subsequent joint optimization, both the encoder LoRA parameters $\psi$ and compressor parameters χ remain trainable as part of $\phi ,$ while the teacher backbone $\bar { \theta }$ stays fixed (details in Appendix A.2).

## 3.3 LATENT ON-POLICY SELF-DISTILLATION

Having specified how the teacher is constructed, we now describe the self-distillation process. The student first produces a multi-turn trajectory conditioned only on $\mathbf { \boldsymbol { s } } _ { t }$ . The composer then constructs $c _ { \phi }$ from retrieved experience, and the teacher re-evaluates the same student prefixes with this additional context. We instantiate the teacher as a frozen reference copy $\bar { \theta }$ initialized from the same backbone as the student; the two share architecture and initialization, but not weights after the stu dent begins updating. For every turn t and token position $n ,$ we define:

$$
\begin{array} { r } { \pmb { p } _ { t , n } ^ { S } = \pi _ { \pmb { \theta } } ^ { S } \big ( \cdot \mid s _ { t } , a _ { t , < n } \big ) , \ p _ { t , n } ^ { T } = \pi _ { \pmb { \theta } , \phi } ^ { T } \big ( \cdot \mid s _ { t } , \pmb { c } _ { \phi } ( x , \pmb { \mathcal { E } } ) , a _ { t , < n } \big ) , } \end{array}\tag{8}
$$

where $p _ { t , n } ^ { S }$ and $p _ { t , n } ^ { T }$ are next-token distributions over the vocabulary V, and $\bar { \theta }$ denotes a fixed basemodel checkpoint whose parameters receive no gradient updates; however, the teacher’s forward activations remain in the computation graph, so $p _ { t , n } ^ { T }$ is differentiable with respect to $c _ { \phi }$ and hence to $\phi .$ To keep logit-level distillation efficient, we follow (Zhao et al., 2026) and use the teacher top-M vocabulary entries plus a tail bucket. This preserves the teacher’s high-probability modes while accounting for the remaining probability mass:

$$
\begin{array} { r } { \tilde { p } _ { t , n } ^ { T } ( v ) = \left\{ \begin{array} { l l } { p _ { t , n } ^ { T } ( v ) , } & { v \in \mathcal { V } _ { t , n } ^ { M } , } \\ { 1 - \sum _ { u \in \mathcal { V } _ { t , n } ^ { M } } p _ { t , n } ^ { T } ( u ) , } & { v = \bot , } \end{array} \right. \tilde { p } _ { t , n } ^ { S } ( v ) = \left\{ \begin{array} { l l } { p _ { t , n } ^ { S } ( v ) , } & { v \in \mathcal { V } _ { t , n } ^ { M } , } \\ { 1 - \sum _ { u \in \mathcal { V } _ { t , n } ^ { M } } p _ { t , n } ^ { S } ( u ) , } & { v = \bot , } \end{array} \right. , } \end{array}\tag{9}
$$

where $\nu _ { t , n } ^ { M }$ is the teacher top-M support at prefix (t, n) and ⊥ denotes the aggregated tail event. The distillation objective matches teacher and student on the student’s own multi-turn trajectory:

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } ( \theta , \phi ) = \mathbb { E } _ { x \sim \mathcal { D } } \mathbb { E } _ { \tau \sim \pi _ { \theta } ^ { S } ( \cdot | x ) } \left[ \frac { \sum _ { t = 1 } ^ { | \tau | } \sum _ { n = 1 } ^ { L _ { t } } \omega _ { t , n } D _ { \mathrm { K L } } \left( \tilde { p } _ { t , n } ^ { S } \left\| \tilde { p } _ { t , n } ^ { T } \right) \right. } { \sum _ { t = 1 } ^ { | \tau | } \sum _ { n = 1 } ^ { L _ { t } } \omega _ { t , n } } \right] ,\tag{10}
$$

where D is the task distribution and $\omega _ { t , n } \in \{ 0 , 1 \}$ masks supervised action tokens. We use reverse KL so that the student concentrates on teacher-supported behavior. Because $\bar { \theta }$ denotes a fixed checkpoint while $\phi$ remains trainable, the distillation gradient flows through the frozen network back to the latent input (injection mechanism in Appendix A.4). This allows the composer to learn which experience features produce effective teacher supervision for the current student trajectory.

Algorithm 1: Latent On-Policy Self-Distillation (LOPD).   
Input : Task distribution D; retriever Ret; student $\pi _ { \theta } ^ { S } ;$ fixed teacher $\pi _ { \bar { \theta } } ^ { T } \{$ ; composer $\Phi _ { \phi }$ (init.   
from $\phi _ { 0 } )$ ; reward verifier $V : \tau \mapsto [ 0 , 1 ] ;$ ; margin $m ;$ dual step η<sub>β</sub>   
Output: Trained student policy $\pi _ { \theta } ^ { S }$   
1 $\beta \gets 0 ;$ cache $c _ { \phi _ { 0 } }$ per task   
2 for each training iteration do   
3 Sample tasks $\{ x _ { i } \}$ from D   
4 foreach task $x _ { i }$ do   
$^ { \prime * }$ On-policy student rollout \*/   
5 Sample trajectory $\tau _ { i } \sim \pi _ { \theta } ^ { S } ( \cdot \mid x _ { i } ) ;$ $r _ { i } \gets V ( \tau _ { i } ) ;$ $A _ { i } \gets 2 r _ { i } - 1$   
$^ { \prime * }$ Teacher evaluation with learned privileged context \*/   
6 $\mathcal { E } _ { i }  \mathrm { R e t } ( x _ { i } ) ;$   
7 $\pmb { c } _ { \phi , i } \gets \Phi _ { \phi } ( x _ { i } , \pmb { \mathcal { E } } _ { i } )$   
8 foreach turn t and action prefix $a _ { t , < n }$ in $\tau _ { i }$ do   
9 Compute $\pmb { p } _ { t , n } ^ { S }$ and $\boldsymbol { p _ { t , n } ^ { T } }$ (Equation (8)); form top-M-plus-tail   
10 Accumulate $D _ { \mathrm { K L } } ( \tilde { p } _ { t , n } ^ { S } \Vert \tilde { p } _ { t , n } ^ { T } )$   
11 $\delta _ { t , n } \gets \log { p _ { t , n } ^ { T } ( a _ { t , n } ) } - \mathrm { s g } [ \log { p _ { t , n } ^ { S } ( a _ { t , n } ) } ]$   
12 $\Delta \gets$ weighted mean of $A _ { i } \cdot \delta _ { t , n }$ over masked tokens   
13 Update $( { \bar { \theta } } , \phi )$ by minimizing Equation (13)   
14 $\beta  [ \beta + \eta _ { \beta } ( m - \Delta ) ] .$ +

Privileged-Margin Constraint. The distillation loss alone does not ensure that the teacher provides more effective supervision: an unconstrained composer can instead minimize Equation (10) by moving $p ^ { T }$ toward $p ^ { S }$ , yielding uninformative latent context without improving the student. We therefore prevent this collapse in two ways. ❶ First, the cold start in Section 3.2 initializes the composer to transform experience into informative latent context. ❷Second, we introduce outcome reward into the supervision signal, and more concretely, impose a privileged-margin constraint that ties the teacher’s token-level advantage to the verified outcome of the complete trajectory. For each supervised token, we define the per-token privilege:

$$
\delta _ { t , n } ( \phi ) = \log \pi _ { \bar { \theta } , \phi } ^ { T } ( a _ { t , n } \mid s _ { t } , c _ { \phi } , a _ { t , < n } ) - \mathrm { s g } \big [ \log \pi _ { \theta } ^ { S } ( a _ { t , n } \mid s _ { t } , a _ { t , < n } ) \big ] ,\tag{11}
$$

where $\boldsymbol { a } _ { t , n }$ is the student’s sampled token and sg blocks gradients to $\theta$ through this path. Both logprobabilities are already computed during the teacher and student forward passes; $\delta _ { t , n }$ requires only a gather, not an additional forward. We then use the trajectory-level verification signal $A ( \tau )$ = $2 \bar { r ( \tau ) } - 1 \in [ - 1 , 1 ]$ , where $r ( \tau )$ is the outcome reward, to favor teacher advantages on successful trajectories and suppress them on unsuccessful ones:

$$
\Delta ( \phi ) = \mathbb { E } _ { \pmb { \tau } } \left[ \frac { \sum _ { t , n } \omega _ { t , n } A ( \pmb { \tau } ) \delta _ { t , n } ( \phi ) } { \sum _ { t , n } \omega _ { t , n } } \right] .\tag{12}
$$

The full LOPD objective constrains $\phi$ to maintain a minimum privilege level m $> 0 \colon$

$$
\operatorname* { m i n } _ { \theta , \phi } \ \operatorname* { m a x } _ { \beta \geq 0 } \ \mathcal { L } _ { \mathrm { d i s t i l l } } ( \theta , \phi ) \ + \ \beta ( m - \Delta ( \phi ) ) + \ \lambda \left\| \pmb { c } _ { \phi } - \mathrm { s g } [ \pmb { c } _ { \phi _ { 0 } } ] \right\| _ { 2 } ^ { 2 } ,\tag{13}
$$

where $\beta$ is a dual variable updated by $\beta  [ \beta + \eta _ { \beta } ( m - \Delta ( \phi ) ) ] _ { + }$ , and the anchor term penalizes drift from the initialization $\phi _ { 0 }$ in latent space, requiring no additional forward pass. If the composer degenerates to uninformative context, $\tilde { \pi ^ { T } }  \pi ^ { \hat { S } } , \delta _ { t , n }  0 , \Delta  0 < m$ , and the dual penalty activates—structurally excluding the trivial solution. Within the feasible region, outcome-weighted privilege steers the composer toward evidence that supports successful behavior, while the distillation gradient determines how that evidence is selected and encoded.

Algorithm Summary. Algorithm 1 summarizes the overall workflow of LOPD. The loop collects on-policy trajectories from the student, constructs learnable latent privileged context from retrieved experience, evaluates the same visited prefixes with the teacher, and jointly updates the student and composer while adjusting the dual variable to maintain the privilege margin. At inference, only the student policy $\pi _ { \theta } ^ { S }$ is deployed and used.

Table 1: Tool-use results across EnvScaler, BFCL-v3, and ACEBench. All results are reported on a 0–100 scale; higher is better. Green and light-green cells mark the best and second-best results.
<table><tr><td rowspan="2">LLM</td><td rowspan="2">Method EnvScaler</td><td rowspan="2"></td><td colspan="5">BFCL-v3</td><td colspan="3">ACEBench</td></tr><tr><td>Base</td><td>Miss Func</td><td>Miss Param</td><td>Long Ctx</td><td>Avg</td><td>M-Step</td><td>M-Turn</td><td>Avg</td></tr><tr><td rowspan="7">QWEN3-4B</td><td>Vanilla</td><td>48.6</td><td>28.50</td><td>20.50</td><td>20.50</td><td>22.00</td><td>22.88</td><td>48.3</td><td>52.2</td><td>50.6</td></tr><tr><td>GRPO</td><td>61.8</td><td>33.00</td><td>20.50</td><td>24.50</td><td>23.00</td><td>25.25</td><td>58.3</td><td>54.4</td><td>56.0</td></tr><tr><td>Skill-SD</td><td>59.1</td><td>33.00</td><td>19.00</td><td>24.00</td><td>22.50</td><td>24.63</td><td>56.6</td><td>55.6</td><td>56.0</td></tr><tr><td>OPSD</td><td>51.2</td><td>31.50</td><td>19.50</td><td>20.50</td><td>29.00</td><td>25.13</td><td>41.6</td><td>53.3</td><td>48.6</td></tr><tr><td>SDPO</td><td>50.6</td><td>21.50</td><td>14.00</td><td>13.50</td><td>14.00</td><td>15.75</td><td>45.0</td><td>33.3</td><td>38.0</td></tr><tr><td>SDFT</td><td>50.3</td><td>31.50</td><td>17.00</td><td>17.00</td><td>21.50</td><td>21.75</td><td>53.3</td><td>47.8</td><td>50.0</td></tr><tr><td>LOPD</td><td>63.7</td><td>32.50</td><td>26.50</td><td>24.50</td><td>26.00</td><td>27.38</td><td>56.6</td><td>63.3</td><td>60.6</td></tr><tr><td rowspan="7">QWEN3-8B</td><td>Vanilla</td><td>49.2</td><td>34.00</td><td>28.50</td><td>25.50</td><td>25.50</td><td>28.38</td><td>61.6</td><td>50.0</td><td>54.6</td></tr><tr><td>GRPO</td><td>57.3</td><td>36.00</td><td>28.00</td><td>27.00</td><td>25.00</td><td>29.00</td><td>65.0</td><td>53.3</td><td>58.0</td></tr><tr><td>Skill-SD</td><td>60.2</td><td>33.00</td><td>27.00</td><td>25.00</td><td>24.50</td><td>27.38</td><td>63.3</td><td>51.1</td><td>56.0</td></tr><tr><td>OPSD</td><td>52.0</td><td>30.50</td><td>25.00</td><td>23.50</td><td>24.00</td><td>25.75</td><td>60.0</td><td>47.8</td><td>52.7</td></tr><tr><td>SDPO</td><td>55.3</td><td>29.00</td><td>24.50</td><td>23.00</td><td>23.50</td><td>25.00</td><td>56.7</td><td>48.9</td><td>52.0</td></tr><tr><td>SDFT</td><td>56.2</td><td>31.50</td><td>26.00</td><td>24.50</td><td>25.50</td><td>26.88</td><td>65.0</td><td>47.8</td><td>54.7</td></tr><tr><td>LOPD</td><td>66.4</td><td>37.00</td><td>29.00</td><td>27.50</td><td>26.00</td><td>29.88</td><td>73.3</td><td>55.6</td><td>62.7</td></tr></table>

## 4 EXPERIMENTS

## 4.1 EXPERIMENT SETUP

Training. We evaluate LOPD under two post-training settings: ❶ agentic tool-use and ❷ coding. For tool-use training, we use the EnvScaler-derived tool-interactive corpus (Song et al., 2026), comprising 2,349 tasks. For coding, we use the TACO subset of DeepCoder (TogetherAI, 2025), comprising 7K verified Python problems (see Appendix B.2 for training details). The used model backbones include QWEN3-4B, QWEN3-8B (Yang et al., 2025), OLMO3-7B (Olmo et al., 2026).

Evaluation. Our evaluation mirrors the two training regimes while keeping all test sets disjoint from training data. For tool-use, we report the EnvScaler task success metric on a held-out test split of 200 tasks disjoint from the training pool, BFCL-v3 scores across base, missing-function, missingparameter, and long-context subsets (Patil et al., 2025), and ACEBench multi-step and multi-turn scores (Chen et al., 2025). For coding, we report pass@1 on LiveCodeBench v5/v6 (Jain et al., 2024), HumanEval+ and MBPP+ (Liu et al., 2023) (evaluation protocols in Appendix B.3).

Baselines. We compare against the baselines reported in Tables 1 and 2. Vanilla denotes the unadapted backbone. GRPO is the outcome-reward RL baseline (Shao et al., 2024). SDFT (Shenfeld et al., 2026) is a demonstration-conditioned distillation baseline. The OPSD-style baselines include OPSD (Zhao et al., 2026), SDPO (Hubotter et al.¨ , 2026), and Skill-SD (Wang et al., 2026), whose privileged contexts respectively instantiate answer/trace, feedback-conditioned self-teacher, and skill-conditioned supervision. Detailed baseline configurations are provided in Appendix B.1.

Configurations. All trainable methods share the same backbone, training split and evaluation protocol within each setting. For LOPD, the teacher receives latent context composed from J=3 retrieved experiences, each compressed into K=32 latent tokens (96 total); the distillation loss uses reverse KL with SDPO-style top-M logits plus a tail bucket, with M=20 by default. The privileged-margin threshold is m=0.05; the verification signal is $A ( \tau ) = 2 r ( \tau ) - 1$ where r is the environment reward. Tool-use rollouts are capped at 30 environment steps. Coding distillation uses a 16,384-token response budget. We keep decoding temperature/top-p fixed across methods within each benchmark, each baseline retains its original KL direction and loss formulation (Appendix B.1), and within the on-policy training loop the methods differ only in privileged context construction and distillation loss. Prompt templates are provided in Appendix A.5.

## 4.2 MAIN RESULTS

We evaluate whether latent privileged context improves self-distillation learning across tool use and code generation domains in Tables 1 and 2.

Performance. LOPD obtains the best aggregate result in all ten backbone–benchmark comparisons. On tool use with QWEN3-4B, it improves over the strongest competing method from 61.8 to 63.7 on EnvScaler, from 25.25 to 27.38 on BFCL-v3, and from 56.0 to 60.6 on ACEBench. The advantage becomes larger with QWEN3-8B: LOPD reaches 66.4/29.88/62.7 on EnvScaler, BFCLv3, and ACEBench, compared with the strongest baseline results of 60.2/29.00/58.0, respectively. The same trend extends beyond interactive agents. With QWEN3-4B, LOPD improves the Live-CodeBench and EvalPlus aggregates to 48.78 and 81.36; with OLMO3-7B, it reaches 50.98 and 78.41, outperforming the strongest alternatives by 2.69 and 0.55 points. These results indicate that LOPD’s improvement transfers across task formats and model families.

Table 2: Code results across LiveCodeBench and EvalPlus. Each cell reports pass@1 (%); higher is better. Green and light-green cells mark the best and second-best within each backbone group.
<table><tr><td rowspan="2">LLM</td><td rowspan="2">Method</td><td colspan="3">LiveCodeBench</td><td colspan="3">EvalPlus</td></tr><tr><td>v5</td><td>v6</td><td>Avg</td><td>HumanEval+</td><td>MBPP+</td><td>Avg</td></tr><tr><td rowspan="7">QWEN3-4B</td><td>Vanilla</td><td>47.67</td><td>41.22</td><td>45.61</td><td>85.37</td><td>75.93</td><td>78.79</td></tr><tr><td>GRPO</td><td>50.18</td><td>44.27</td><td>48.29</td><td>86.59</td><td>76.19</td><td>79.34</td></tr><tr><td>Skill-SD</td><td>49.82</td><td>41.98</td><td>47.32</td><td>85.98</td><td>76.98</td><td>79.70</td></tr><tr><td>OPSD</td><td>42.65</td><td>35.11</td><td>40.24</td><td>85.36</td><td>76.72</td><td>79.33</td></tr><tr><td>SDPO</td><td>40.86</td><td>34.35</td><td>38.78</td><td>84.76</td><td>74.87</td><td>77.86</td></tr><tr><td>SDFT</td><td>48.75</td><td>43.51</td><td>47.07</td><td>87.20</td><td>76.98</td><td>80.07</td></tr><tr><td>LOPD</td><td>50.54</td><td>45.04</td><td>48.78</td><td>87.80</td><td>78.57</td><td>81.36</td></tr><tr><td rowspan="7">OLM03-7B</td><td>Vanilla</td><td>47.31</td><td>44.27</td><td>46.34</td><td>86.59</td><td>70.37</td><td>75.28</td></tr><tr><td>GRPO</td><td>49.46</td><td>45.80</td><td>48.29</td><td>89.02</td><td>72.75</td><td>77.67</td></tr><tr><td>Skill-SD</td><td>49.10</td><td>45.03</td><td>47.80</td><td>87.20</td><td>72.22</td><td>76.75</td></tr><tr><td>OPSD</td><td>44.09</td><td>45.04</td><td>44.39</td><td>87.80</td><td>71.96</td><td>76.75</td></tr><tr><td>SDPO</td><td>49.82</td><td>43.51</td><td>47.80</td><td>88.41</td><td>72.75</td><td>77.49</td></tr><tr><td>SDFT</td><td>47.67</td><td>44.27</td><td>46.58</td><td>89.02</td><td>73.02</td><td>77.86</td></tr><tr><td>LOPD</td><td>53.41</td><td>45.80</td><td>50.98</td><td>90.24</td><td>73.28</td><td>78.41</td></tr></table>

No Hand-Crafted Context Is Universally Optimal. A careful reader may notice that adding privileged information does not always improve upon the vanilla model in Tables 1 and 2. For example, with QWEN3-4B, SDPO falls from 22.88 to 15.75 on BFCL-v3, from 50.6 to 38.0 on ACEBench, and from 45.61 to 38.78 on LiveCodeBench; OPSD likewise reduces the LiveCodeBench aggregate to 40.24. More revealingly, the utility of the same context can reverse across settings: OPSD raises the BFCL-v3 long-context score from 22.00 to 29.00, yet remains below vanilla on the overall BFCL-v3 score with QWEN3-8B (25.75 vs. 28.38) and on LiveCodeBench with both backbones (40.24 vs. 45.61 and 44.39 vs. 46.34). This behavior is consistent with how these contexts are constructed. SDPO can condition only on successful siblings produced by the current rollout group, making its privileged signal dependent on the current policy’s success coverage; OPSD instead injects a fixed oracle trace, which can be highly informative when its procedure matches the current task but need not align with the particular prefix or valid solution path visited by the student. A context format that helps in one regime can therefore become sparse, mismatched, or overly prescriptive in another. The issue is not simply whether privileged information is available, but whether its representation remains appropriate across tasks and learning states.

Learnable Context Matters. LOPD avoids committing the teacher to a fixed answer, demonstration, sibling rollout, or discrete skill. Instead, it learns a compact continuous representation directly from retrieved trajectories, allowing the privileged signal to adapt to the task and the student’s visited prefixes. Unlike the reversals above, LOPD remains above the vanilla model in all ten aggregate backbone–benchmark settings, with gains ranging from 1.50 points on BFCL-v3 with QWEN3-8B to 17.2 points on EnvScaler with the same backbone. It also consistently improves over prescribedcontext baselines: with QWEN3-8B, LOPD exceeds OPSD, SDFT, and Skill-SD by 14.4/10.2/6.2 points on EnvScaler and 10.0/8.0/6.7 points on ACEBench, respectively. Together, the results favor optimizing the representation of experience for supervision over searching for an increasingly elaborate hand-crafted artifact.

## Takeaway

Experience should not be treated as a designer-specified artifact, but as a learnable substrate for supervision. LOPD enables the agent to discover directly from trajectories what evidence to preserve and how to encode it, making experience learning end-to-end.

<table><tr><td>Metric</td><td>Vanilla</td><td>Base + Composer</td><td>LOPD</td></tr><tr><td>Reward</td><td>0.486</td><td>0.631</td><td>0.637</td></tr><tr><td>Interaction steps</td><td>11.12</td><td>16.31</td><td>17.04</td></tr><tr><td>Tool calls / step</td><td>3.50</td><td>1.21</td><td>1.11</td></tr><tr><td>First-step length</td><td>9,937</td><td>6,695</td><td>6,210</td></tr><tr><td>Reward / tool call</td><td>0.038</td><td>0.053</td><td>0.050</td></tr><tr><td>Repeated tool calls</td><td>8.89</td><td>4.49</td><td>5.25</td></tr></table>

Table 3: Fine-grained behavior on the EnvScaler test set. Base + Composer denotes the unadapted backbone conditioned on composed latent context; the student receives no privileged context.  
![](images/38ebd5fe6098c7d76d6e3719e54c7f9e2a94f9ca1204e28d45f5f86937929dfb.jpg)  
Figure 4: EnvScaler training dynamics. Mean reward over 1,600 rollouts.

## 4.3 FRAMEWORK ANALYSIS

Effect of Joint Optimization. Does joint optimization actually produce a better privileged context, or merely introduce additional trainable parameters? This is the central ablation behind our claim. Figure 3 compares a frozen composer with jointly optimized variants and evaluates each resulting student without retrieved experience or latent context, so the difference reflects what context learning transfers into the policy. Training with a frozen composer $( \phi _ { 0 } )$ achieves 0.573. Without the margin constraint $( m { = } 0 )$ , the student drops to 0.551, indicating that unconstrained distillation gradients degrade the latent context. Weak margins $( m { \le } 0 . 0 1 )$ do not prevent this decline. With $m { \geq } 0 . 0 2 .$ , the student surpasses the frozencomposer baseline, reaching 0.637 at $\mathrm { { \it m } = 0 . 0 5 }$ and 0.626 at $\scriptstyle { m = 0 . 1 0 }$ . The improvement suggests that the distillation process exposes the composer to a signal unavailable during initialization: what evidence the current student’s trajectory distribution requires from the privileged context. The margin constraint is necessary to realize this benefit—without it, collapsing the teacher toward the student is a lowerresistance path than learning a more informative representation.

![](images/4b0244872cbd24e7f3b586489e852c87a07935d67096333c0e97090e4e993155.jpg)  
Figure 3: Effect of joint optimization.<sub>EnvScaler</sub> <sub>Mean</sub> <sub>Reward</sub> EnvScaler performance across margins.

Training Dynamics and Sample Efficiency. Beyond final performance, a practical question is whether LOPD reaches strong performance with fewer on-policy generations. To test this, Figure 4 tracks the strongest baselines over the same 1,600-generation budget. LOPD exceeds 0.61 mean reward after 320 generations and reaches 0.637 by generation 576. Its reward then remains in a narrow 0.63–0.64 range through generation 1,600, showing that the early gain is sustained rather than caused by a shorter training horizon. GRPO and Skill-SD improve more gradually and finish at 0.611 and 0.588, respectively. The persistent early separation suggests that the latent teacher extracts a denser learning signal from each visited trajectory.

Sensitivity Analysis. How much latent capacity and retrieved experience does the learnable substrate actually need? We vary the number of latent tokens produced by the composer and the number of experiences retrieved during training, then evaluate the resulting student alone. Figure 5(a) shows a capacity threshold in the latent bottleneck. EnvScaler reward remains near 0.56 with 8 or 16 tokens per experience, rises sharply to 0.637 with 32, and then fluctuates without a consistent gain at 64 and 128. We therefore use $K { = } 3 2$ tokens per experience as the smallest setting that escapes the low-capacity regime. Retrieval sensitivity is shown in Figure 5(b–e), where $n _ { \mathrm { r e t } }$ varies from 1 to 10. EnvScaler reward improves from 0.605 with one retrieval to 0.637 with three, but additional retrievals yield no monotonic benefit. At the default $n _ { \mathrm { r e t } } { = } 3 .$ , ACEBench reaches 56.6 on M-Step, 63.3 on M-Turn, and 60.6 overall, matching the main result in Table 1. Larger retrieval counts can improve one ACEBench regime without producing the same trajectory in the other, and the aggregate remains on a broad plateau rather than varying monotonically. We therefore retain $n _ { \mathrm { r e t } } { = } 3$ as the earliest setting that attains the strongest EnvScaler reward while already reaching competitive ACEBench performance.

Behavioral Internalization. Finally, higher reward alone does not reveal what the student has internalized. We therefore ask whether LOPD changes how the student interacts with the environment. Table 3 shows that the distilled student inherits the interaction pattern induced by latent context. Relative to the vanilla model, LOPD uses more environment steps (17.04 vs. 11.12) but makes far fewer tool calls per step (1.11 vs. 3.50), indicating a shift from issuing many speculative calls at once to executing a more sequential plan. Its first-step response is 37.5% shorter, repeated calls fall from 8.89 to 5.25, and reward per tool call rises from 0.038 to 0.050. The latent-context-conditioned base model and the final student exhibit the same qualitative pattern, providing behavioral evidence that LOPD internalizes the teacher’s procedural guidance rather than merely fitting the aggregate reward. Per-experiment configurations are detailed in Appendix B.4.

#Retrieved  
(a) #Latent Tokens  
(b)  
![](images/23708ad0c208f00e080eb0304c977d5447ce2ba4ab2bc2ef4f5748ddc226420e.jpg)  
Latent Tokens

EnvScaler  
![](images/1181235f5fc9d611be0d90ec5b2676ecab418c8a2b7b2f5abcc147f2f8405df1.jpg)

(c)  
ACE Avg  
![](images/3edb1669d8ade8877b2cb622ae78aeddc55366cc7c3bcf9e5721175cafd86e38.jpg)  
#Retrieved

(d)  
ACE M-Step  
![](images/0d6da5788565592bc8fc83f9cf2e4e7a319d1d0275291b101b2f598308c9e083.jpg)  
(e)  
#Retrieved

ACE M-Turn  
![](images/136ffdaf24c0c83a652b7bca4b5ef345e3e81437dc00266bca665d1565876f55.jpg)  
Figure 5: Sensitivity analysis. Performance of the resulting student without privileged context. (a) Latent-token capacity. (b) EnvScaler reward and (c–e) ACEBench aggregate, multi-step, and multiturn scores as the training-time retrieval count varies. Dashed lines mark the default setting.

## Mechanistic Takeaway

Outcome reward aligns the teacher’s aggregate advantage with trajectories that achieve better outcomes, while on-policy distillation converts the resulting context-conditioned guidance into dense supervision at states the student actually visits. The guidance is internalized when it persists after privileged context is removed and reappears as a stable change in the student’s behavior.

## 4.4 CASE STUDY

Having established that the latent context improves learning, a natural question is what these continuous tokens actually encode. To obtain a qualitative view, we apply the frozen language-model head to each of the 32 latent tokens and inspect ten tool-use and ten coding examples, with and without task conditioning. Figure 6 shows two representative cases. In the agentic example, the current task and retrieved trajectory share the same add–update–change–withdraw operation schema despite using different entities. In the coding example, the retrieved Josephus recurrence matches the circular-elimination structure of the target problem. Nevertheless, both projections remain fragmented mixtures of multilingual and code-like tokens, and task conditioning changes the surface projection without yielding a readable procedure or copying the retrieved solution. This is consistent with a distributed latent representation, although direct decodability alone does not establish which information the teacher functionally uses.

## 5 CONCLUSION

The question behind this work is not which new artifact should be appended to an OPSD teacher, but whether the teacher’s privileged context can itself be learned from experience. LOPD answers this question by transforming retrieved raw trajectories into differentiable latent privileged context, using the resulting self-teacher to supervise the student’s own visited prefixes, and constraining the learned context to preserve a verifiable teacher advantage. Across agentic tool use and code generation, this formulation achieves the best aggregate result in all ten backbone–benchmark comparisons and surpasses GRPO and Skill-SD with less than 30% of their rollout budget. The analyses sharpen this interpretation: retrieval alone is insufficient, unconstrained joint optimization can collapse the teacher toward the student, and the privileged margin turns context learning into productive supervision. The induced behavioral shift remains in the trained student, which acts without privileged context. More broadly, scalable self-evolution should not depend on a succession of increasingly elaborate, human-authored experience formats. Raw trajectories provide a minimal substrate, and richer repositories or retrievers may broaden what is available, but learning should decide what becomes useful guidance. In this sense, LOPD is less another privileged-context recipe than evidence for a different design principle: experience representations should be optimized end-to-end for the policies they are meant to improve.

![](images/8110c3778e21b7fa18215ba3e7dbe8bfcd1ec12d4bca6286605c099c1fb065ef.jpg)  
Figure 6: Representative latent decoding examples. Each row pairs a current task with a structurally related rank-2 retrieval and excerpts from LM-head projections of 32 latent tokens, with and without task conditioning. The projections remain fragmented and do not reproduce the retrieved procedure verbatim.

## REFERENCES

Ido Amos, Avi Caciularu, Mor Geva, Amir Globerson, Jonathan Herzig, Lior Shani, and Idan Szpektor. Latent reasoning with supervised thinking states, 2026. URL https://arxiv.org/ab s/2602.08332.

Di Cao, Dongjie Fu, Hai Yu, Siqi Zheng, Xu Tan, and Tao Jin. X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs. arXiv preprint arXiv:2603.24596, 2026. URL https://arxiv.org/abs/2603.24596.

Chen Chen, Xinlong Hao, Weiwen Liu, Xu Huang, Xingshan Zeng, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Yuefeng Huang, Wulong Liu, Xinzhi Wang, Defu Lian, Baoqun Yin, Yasheng Wang, and Wu Liu. Acebench: Who wins the match point in tool usage?, 2025. URL https: //arxiv.org/abs/2501.12851.

Jingcheng Deng, Liang Pang, Zihao Wei, Shicheng Xu, Zenghao Duan, Kun Xu, Yang Song, Huawei Shen, and Xueqi Cheng. Llm latent reasoning as chain of superposition, 2026. URL https: //arxiv.org/abs/2510.15522.

Tao Feng, Chongrui Ye, Tianyang Luo, Jingjun Xu, Xueqiang Xu, Haozhen Zhang, Ge Liu, and Jiaxuan You. Elasticmem: Latent memory as a learnable resource for llm agents, 2026. URL https://arxiv.org/abs/2605.30690.

Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and Dongbin Zhao. Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes. arXiv preprint arXiv:2603.25562, 2026. URL https://arxiv.org/abs/2603.25562.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space, 2025. URL https: //arxiv.org/abs/2412.06769.

Yinghui He, Simran Kaur, Adithya Bhaskar, Yongjin Yang, Jiarui Liu, Narutatsu Ri, Liam Fowl, Abhishek Panigrahi, Danqi Chen, and Sanjeev Arora. Self-Distillation Zero: Self-Revision Turns Binary Rewards into Dense Supervision. arXiv preprint arXiv:2604.12002, 2026. URL https: //arxiv.org/abs/2604.12002.

Yubo Hou, Zhisheng Chen, Tao Wan, and Zengchang Qin. Flashmem: Distilling intrinsic latent memory via computation reuse, 2026. URL https://arxiv.org/abs/2601.05505.

Jonas Hubotter, Frederike L¨ ubeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta,¨ Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas Krause. Reinforcement learning via self-distillation, 2026. URL https://arxiv.org/abs/2601.2 0802.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

Woogyeol Jin, Taywon Min, Yongjin Yang, Swanand Ravindra Kadhe, Yi Zhou, Dennis Wei, Nathalie Baracaldo, and Kimin Lee. Entropy-Aware On-Policy Distillation of Language Models. arXiv preprint arXiv:2603.07079, 2026. URL https://arxiv.org/abs/2603.07079.

Aristotelis Lazaridis, Dylan Bates, Aman Sharma, Brian King, Vincent Lu, and Jack FitzGerald. Edge-opd: Internalizing privileged context with evidence guided on-policy distillation, 2026. URL https://arxiv.org/abs/2605.23493.

Jiaze Li, Hao Yin, Haoran Xu, Boshen Xu, Wenhui Tan, Zewen He, Jianzhong Ju, Zhenbo Luo, and Jian Luan. Video-OPD: Efficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation. arXiv preprint arXiv:2602.02994, 2026a. URL https://arxiv.org/abs/2602.02994.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan ang Gao, Wenkai Yang, Zhiyuan Liu, and Ning Ding. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe, 2026b. URL https://arxiv.or g/abs/2604.13016.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation, 2023. URL https://arxiv.org/abs/2305.01210.

Ruiqi Liu, Xiaolei Lv, Gengsheng Li, Ximo Zhu, Zhiheng Wang, Zhengbo Zhang, Junkai Chen, Zhiheng Li, Bo Li, Jun Gao, and Shu Wu. Visual-Advantage On-Policy Distillation for Vision-Language Models. arXiv preprint arXiv:2605.21924, 2026. URL https://arxiv.org/ab s/2605.21924.

Team Olmo, :, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, Pete Walsh, Pradeep Dasigi, Robert Berry, Saumya Malik, Saurabh Shah, Scott Geng, Shane Arora, Shashank Gupta, Taira Anderson, Teng Xiao, Tyler Murray, Tyler Romero, Victoria Graf, Akari Asai, Akshita Bhagia, Alexander Wettig, Alisa Liu, Aman Rangapur, Chloe Anastasiades, Costa Huang, Dustin Schwenk, Harsh Trivedi, Ian Magnusson, Jaron Lochner, Jiacheng Liu, Lester James V. Miranda, Maarten Sap, Malia Morgan, Michael Schmitz, Michal Guerquin, Michael Wilson, Regan Huff, Ronan Le Bras, Rui Xin, Rulin Shao, Sam Skjonsberg, Shannon Zejiang Shen, Shuyue Stella Li, Tucker Wilde, Valentina Pyatkin, Will Merrill, Yapei Chang, Yuling Gu, Zhiyuan Zeng, Ashish Sabharwal, Luke Zettlemoyer, Pang Wei

Koh, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. Olmo 3, 2026. URL https: //arxiv.org/abs/2512.13961.

Leyi Pan, Shuchang Tao, Yunpeng Zhai, Lingzhe Zhang, Zhaoyang Liu, Bolin Ding, Aiwei Liu, and Lijie Wen. RLCSD: Reinforcement Learning with Contrastive On-Policy Self-Distillation. arXiv preprint arXiv:2606.11709, 2026. URL https://arxiv.org/abs/2606.11709.

Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=2GmDdhBdDk.

Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, and Massimo Caccia. Privileged information distillation for language models, 2026. URL https: //arxiv.org/abs/2602.04942.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Idan Shenfeld, Mehul Damani, Jonas Hubotter, and Pulkit Agrawal. Self-distillation enables con-¨ tinual learning, 2026. URL https://arxiv.org/abs/2601.19897.

Mingyang Song and Mao Zheng. A survey of on-policy distillation for large language models, 2026. URL https://arxiv.org/abs/2604.00626.

Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. Envscaler: Scaling tool-interactive environments for llm agent via programmatic synthesis, 2026. URL https://arxiv.org/abs/2601.05808.

TogetherAI. DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level — together.ai. https: //www.together.ai/blog/deepcoder, 2025.

Hao Wang, Guozhi Wang, Han Xiao, Yufeng Zhou, Yue Pan, Jichao Wang, Ke Xu, Yafei Wen, Xiaohu Ruan, Xiaoxin Chen, and Honggang Qi. Skill-sd: Skill-conditioned self-distillation for multi-turn llm agents, 2026. URL https://arxiv.org/abs/2604.10674.

Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li, Xian Li, Bing Yin, Jingbo Shang, and Julian McAuley. Memoryllm: Towards self-updatable large language models, 2024. URL https://arxiv.org/abs/2402.04624.

Yecheng Wu, Song Han, and Han Cai. Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation. arXiv preprint arXiv:2604.13010, 2026a. URL ht tps://arxiv.org/abs/2604.13010.

Zijun Wu, Yongchang Hao, and Lili Mou. Tokmem: One-token procedural memory for large language models, 2026b. URL https://arxiv.org/abs/2510.00444.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. Self-distilled rlvr, 2026. URL https://arxiv.org/ abs/2604.03128.

Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. On-policy context distillation for language models, 2026. URL https://arxiv.org/abs/2602.12275.

Weichen Yu, Xiaomin Li, Yizhou Zhao, Xiaoze Liu, Ruowang Zhang, Haixin Wang, Yinyi Luo, Chen Henry Wu, Gaurav Mittal, Matt Fredrikson, and Yu Hu. Multi-Rollout On-Policy Dis tillation via Peer Successes and Failures. arXiv preprint arXiv:2605.12652, 2026a. URL https://arxiv.org/abs/2605.12652.

Xinlei Yu, Zhangquan Chen, Yongbo He, Tianyu Fu, Guanting Dong, Cheng Yang, Chengming Xu, Yue Ma, Xiaobin Hu, Zhe Cao, Jie Xu, Guibin Zhang, Jiale Tao, Jiayi Zhang, Siyuan Ma, Kaituo Feng, Haojie Huang, Youxing Li, Ronghao Chen, Huacan Wang, Chenglin Wu, Zikun Su, Xiaogang Xu, Kelu Yao, Kun Wang, Chen Gao, Yue Liao, Ruqi Huang, Tao Jin, Zhucun Xue, Cheng Tan, Jiangning Zhang, Wenqi Ren, Yanwei Fu, Yong Liu, Yu Wang, Xiangyu Yue, Yu-Gang Jiang, and Shuicheng Yan. The latent space: Foundation, evolution, mechanism, ability, and outlook, 2026b. URL https://arxiv.org/abs/2604.02029.

Xinlei Yu, Gen Li, Qingyi Si, Guibin Zhang, Yuqi Xu, Congcong Wang, Shuai Dong, Kaiwen Tuo, Xiangyu Zeng, Kaituo Feng, et al. Dopd: Dual on-policy distillation. arXiv preprint arXiv:2606.30626, 2026c.

Xinlei Yu, Chengming Xu, Guibin Zhang, Zhangquan Chen, Yudong Zhang, Yongbo He, Peng-Tao Jiang, Jiangning Zhang, Xiaobin Hu, and Shuicheng Yan. Vismem: Latent vision memory unlocks potential of vision-language models, 2026d. URL https://arxiv.org/abs/25 11.11007.

Qianhao Yuan, Jie Lou, Xing Yu, Hongyu Lin, Le Sun, Xianpei Han, and Yaojie Lu. Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation. arXiv preprint arXiv:2605.18740, 2026. URL https://arxiv.org/abs/2605.18740.

Guibin Zhang, Muxin Fu, and Shuicheng Yan. Memgen: Weaving generative latent memory for self-evolving agents, 2025. URL https://arxiv.org/abs/2509.24704.

Xinsen Zhang, Zhenkai Ding, Tianjun Pan, Run Yang, Chun Kang, Xue Xiong, and Jingnan Gu. OPSDL: On-Policy Self-Distillation for Long-Context Language Models. arXiv preprint arXiv:2604.17535, 2026a. URL https://arxiv.org/abs/2604.17535.

Yan Zhang, Daiqing Wu, Huawen Shen, Yu Zhou, and Can Ma. Learn where to Click from Yourself: On-Policy Self-Distillation for GUI Grounding. arXiv preprint arXiv:2605.00642, 2026b. URL https://arxiv.org/abs/2605.00642.

Yuwei Zhang, Sha Li, Changlong Yu, Qin Lu, Shuowei Jin, Chengyu Dong, Haoran Liu, Ilgee Hong, Xintong Li, Zhenyu Shi, Bing Yin, and Jingbo Shang. Learning with Rare Success but Rich Feedback via Reflection-Enhanced Self-Distillation. arXiv preprint arXiv:2605.12741, 2026c. URL https://arxiv.org/abs/2605.12741.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models, 2026. URL https: //arxiv.org/abs/2601.18734.

Rui-Jie Zhu, Tianhao Peng, Tianhao Cheng, Xingwei Qu, Jinfa Huang, Dawei Zhu, Hao Wang, Kaiwen Xue, Xuanliang Zhang, Yong Shan, Tianle Cai, Taylor Kergan, Assel Kembay, Andrew Smith, Chenghua Lin, Binh Nguyen, Yuqi Pan, Yuhong Chou, Zefan Cai, Zhenhe Wu, Yongchi Zhao, Tianyu Liu, Jian Yang, Wangchunshu Zhou, Chujie Zheng, Chongxuan Li, Yuyin Zhou, Zhoujun Li, Zhaoxiang Zhang, Jiaheng Liu, Ge Zhang, Wenhao Huang, and Jason Eshraghian. A survey on latent reasoning, 2025. URL https://arxiv.org/abs/2507.06203.

## A IMPLEMENTATION DETAILS

## A.1 COMPOSER ARCHITECTURE

The latent composer $\Phi _ { \phi }$ consists of an encoder $\operatorname { E n c } _ { \psi }$ and a compressor $\mathrm { C o m p } _ { \chi } ,$ with $\phi : = ( \psi , \chi )$ The encoder uses the frozen backbone augmented with a trainable LoRA adapter (rank $8 , \alpha { = } 1 6$ dropout 0, applied to all seven linear projections per transformer block: $\mathbf { W } _ { q } , \mathbf { W } _ { k } , \mathbf { W } _ { v } , \mathbf { W } _ { o } ,$ gate, up, and down projections). During encoding, the task description is prepended to each retrieved experience to enable task-conditional compression, so that the QFormer’s cross-attention can condition on the current task when compressing the experience (see Appendix A.5 for the exact format).

The compressor is a QFormer-style perceiver where learned queries cross-attend to the encoder’s hidden states. It has 8 cross-attention layers with shared parameters (i.e., all layers reuse the same weights), attention heads and hidden dimension inherited from the backbone model, and a feedforward multiplier of 4. Learned queries $\mathbf { Q } _ { \boldsymbol { \chi } } \in \mathbb { R } ^ { K \times d }$ are initialized from $\mathcal { N } ( 0 , 1 / \sqrt { d } )$ . The compressor operates independently on each retrieved experience, producing K latent tokens per item; the outputs are concatenated to form the full latent context $\scriptstyle c _ { \phi } .$ . Based on the sensitivity analysis in Section 4.3, we set K=32 tokens per retrieved experience for the final LOPD training runs. The encoder LoRA and QFormer are updated during both cold-start and joint optimization; the backbone weights remain frozen throughout.

## A.2 COLD-START INITIALIZATION

The composer is cold-started by supervised finetuning on retrieved-experience–trajectory pairs with frozen backbone. The training objective is next-token NLL on assistant tokens with latentaugmented input:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { i n i t } } ( \phi ) = - \mathbb { E } _ { ( x , y ^ { \star } ) \sim \mathcal { D } _ { \mathrm { e x p } } , \mathcal { E } = \mathrm { R e t } ( x ) } \left[ \log \pi _ { \bar { \theta } } ( y ^ { \star } \mid s , c _ { \phi } ( x , \mathcal { E } ) ) \right] , } \end{array}\tag{14}
$$

where $y ^ { \star }$ is a successful trajectory and $\bar { \theta }$ denotes the frozen backbone. Concretely, the composer first produces latent tokens $c _ { \phi }$ from retrieved experiences and inserts them into the input embedding sequence at designated placeholder positions. The frozen backbone then performs a standard forward pass over this latent-augmented input. Although the backbone parameters receive no gradient updates, the loss gradient flows backward through the frozen layers’ activations to the latent token positions, and from there to the QFormer and LoRA parameters—the same mechanism as prefix-tuning.

The training data is synthesized from the base model’s own rollouts—no external expert or stronger model is required—and filtered by task success. The data volume and number of training steps are adjusted per domain. Training uses AdamW with learning rate $1 0 ^ { - 5 }$ , batch size 8, gradient clipping 3.0, task-conditional compression, and $n _ { \mathrm { r e t } } { = } 3$ retrieved experiences per task. The LoRA adapter and QFormer are updated during cold-start and remain trainable during subsequent joint optimization; the backbone remains frozen.

## A.3 RETRIEVAL CONFIGURATION

Let $B = \{ ( x _ { i } , \tau _ { i } ) \} _ { i = 1 } ^ { | B | }$ denote the experience bank, where each entry pairs a task description $x _ { i }$ with a successful trajectory $\tau _ { i }$ . A dense encoder $f _ { \mathrm { r e t } }$ maps text to a unit-norm embedding. We adopt asymmetric encoding: document-side embeddings encode the concatenation of task description and trajectory, while query-side embeddings encode only the task description with an instruction-aware query prompt:

$$
z _ { i } ^ { \mathrm { d o c } } = \frac { f _ { \mathrm { r e t } } ( [ x _ { i } ; \tau _ { i } ] ) } { \| f _ { \mathrm { r e t } } ( [ x _ { i } ; \tau _ { i } ] ) \| _ { 2 } } , \qquad z _ { q } = \frac { f _ { \mathrm { r e t } } ^ { \mathrm { q u e r y } } ( x ) } { \| f _ { \mathrm { r e t } } ^ { \mathrm { q u e r y } } ( x ) \| _ { 2 } } , \qquad z _ { i } ^ { \mathrm { d o c } } , z _ { q } \in \mathbb { R } ^ { d _ { \mathrm { r e t } } } .\tag{15}
$$

Given a query task x, retrieval returns the $n _ { \mathrm { r e t } }$ entries with the highest cosine similarity:

$$
\mathrm { R e t } ( \boldsymbol { x } ) = \mathrm { t o p } \mathrm { - n } _ { \mathrm { r e t } } \left\{ ( \boldsymbol { x } _ { i } , \boldsymbol { \tau } _ { i } ) \in \mathcal { B } \ \middle | \ z _ { q } ^ { \top } z _ { i } ^ { \mathrm { d o c } } \right\} .\tag{16}
$$

We use QWEN3-EMBEDDING-8B as $f _ { \mathrm { r e t } }$ , producing $d _ { \mathrm { r e t } } { = } 4 { , } 0 9 6$ -dimensional embeddings. The index is implemented as a FAISS Index $\vec { : } \perp \mathsf { a t } \mathtt { I P }$ for exact inner-product search over the precomputed document embeddings. The bank B is constructed offline exclusively from successful rollouts generated on the training split: agentic trajectories must exceed a task-reward threshold, while coding trajectories must pass all test cases. No evaluation task, trajectory, or outcome is ever inserted into the bank, and the bank is frozen before evaluation. Consequently, LOPD receives neither evaluation-set information nor an additional task corpus; it uses only experience produced from the same training-task split available to the baselines. Trajectories are stored in an observation-lite format that omits verbose environment observations to reduce storage and encoding length. All query embeddings $z _ { q }$ are precomputed and cached per task, so the retriever does not need to be loaded during training or evaluation.

Each bank entry stores the task description followed by a linearized action–result trace in observation-lite format, as illustrated below:

Example experience-bank entry (agentic, truncated)   
Task:   
Acting on behalf of Alice Chan, update her health policy XJ-439220: add Psychiatric Care   
coverage (\$3500 limit, \$300 deductible), increase ER coverage limit to \$7000, .   
Example steps (from a different task, NOT yet executed):   
1. get policy by policy number("XJ-439220") -> policy id=POL-001   
2. get exclusions for policy("POL-001") -> exclusion id=EXCL-001, ...   
3. remove exclusion from policy("POL-001", "EXCL-002") -> Success   
4. add coverage item to policy("POL-001", ...) -> Coverage item added   
5. update coverage limit or deductible("COV-002", limit=7000) -> Updated

## A.4 LATENT INJECTION MECHANISM

The latent tokens produced by the composer must be injected into the backbone’s input in a way compatible with the inference engine’s pipeline. Unlike prior latent-state injection work that typically relies on HuggingFace Transformers or TRL for direct inputs embeds manipulation, our implementation operates on top of production inference engines (SGLang and vLLM).

Placeholder strategy. A text-level sentinel string <|LATENT PH|> is inserted into the first user message of the chat template, sandwiched between natural-language framing text (e.g., “Thefollowing is a reference example. . . ” before and “Now complete your task. . . ” after). The full prompt is then rendered via the tokenizer’s chat template and split on the sentinel. Each half is tokenized independently, and J · K dummy token IDs (using the tokenizer’s pad token) are inserted between them:

$$
\mathrm { i n p u t . i d s } = [ \underbrace { t _ { 1 } , \dots , t _ { a } } _ { \mathrm { b e f o r e } } , \underbrace { t _ { \mathrm { p a d } } , \dots , t _ { \mathrm { p a d } } } _ { \substack { J \cdot K \mathrm { p l a c e h o l d e r s } } } , \underbrace { t _ { a + 1 } , \dots , t _ { L } } _ { \substack { \mathrm { a f t e r } } } ] .\tag{17}
$$

This produces a contiguous span of placeholder positions at known absolute indices, without any tokenizer modification.

Embedding replacement. The engine embeds the full input ids as usual, producing $\textbf { X } \in$ $\mathbb { R } ^ { L \times d }$ . Before the first transformer layer, the placeholder embeddings are replaced with the latent tokens from the composer:

$$
\mathbf { X } [ p _ { k } ]  \langle e _ { k } \rangle , \quad k = 1 , \ldots , J \cdot K .\tag{18}
$$

The replacement uses torch.cat over slices (rather than in-place assignment) so that autograd can track gradients through $\langle e _ { k } \rangle$ back to the composer parameters during training. The modified embedding tensor is then passed through the backbone’s transformer layers without any architectural change; the operation is transparent to the engine’s KV-cache, attention mask, and batching logic.

Engine-specific implementations. SGLang provides a positional embed overrides interface that accepts embeddings at specified absolute positions and performs the replacement in-GPU. vLLM provides a prompt embeds interface: the full sequence is first embedded on CPU via a frozen copy of the embedding table, the latent positions are overwritten with the composer’s output, and the complete embedding tensor is submitted to the engine. In both cases, the backbone model is unmodified.

Equivalence verification. To confirm that the placeholder-based injection introduces no numerical artifacts, we replaced the latent span with known token embeddings from the vocabulary and compared the token-ID input path against the embedding-tensor input path. The two paths produce identical greedy-decoded sequences with negligible log-probability differences, verifying that latent injection is numerically equivalent to standard token-ID forwarding.

Encoder LoRA isolation. The encoder $\mathrm { E n c } _ { \psi }$ uses a LoRA adapter to specialize hidden-state extraction for experience compression. This adapter is explicitly disabled during the main forward pass (both at inference and during teacher evaluation in training), so that the model behaves as a pure base actor conditioned on the injected latent context. This dual-use pattern—LoRA active for encoding, inactive for generation—avoids interference between the two roles.

## A.5 PROMPT AND INTERACTION FORMAT

Agentic system prompt. For tool-use tasks (EnvScaler, ACEBench), the agent receives a system prompt instructing it to complete tasks via step-by-step tool invocation:

Agentic system prompt (EnvScaler / ACEBench)   
You are a helpful assistant. When given a specific task, your goal is to complete it in an   
interactive environment by making step-by-step use of available tools.   
- Before completing the task, at each step, select a tool from the tool list and fill in   
all required parameters, making sure that the values are valid. Avoid making parallel tool   
calls in one step.   
- When you believe the task has been completed, respond only with ‘Task Completed’ to end   
the trajectory, without adding any other content or making any tool calls.   
- It is recommended to first call query tools to gather sufficient information, then use   
modification tools to complete the task. Adjust actions promptly based on the feedback   
from the environment.

Coding system prompt. For TACO and LiveCodeBench, the model receives a minimal system instruction:

Coding system prompt (TACO / LiveCodeBench)   
You are an expert Python programmer. You will be given a question (problem specification)   
and will generate a correct Python program that matches the specification and passes all   
tests.

HumanEval+ and MBPP+ use no system prompt, following the official EvalPlus evaluation protocol.   
All coding benchmarks use Python as the target language.

Latent-context framing text. The latent privileged context is injected into the first user message, wrapped by domain-specific framing text. The framing instructs the model to treat the injected content as a reference from a different task, not as instructions to follow verbatim.

![](images/a6e59b960f082776e76784c0bf3f000fe97f4c3e45967197ba1f994d7fa02ea8.jpg)  
The latent tokens produced by the composer replace the framing’s interior (as described in Appendix A.4).

Encoder input format. When the encoder Enc<sub>ψ</sub> processes a retrieved experience for compression, the input is formatted as “Task to solve:\nx\n\nReference past $\mathtt { t r a j e c t o r y : } \mathtt { n } m _ { j } ,$ , where x is the current task description and $m _ { j }$ is the retrieved trajectory. This task-conditional formatting allows the QFormer to attend to task-relevant features when producing the latent tokens.

## B EXPERIMENT DETAILS

## B.1 BASELINE SETUP

All trainable methods share the same base model, training task distribution, on-policy rollout budget (32 rollouts per step), optimizer (AdamW, learning rate $1 0 ^ { - 5 }$ , gradient clipping 1.0), and evaluation protocol. Each distillation-based method uses the KL direction prescribed by its original paper: OPSD and SDFT use forward KL over the full vocabulary, SDPO uses reverse KL with top-K truncation and a tail bucket, and Skill-SD uses sampled-token reverse KL with importance weighting. SDFT and OPSD require ground-truth demonstrations as privileged context for the teacher. These are drawn from a shared pool of verified-successful trajectories: for agentic tasks, trajectories must exceed a reward threshold; for coding tasks, solutions must pass all test cases. The pool is first populated from the base model’s own rollouts; when its coverage is insufficient, we synthesize additional ground-truth trajectories by querying stronger models (DeepSeek-V4-Pro and Qwen3.7-Max in our experiments). Below we describe the method-specific configurations.

Vanilla. The unadapted backbone model, evaluated without any post-training.

GRPO. Standard group relative policy optimization (Shao et al., 2024). Each step samples G=4 rollouts per task with stochastic decoding; group-normalized advantages weight a PPO-clip objective with $\epsilon _ { \mathrm { l o } } { = } \epsilon _ { \mathrm { h i } } { = } 0 . 2$ . No teacher or privileged context is used; the training signal comes entirely from environment rewards.

SDFT. Demonstration-conditioned distillation (Shenfeld et al., 2026). The teacher receives oracle trajectories from the shared pool as textual in-context demonstrations appended to the first user message. The teacher is an EMA shadow of the student $( \alpha _ { \mathrm { E M A } } { = } 0 . 0 1$ , synced every step). The distillation loss is forward KL over the full vocabulary, following the official implementation.

SDFT demonstration injection format   
Appended to user message:   
This is an example for a response to the question:   
{demo text}   
Now answer with a response of your own, including the thinking process.

OPSD. On-policy self-distillation (Zhao et al., 2026). The teacher receives oracle trajectories from the shared pool directly as privileged context in the system message. The teacher is frozen throughout training. The distillation loss is forward KL over the full vocabulary, as prescribed by the original paper.

OPSD GT injection format   
A ended to s stem messa e:   
The following is a reference plan for the upcoming task. Use it only as guidance for your   
own reasoning; you must still interact with the environment to complete the task.   
{gt text}   
After reading the reference solution above, make sure you truly understand the reasoning   
behind each step. Now, using your own words and independent reasoning, derive the same   
final answer.  
SDPO. Self-distillation policy optimization (Hubotter et al.¨ , 2026). The teacher is an EMA shadow of the student (update rate 0.05), conditioned on successful sibling trajectories sampled within the same training step (G=4 rollouts per task). Unlike SDFT and OPSD, SDPO does not use an external trajectory pool; it filters from its own rollouts using the same domain-specific success criterion (reward threshold for agentic tasks, full test-case pass for coding tasks).

Skill-SD. Skill-conditioned self-distillation (Wang et al., 2026). Each task selects one skill from a skill bank via UCB1 $\scriptstyle ( c = { \sqrt { 2 } } )$ . Skills are distilled from rollout trajectories into structured summaries and injected into the teacher’s first user message:

Skill-SD skill injection format   
Appended to user message:   
Here is a skill summary from a similar past task that may help guide your approach:   
<sub>\*\*</sub>What works:<sub>\*\*</sub> {success analysis}   
<sub>\*\*</sub>What to avoid:<sub>\*\*</sub> {mistake analysis}   
<sub>\*\*</sub>Suggested workflow:<sub>\*\*</sub> {golden workflow}   
Now solve the current task using your own reasoning.

The loss combines PPO-clip $( \epsilon _ { \mathrm { l o } } { = } 0 . 2 , \epsilon _ { \mathrm { h i } } { = } 0 . 2 8 )$ with importance-weighted sampled-token reverse KL $( \lambda _ { \mathrm { s d l } } { = } 0 . 0 0 1 )$ , following the original paper.

## B.2 TRAINING DATA AND DOMAIN SETUP

All methods are trained exclusively on the two datasets listed in Table 4; no additional task dataset is used. In particular, the LOPD experience bank contains only rollouts from the corresponding training split and excludes every evaluation task and trajectory. The agentic training corpus (EnvScaler) covers diverse tool-interactive scenarios including insurance, logistics, e-commerce, and healthcare, with each task requiring multi-turn API interactions to complete. The coding training corpus (TACO subset of DeepCoder) consists of competitive programming problems with verified test cases, spanning algorithmic topics such as dynamic programming, graph traversal, and string manipulation. The environment reward differs by domain: EnvScaler returns a continuous reward in [0, 1] reflecting the fraction of subtasks completed, while TACO returns a binary reward (1 if all test cases pass, 0 otherwise).

<table><tr><td>Parameter</td><td>Agentic</td><td>Coding</td></tr><tr><td>Training data</td><td>EnvScaler</td><td>TACO (DeepCoder)</td></tr><tr><td>Training tasks</td><td>2,349</td><td>~7,000</td></tr><tr><td>Total generation budget (tokens)</td><td>40,960</td><td>16,384</td></tr><tr><td>Max environment steps</td><td>30</td><td>— (single-turn)</td></tr><tr><td>Decoding temperature</td><td></td><td>0.7</td></tr><tr><td>top-p</td><td></td><td>0.95</td></tr><tr><td>top-k (sampling)</td><td></td><td>20</td></tr><tr><td>LOPD constraint parameters</td><td></td><td></td></tr><tr><td>Composer learning rate (lrφ)</td><td></td><td>10-5</td></tr><tr><td>Anchor weight (λ)</td><td></td><td>0.2</td></tr><tr><td>Dual step size (ηβ)</td><td></td><td>0.5</td></tr></table>

Table 4: Domain-specific training configurations.

## B.3 EVALUATION PROTOCOLS

Table 5 summarizes the inference configuration for each benchmark. All benchmarks use thinking mode enabled.

<table><tr><td>Benchmark</td><td>Tasks</td><td>Temp</td><td>top-p</td><td>top-k</td><td>Budget (tokens)</td><td>Max steps</td></tr><tr><td>EnvScaler</td><td>200</td><td>0.7</td><td>0.95</td><td>20</td><td>40,960</td><td>30</td></tr><tr><td>BFCL-v3</td><td>4×200</td><td>0.7</td><td>0.95</td><td>20</td><td>40,960</td><td>30</td></tr><tr><td>ACEBench</td><td>50</td><td>0.7</td><td>0.95</td><td>20</td><td>40,960</td><td>20</td></tr><tr><td>LiveCodeBench</td><td>279+131</td><td>0.7</td><td>0.95</td><td>20</td><td>16,384</td><td>一</td></tr><tr><td>HumanEval+</td><td>164</td><td>0.7</td><td>0.95</td><td>20</td><td>16,384</td><td></td></tr><tr><td>MBPP+</td><td>378</td><td>0.7</td><td>0.95</td><td>20</td><td>16,384</td><td></td></tr></table>

Table 5: Evaluation inference configurations per benchmark.

EnvScaler. We evaluate on a held-out set of 200 tasks, randomly sampled from the full task pool and fully disjoint from the training split.

BFCL-v3. We use the official Berkeley Function Calling Leaderboard V3 codebase with four multi-turn subsets: base, missing-function, missing-parameter, and long-context. Models are served in function-calling (FC) mode. We report per-subset and average scores.

ACEBench. ACEBench evaluates 50 tasks split into multi-step (20) and multi-turn (30) categories.   
A user simulator (deepseek-v4-flash via API) drives the multi-turn dialogues.

LiveCodeBench. We use the official code generation lite evaluation harness and report pass@1 on release v5 (279 tasks, Aug 2024–Feb 2025) and release v6 (131 tasks, Feb–May 2025).

EvalPlus. We use HumanEval+ v0.1.10 (164 tasks) and MBPP+ v0.2.0 (378 tasks) from the offi cial EvalPlus codebase.

## B.4 ABLATION AND ANALYSIS SETUP

This section details the configuration of each analysis experiment in Section 4.3.

Effect of Joint Optimization (Figure 3). Each row trains LOPD with a different margin m and evaluates the resulting student on the EnvScaler test set without retrieval, the composer, or latent privileged context at inference, isolating the effect of joint optimization on the resulting policy. The “Frozen $\phi _ { 0 } { } ^ { , }$ row trains with a frozen composer (no joint optimization). All jointly-optimized rows share the same training configuration except for m.

Training Dynamics (Figure 4). Mean reward is periodically evaluated on the EnvScaler test set throughout training using QWEN3-4B.

Sensitivity Analysis (Figure 5). (a) Latent-token capacity: we train separate LOPD variants with $K \in \{ 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}$ and evaluate each resulting student alone on the EnvScaler test set $( n _ { \mathrm { r e t } } { = } 3$ during training). (b–e) Retrieval count: we train separate LOPD variants with $K { = } 3 2 , m { = } 0 . 0 5 .$ , and $n _ { \mathrm { r e t } } \in \{ 1 , \ldots , 1 0 \}$ , then evaluate the resulting students on the EnvScaler test set and ACEBench without retrieval, the composer, or latent privileged context.

Behavioral Internalization (Table 3). Vanilla: the unadapted QWEN3-4B backbone. Base + Composer: the jointly optimized composer (m=0.05) paired with the unadapted backbone and latent privileged context $( n _ { \mathrm { r e t } } { = } 3 , K { = } 3 2 )$ . LOPD: the distilled student policy evaluated without retrieval, the composer, or latent privileged context. All three are evaluated on the EnvScaler test set with otherwise identical inference settings.