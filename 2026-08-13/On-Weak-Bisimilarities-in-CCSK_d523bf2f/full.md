# On Weak Bisimilarities in CCSK

<sub>Baptiste</sub> <sub>Vall´ee</sub>1[0009-0002-1268-3761] <sub>and</sub> <sub>Ivan</sub> <sub>Lanese</sub>2[0000-0003-2527-9995]

baptistevalleedupont@gmail.com, ivan.lanese@gmail.com

<sup>1</sup> Ecole Normale Sup´erieure Paris-Saclay (France)<sup>´</sup>

<sup>2</sup> Olas Team, University of Bologna/INRIA (Italy)

This is a pre-copy-editing, author-produced PDF of an article accepted for publication in RC 2026 following peer review. The definitive publisher-authenticated version is available online at https://link.springer.com/chapter/10.1007/978-3-032-30839-9

Abstract. In the context of CCSK, a reversible extension of CCS, we study diferent notions of bisimilarity (strong/weak, forward-only/reversible) and highlight their diferences and commonalities. In particular, for the weak reversible case, not previously studied in the literature, we propose two variants, dubbed directional and mixed bisimilarity, depending on whether τ actions should be in the same direction (forward/backward) as the action being matched or not. We show, in particular, that mixed bisimilarity is a congruence and completely abstracts away from τ actions.

Keywords: CCS, CCSK, Reversible Computation, Weak Bisimilarity, Behavioral Theory

## 1 Introduction

Building concurrent systems is challenging due to the complexity of reasoning about numerous possible interleavings, yet concurrency is essential in modern systems like the Internet, cloud computing, and parallel processing. Reversible computing, which allows systems to execute both forwards and backwards, recovering past states, has significant applications in low-energy computing [9], simulation [5], biological modeling [4,19], and program debugging [8,13,10]. Many of these applications involve concurrent systems, leading to the development of reversible extensions of concurrent process calculi such as CCS [7,18] and the π-calculus [6], and even of concurrent programming languages such as Erlang [11] and Go [17].

A main notion in the theory of process calculi is the notion of bisimilarity [20], allowing one to prove two processes equivalent, e.g., to prove an implementation equivalent to a more abstract specification. In particular, bisimilarity requires equivalent processes to be able to match each other actions, and in doing so going to processes which are still equivalent. While strong and weak bisimilarity (weak bisimilarity difers from the strong one as the former abstracts away from internal actions, focusing only on interactions with the context) have been extensively studied in concurrent systems, the literature lacks an analysis of weak bisimilarities in a reversible setting. Our study addresses this gap by investigating the relationships between diferent notions of bisimilarity (strong/weak, forward-only/reversible) in the context of CCSK [18], a causal-consistent reversible extension of Milner CCS [15]. In particular, in the definition of weak reversible bisimilarity, a main decision is whether auxiliary τ actions (representing internal steps) allowed in the simulation of some action α need to be in the same direction as α or not. The two alternatives lead to diferent equivalences.

We consider this work as a first step in the exploration of weak bisimilarity in a reversible setting, paving the way for a deeper exploration in the future. We claim as our main contributions the proposal of two notions of weak reversible bisimilarity (mixed bisimilarity in Definition 3.13 and directional bisimilarity in Definition 3.15, both in Section 3), the study of the relations between diferent notions of bisimilarity in CCSK (Section 4), and the study of which of these notions are congruences (Section 5). In particular, we show that mixed bisimilarity is a congruence (Theorem 5.6) and completely abstracts away from τ actions (Proposition 5.5 and Theorem 5.7). Another surprising result is that extending CCS bisimilarities to CCSK gives equivalences which are not congruences (Proposition 5.2), even when they are congruences in CCS, as in the case of strong bisimilarity.

## 2 CCSK

In this section we recall the main elements of CCSK, while referring to [18] for further details. We assume an infinite set of Names A, ranged over by $a , b , c , \ldots$ and a disjoint infinite set of Co-names ${ \overline { { \mathcal { A } } } } ,$ ranged over by ${ \overline { { a } } } , { \overline { { b } } } , { \overline { { c } } } , \dots ,$ where ∗ is an operator such that ${ \overline { { \overline { { a } } } } } = a$ . We call actions the elements of ${ \mathcal { A } } \cup { \overline { { { \mathcal { A } } } } } \cup \{ \tau \}$ where $\tau \not \in { \mathcal { A } }$ and $\overline { { \tau } }$ is undefined, ranged over by $\alpha , \beta , \ldots$ . Intuitively, names represent input actions, co-names represent output actions, and $\tau$ is an internal synchronisation. CCS processes, which we shall also call standard processes, are given by:

$$
P , Q : = 0 \left| \left| \alpha . P \right. \right| \left| \ J + Q \right. \left| \left| \ J \right. \right| Q \left| \right| ( \nu a ) P
$$

Intuitively, 0 is the inactive process, α.P is a process that performs action α and continues as $P , P + Q$ is nondeterministic choice, $P \mid Q$ is parallel composition and restriction $( \nu a ) P$ binds name a and the corresponding co-name a inside P. A name is bound if it is inside the scope of a restriction operator, free otherwise. Function $\mathtt { f n } ( P )$ computes the set of free names in process P. We set this convention: unary operators bind stronger than binary operators.

CCSK extends CCS with the possibility of executing backwards. In order to remember which input interacted with which output while going forwards, fresh keys are created at each forward step, and the same key is used to label an input and the corresponding output during a synchronisation.

We denote the set of keys by Keys, ranged over by $m , n , k , \ldots$ Prefixes, ranged over by π, are of the form $\alpha [ m ]$ or α. The former denotes that α has already been executed, the latter that it has not.

CCSK processes are given by:

$$
P , Q : = 0 \left| \left| \pi . P \right. \right| \left| \ P + Q \right. \left| \left| \ P \right. \right| Q \left| \right| ( \nu a ) P
$$

hence they are like CCS processes but for the fact that prefixes may be labelled with a key. In the following, we may drop trailing 0s.

Definition 2.1 (Context). A CCSK context is a process with a hole, as generated by the grammar below:

$$
C : = \bullet \parallel \pi . C \parallel C + Q \parallel P + C \parallel C \lvert Q \rvert \parallel P \lvert C \rvert \parallel ( \nu a ) C
$$

We denote with $C [ P ]$ the process obtained by replacing • with P inside C.

We use predicate std $( P )$ to mean that P is standard, that is none of its actions has been executed, hence it has no keys. We assume function $\mathtt { t o S t d } ( P )$ that takes a CCSK process P and gives back the standard process obtained by removing all keys from P.

We take from [12, Def. 2.1] the notions of free and bound keys.

Definition 2.2 (Free and bound keys). A key k is bound in a process X if it occurs either twice, attached to complementary prefixes, or once, attached to $a \tau p r e f i x .$ A key k is free if it occurs once, attached to a non-τ prefix.

Figure 1 shows the forward rules of CCSK. Backward rules in Figure 2 are obtained from forward rules by reversing the direction of transitions. Both relations rely on a definition of structural congruence allowing one to α-convert bound keys, applicable only at top level (this condition is needed to ensure that there are no other occurences of n in the context):

$$
P \equiv P [ n / m ] \qquad m { \mathrm { ~ b o u n d ~ i n ~ } } P , n \not \in \mathrm { k e y s } ( P )
$$

Rule (TOP) allows a prefix to execute. The rule generates a key m. Freshness of m is guaranteed by the side conditions of the other rules (cf. rule (PAR)). Rule (PREFIX) states that an executed prefix does not block execution. The two rules for (CHOICE) and the two for (PAR) allow processes to execute inside a choice or a parallel composition. The side condition of rule (CHOICE) ensures that at most one branch can execute. Rule (SYNCH) allows two complementary actions to synchronise producing $\mathrm { ~ a ~ } \tau .$ The key of the two actions needs to be the same. Rule (RES) allows an action which does not involve the restricted name to propagate through restriction.

The forward semantics of a CCSK process is the smallest relation $ { } _ { f }$ closed under the rules in Figure 1. Analogously, its backward semantics is the smallest relation $ _ { r }$ closed under the rules in Figure 2. The semantics is the union of the two relations. From

$$
\begin{array} { r l } { \mathrm { ( T o P ) } \quad } & { \underset { ( \mathcal { A } , P ) \leq M } { \leq \mathrm { a d } } ( P ) } \\ & { \underset { ( \mathcal { A } , P ) \leq \nu } { \leq \mathrm { a d } } \underset { f ( \mathcal { A } | \mathcal { C } ) \leq P } { \leq \mathrm { a d } } , \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad }  \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad }  \\ & { \quad \mathrm { ( c d u o t A ) } \quad \quad \quad \underset { ( \mathcal { A } , P ) < \nu } { \leq \mathrm { a d } } \underset { ( P ) < \nu } { \leq \mathrm { a d } } { \leq \mathrm { a d } } \underset { f ( \mathcal { A } ) < \nu } { \leq \mathrm { a d } } { \leq \mathrm { a d } } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

Fig. 1. Forward SOS rules for CCSK

now on, we let ϑ range over α[m] and $\mu$ range over α[m] with α $\neq \tau .$ Let $x , y , . . .$ . range over the set of directions $\{ f , r \}$ , for forward and reverse.

As standard in reversible computing (see, e.g., [18] or the notion of coherent process in [7]), all the developments consider only processes reachable from a standard process.

Definition 2.3 (Reachable process). A process $Q$ is reachable $i f f$ there exists a standard process P and a finite sequence of transitions from P to $Q$

## 3 Bisimilarities

In this section we introduce the various notions of bisimilarity we study. We start from CCS bisimilarities, and then move to bisimilarities specific for CCSK. Note that CCS can be seen as a subset of CCSK, considering only standard processes and only forward semantics. Hence, we will extend CCS bisimulations to CCSK by just considering forward transitions of CCSK terms. As a consequence, history becomes inaccessible, hence ideally irrelevant. However, requiring that a transition is matched by a transition with identical label would leak information on which keys are used in the history, as shown by the following example.

Example 3.1. We have $b { \xrightarrow { \ b [ n ] \ } } _ { f } \ b [ n ]$ while no transition with the same label is enabled from $a [ n ] . b$ . The second process hence cannot match the transition of the first one in the bisimulation game if equality of labels is required. Hence, keys of forward transitions leak information about which keys are used in the history.

In order to avoid this issue, we will remove keys from the labels. Hence, we define CCS semantics for CCSK processes as follows:

$$
\begin{array} { r l } & { \begin{array} { r l } & { \mathrm { ~ i n K e r n } } \\ & { \mathrm { ~ s i n i t y } } \\ & { \mathrm { ( i n K e ~ r i n g l e r ) } } \\ & { \mathrm { ( i n K e ~ t r i g n i t ) } } \end{array} \quad \begin{array} { r l } { \mathrm { s i n i t } ( P ) } & { \mathrm { ( i n K e ~ r n , F r ) } } \\ & { \mathrm { ( i n K e ~ r n , F r ) } } \\ & { \mathrm { ( i n K e ~ r n ) } } \end{array} } \\ & { \begin{array} { r l } { \mathrm { ( i n K e ~ t r a l t ) } } & { \mathrm { ~ i n K e r n } } \\ & { \mathrm { ( i n K e ~ r n u t ) } } \end{array} \quad \frac { P } { P } \frac { \mathrm { d i s i n g } } { P } _ { \mathrm { t } } \cdot \begin{array} { r l } { \mathrm { e n e q } ( Q ) } & { Q } \\ & { \mathrm { P l } } \end{array} \begin{array} { r l } & { \mathrm { d i s i n g } _ { \mathrm { r } } } \\ & { \mathrm { P l } } \end{array} \begin{array} { r l } & { \mathrm { ( i n K e ~ r n u t ) } } \\ & { \mathrm { P l } } \end{array} \begin{array} { r l } & { \mathrm { ~ i n K e r n } } \\ & { \mathrm { P l } } \end{array} \end{array} } \\ & { \begin{array} { r l } & { \mathrm { ( i n K e ~ r n u t ) } } \end{array} \begin{array} { r l } & { \mathrm { ( i n K e ~ r n u t ) } } \\ & { \mathrm { ( i n K e ~ r n u t ) } } \end{array} \begin{array} { r l } & { \mathrm { ~ i n K e r n } } \\ & { \mathrm { P l } } \end{array} \end{array} } \\ & { \begin{array} { r l } & { \mathrm { ( i n K e ~ r n u t ) } } \end{array} \begin{array} { r l } & { \mathrm { ~ i n K e r n } } \\ & { \mathrm { ( i n K e ~ r n u t ) } } \end{array} \begin{array} { r l } & { \mathrm { ~ i n K e r n } } \\ & { \mathrm { ( i n K e ~ r n u t ) } } \end{array} } \\ &  \begin{array} { r l } & { \mathrm { ( i n K e ~ r n u t ) } } \end{array} \begin{array} { r l } &  \mathrm { ~ o p ~ } \begin{array}  \end{array}
$$

Fig. 2. Reverse SOS rules for CCSK

Definition 3.2 (CCS semantics for CCSK processes). Given a CCSK process $P ,$ $P \stackrel { \alpha } {  } P ^ { \prime }$ if there is k such that $P \xrightarrow { \alpha [ k ] } _ { f } P ^ { \prime }$

We show now that the semantics above, defined on all CCSK processes, is indeed strictly related to the classical semantics of CCS defined in [14, Chapter $5 ] ^ { 3 }$ , which we denote as $ _ { c c s }$ . Let delHist() be the function that extracts the standard part of a process.

Definition 3.3 (delHist()). The delHist() function is inductively defined as follows:

$$
\mathsf { d e l H i s t } ( P ) = P \ i f \ \mathsf { s t d } ( P )
$$

$$
{ \bf d e l H i s t } ( a [ n ] . P ) = { \bf d e l H i s t } ( P )
$$

$$
{ \bf d e l H i s t } ( P + Q ) = { \bf d e l H i s t } ( P ) ~ i f \neg { \bf s t d } ( P )
$$

$$
{ \tt d e l H i s t } ( P + Q ) = { \tt d e l H i s t } ( Q ) \ i f \neg { \tt s t d } ( Q )
$$

$$
{ \bf d e l H i s t } ( P | Q ) = { \bf d e l H i s t } ( P ) | { \bf d e l H i s t } ( Q )
$$

$$
{ \tt d e l H i s t } ( ( \nu a ) P ) = ( \nu a ) { \tt d e l H i s t } ( P )
$$

The following proposition holds.

Proposition 3.4. Let $P , P ^ { \prime }$ be CCSK processes, and Q a CCS process. $I f P \stackrel { \alpha } {  } P ^ { \prime }$ then delHist $( P ) \stackrel { \alpha } { \to } _ { c c s }$ delHist(P<sup>′</sup>). If delHist $( P ) { \stackrel { \alpha } { \to } } _ { c c s } Q$ then there exists $P ^ { \prime }$ in CCSK such that $P \ { \stackrel { \alpha } { \to } } \ P ^ { \prime }$ and $Q = \mathsf { d e l H i s t } ( P ^ { \prime } )$

Proof. $\mathrm { B y }$ rule inspection.

Bisimilarities for CCS We start with the classical notion of CCS bisimilarity, extended as mentioned above.

Definition 3.5 (Strong Bisimulation). A symmetric relation R on CCSK processes is a strong bisimulation if whenever $P \mathcal { R } Q .$

$- \ i f \ P \ { \stackrel { \alpha } { \to } } \ P ^ { \prime }$ then there exists $Q ^ { \prime }$ such that $Q \stackrel { \alpha } { \to } Q ^ { \prime }$ and $P ^ { \prime } \mathcal { R } Q ^ { \prime }$

Let ∼ be the largest strong bisimulation. Two CCSK processes $P , Q$ are strongly bisimilar $i f \left( P , Q \right) \in \sim$

We give below some examples of strongly bisimilar processes.

Example 3.6 (Strongly Bisimilar CCSK Processes).

1. $a + a \sim a$

2. $a | a \sim a . a$

3. $a [ m ] \sim b [ n ]$

4. $a | b \sim a . b + b . a$

Note that Item 3 shows that strong bisimilarity (like all other CCS bisimilarities) abstracts away from the history. Item 4 is actually an instance of the Expansion Law [15], a cornerstone of the theory of classical CCS bisimilarity, whose general form is as follows:

$$
\begin{array} { l } { { P _ { 1 } \mid P _ { 2 } = \sum \{ \alpha . ( P _ { 1 } ^ { \prime } \mid P _ { 2 } ) : P _ { 1 } \stackrel { \alpha } {  } P _ { 1 } ^ { \prime } \} + \sum \{ \alpha . ( P _ { 1 } \mid P _ { 2 } ^ { \prime } ) : P _ { 2 } \stackrel { \alpha } {  } P _ { 2 } ^ { \prime } \} + } } \\ { { \phantom { \sum \{ \alpha . ( P _ { 1 } ^ { \prime } \mid P _ { 2 } ^ { \prime } ) : P _ { 1 } \stackrel { \alpha } {  } P _ { 1 } ^ { \prime } , \stackrel { \alpha } {  } \stackrel { \alpha } {  } P _ { 2 } ^ { \prime } , \alpha \stackrel { \alpha } {  } \tau \} } } } \end{array}
$$

where $\displaystyle \sum$ is n-ary choice.

It is well-known [18] that the Expansion Law does not hold for reversible calculi, and indeed we can provide a counterexample using strong forward-reverse bisimilarity (cf. Def. 3.11 and Ex. 3.12).

In order to introduce weaker notions of bisimilarity we need the notation below. Let $\Rightarrow = ( \stackrel { \tau } {  } ) ^ { * }$ be the reflexive and transitive closure of τ steps.

Definition 3.7 (Weak Bisimulation). A symmetric relation R on CCSK processes is a weak bisimulation if whenever $P \mathcal { R } Q$

$- \ i f \ P \stackrel { \tau } { \to } P ^ { \prime }$ then there exists $Q ^ { \prime }$ such that $Q \Rightarrow Q ^ { \prime }$ and $P ^ { \prime } \mathcal { R } Q ^ { \prime }$ ;

$- \ i f \ P \ { \stackrel { \alpha } { \to } } \ P ^ { \prime }$ with α ̸= τ then there exists $Q ^ { \prime }$ such that $Q \Rightarrow { \xrightarrow { \alpha } } \Rightarrow Q ^ { \prime }$ and $P ^ { \prime } \mathcal { R } Q ^ { \prime }$

Let ≈ be the largest weak bisimulation. Two CCSK processes $P , Q$ are weakly bisimilar $i f \left( P , Q \right) \in \approx$

Example 3.8 (Weakly Bisimilar CCSK Processes).

1. $\tau . a \approx a$

2. $a + \tau . a \approx a$

3. $a [ m ] \approx b [ n ]$

4. $a \left| b \approx a . b + b . a \right|$

We introduce also an intermediate notion, taken from [16], where $\tau$ steps need to be matched by at least one τ step.

Definition 3.9 (Semi-Weak Bisimulation). A symmetric relation R on CCSK processes is a semi-weak bisimulation if whenever $P \mathcal { R } Q$

if $P \ { \stackrel { \alpha } { \to } } \ P ^ { \prime }$ then there is $Q ^ { \prime }$ such that $Q \Rightarrow { \xrightarrow { \alpha } } \Rightarrow Q ^ { \prime }$ and $P ^ { \prime } \mathcal { R } Q ^ { \prime }$

$L e t \cong$ be the largest semi-weak bisimulation. Two CCSK processes $P , Q$ are semi-weak bisimilar if $( P , Q ) \in { \cong }$

Example 3.10 (Semi-Weakly Bisimilar Processes).

$$
- \ a + a \cong a
$$

$$
a | a \cong a . a
$$

$$
a [ n ] \cong b [ m ]
$$

$$
- \ a + \tau . a \cong \tau . a
$$

Bisimilarities for CCSK The bisimulations below make sense only in reversible calculi, since they consider both forward and backward transitions. We start with the notion of (revised) forward-reverse bisimulation from [12].

Definition 3.11 (Strong Forward-Reverse Bisimulation). A symmetric relation R is a strong forward-reverse bisimulation (also called FR-bisimulation) if whenever $P \mathcal { R } Q .$

if $P \stackrel { \vartheta } {  } _ { x } P ^ { \prime }$ then there is $Q ^ { \prime }$ such that $Q \stackrel { \vartheta } {  } _ { x } Q ^ { \prime }$ and $P ^ { \prime } \mathcal { R } Q ^ { \prime }$

Let ${ \sim } _ { F R }$ be the largest FR-bisimulation. Two CCSK processes $P , Q$ are FR-bisimilar if $( P , Q ) \in \sim _ { F R }$

Example 3.12 (FR Bisimilar Processes).

$$
- \ a + a \sim _ { F R } a
$$

$$
- \ a | a \ \nearrow _ { F R } a . a
$$

$$
- \ a | b \not \sim _ { F R } \ a . b + b . a
$$

As expected, instances of the Expansion Law do not hold any more.

We now introduce notations to study the weak bisimulations in CCSK, and to manipulate τ steps easily. Let $\Rightarrow _ { x } = ( \stackrel { \tau } { \to } _ { x } ) ^ { * }$ the reflexive and transitive closure of τ steps in the direction x. Let mixed τ reachability be $\Rightarrow _ { m } = ( \stackrel { \tau } { \to } _ { f } \cup \stackrel { \tau } { \to } _ { r } ) ^ { * }$ <sup>∗</sup>. This is an equivalence relation. Let $P \stackrel { \mu } { \Rightarrow } _ { m , x } P ^ { \prime }$ if ∃Q, Q<sup>′</sup> s.t. $( P \Rightarrow _ { m } Q \stackrel { \mu } {  } _ { x } Q ^ { \prime } \Rightarrow _ { m } P ^ { \prime } )$

We now define two variants of weak reversible bisimulation, which difer in whether the τ steps used to match some action $\mu$ need to be in the same direction as $\mu$ or not.

Definition 3.13 (Weak Mixed Bisimulation). A symmetric relation R is a weak mixed bisimulation (called mixed bisimulation) if whenever $P \mathcal { R } Q$ :

1. if $P \stackrel { \tau } { \to } _ { x } P ^ { \prime }$ then there exists $Q ^ { \prime }$ such that $Q \Rightarrow m \ Q ^ { \prime }$ and $P ^ { \prime } \mathcal { R } Q ^ { \prime } ;$

2. if $P \ { \stackrel { \mu } { \to } } _ { x } P ^ { \prime }$ then there exists $Q ^ { \prime }$ such that $Q \stackrel { \mu } { \Rightarrow } _ { m , x } Q ^ { \prime }$ and $P ^ { \prime } \mathcal { R } Q ^ { \prime }$

Let $\approx _ { m }$ be the largest mixed weak bisimulation. Two CCSK processes $P , Q$ are weakly mixed bisimilar $i f \left( P , Q \right) \in \approx _ { m }$

Intuitively, weak mixed bisimilarity abstracts away all the τ steps.

Example 3.14 (Mixed Bisimilar Processes).

$$
\textrm { -- } \tau . a \approx _ { m } a
$$

$$
- \ \tau \mid a \approx _ { m } a
$$

$$
\tau + a \approx _ { m } a
$$

$$
- \ \tau [ m ] + a \approx _ { m } a
$$

We also introduce a ”directional” weak bisimulation, in which the τ steps have to be in the same direction as the action $\mu .$ Let $P \stackrel { \mu } { \Rightarrow } _ { d , x } P ^ { \prime } \mathrm { i f f } \exists Q , Q ^ { \prime }$ s.t. $( P \Rightarrow _ { x } Q \stackrel { \mu } { \to } _ { x } Q ^ { \prime } \Rightarrow _ { x }$ $P ^ { \prime } )$

Definition 3.15 (Weak Directional Bisimulation). A symmetric relation R is a weak directional bisimulation (called directional bisimulation) if whenever $P \mathcal { R } Q$

1. if $P \stackrel { \tau } { \to } _ { x } P ^ { \prime }$ then there exists $Q ^ { \prime }$ such that $Q \Rightarrow _ { x } Q ^ { \prime }$ and $P ^ { \prime } \mathcal { R } Q ^ { \prime } ,$

2. if $P \ { \stackrel { \mu } { \to } } _ { x } P ^ { \prime }$ then there exists $Q ^ { \prime }$ such that $Q \stackrel { \mu } { \Rightarrow } _ { d , x } Q ^ { \prime }$ and $P ^ { \prime } \mathcal { R } Q ^ { \prime } ;$

Let ${ \approx } _ { d }$ be the largest directional bisimulation. Two CCSK processes $P , Q$ are directionally bisimilar if $( P , Q ) \in \approx _ { d }$

Example 3.16 (Directionally Bisimilar Processes).

$$
\begin{array} { l } { -  { \tau } . a \approx _ { d } a } \\ { -  { \tau } | a \approx _ { d } a } \\ { -  { \tau } + a \approx _ { d } a } \end{array}
$$

## 4 Relations between Bisimilarities

We now compare the notions of bisimilarity introduced in the previous section. Interestingly, the considered notions give rise to two hierarchies.

Proposition 4.1 (Hierarchies of Bisimilarities).

$$
\begin{array} { r } { 2 . \sim _ { F R } \subset \approx _ { d } \subset \left\{ \begin{array} { l l } { \approx _ { m } } & { . } \\ { \approx } & { . } \end{array} \right. } \end{array}
$$

$$
{ \sim } _ { F R } { \subset } { \sim } { \subset } \cong { \subset } { \approx } ;
$$

![](images/1188d4f59741f3dd82a27125c7ea98555b889f12788f261fc5329e5b5ffa3465.jpg)  
Fig. 3. Hierarchies of bisimilarities

Proof. Inclusions follow directly from the definitions. The proof of their strictness will actually be deferred to Proposition 4.2, providing witnesses for each of them. □

The first hierarchy relates strong forward-reverse bisimilarity to CCS bisimilarities. The second hierarchy instead focuses on reversible bisimilarities. Note that $\approx _ { d }$ is included in both $\approx _ { m }$ and ≈ (hence in their intersection). The two hierarchies are graphically represented in Fig. 3. The relations are written just inside the elliptical set they represent. The lines on the set for ${ \approx } _ { d }$ are just a visual help to distinguish the corresponding ellipse.

We now show that there are no other inclusions beyond the ones in Proposition 4.1, and that all the inclusions there are actually strict. Graphically, it means that all the areas in Figure 3 are not empty. We show this by providing examples of pairs of processes in each of them. Notably, all the examples are made of standard processes, hence none of these notions collapse when restricting the attention to standard processes. In other words, all these notions induce diferent equivalence relations on CCS processes.

Proposition 4.2 (Hierarchies are Strict on Standard Processes). All the areas in Fig. 3 are not empty, and each of them contains at least a pair of standard processes.

Proof. The ones below are witnesses for every area.

1. ${ \sim } _ { F R } : ( a + a , a )$ . The two as are indistinguishable.

2. $( \sim \cap \approx _ { d } ) \backslash \sim _ { F R } : ~ ( \tau . \tau . ( a . b + b . a ) + \tau . ( a \mid b ) , ~ \tau . \tau . ( a \mid b ) + \tau . ( a . b + b . a ) )$ . We have that $a . b + b . a \sim a \mid b$ is an instance of the expansion law, but the same equivalence is not valid for ${ \sim } _ { F R }$ . Under ${ \approx } _ { d }$ one can use equal subterms to match the challenge since they can be reached by taking a diferent number of τ steps.

3. $( \sim \cap \approx _ { m } ) \backslash \approx _ { d } : ( \tau . ( ( a . b + b . a ) + ( c | d ) ) + \tau . ( ( a | b ) + \tau . ( c . d + d . c ) ) , \tau . ( ( a | b ) + \tau . ( c . d . ) ) ,$ $\tau . ( c \left| \boldsymbol { d } \right. ) + \tau . ( ( a . b + b . a ) + ( c . d + d . c ) )$ . The equivalence holds under ∼ thanks to the expansion law. It holds also under $\approx _ { m }$ since the choice of which τ to execute can always be undone to select the desired branch. This is not the case under $\approx _ { d }$ where on the left one can reach a state where the only forward actions enabled are from $c . d + d . c .$ , while on the right no such state exists (if c.d + d.c is forward enabled, then also $a . b + b . a$ is enabled).

4. $\sim \backslash \approx _ { m } : ( a \ : | \ : b , \ a . b + b . a )$ . We use again an instance of the expansion law.

$5 . \ \cong \ \backslash ( \sim \cup \approx _ { m } ) ~ : ~ ( b \mid ( a + a . \tau ) , ~ a . ( \tau \mid b ) + a . b + b . a . \tau )$ . This fails under $\approx _ { m } .$ since the right hand side has no a | b to match the left hand side one. This fails under ∼ since on the right if one starts from $b ,$ there is no way to avoid $\mathrm { ~ a ~ } \tau ,$ , while this can be avoided on the left. This is not an issue under <sup>∼</sup>= since the a without τ can be matched by executing both a and τ.

6. ≈ $\backslash ( \cong \cup \approx _ { m } ) : ( a . \tau | b , a . b + b . a )$ . This holds under ≈ thanks to the expansion law and since the τ can be abstracted away. Instead, the expansion law fails under $\approx _ { m }$ and the τ needs to be matched by another τ under <sup>∼</sup>=.

7. $( \cong \cap \approx _ { d } ) \backslash \sim : ( a + a . \tau , \ a . \tau )$ . This fails under ∼ since on the left executing an a leads to a state where no τ can be performed. This is not an issue for $\approx _ { d }$ where the τ can be matched by staying idle. For <sup>∼</sup>=, left a can be matched by executing both a and τ.

8. $\approx _ { d } \ \backslash \ \cong \ : \ ( \tau , \ 0 )$ . This fails under <sup>∼</sup>= since the τ cannot be matched, instead under ${ \approx } _ { d }$ the τ can be matched by staying idle.

9. (≈ . (≈ $\cap \approx _ { m } ) \backslash ( \cong \cup \approx _ { d } )$

$( \tau . ( ( a . b + b . a ) + ( c | d ) ) + \tau . ( ( a | b ) + \tau . ( c . d + d . c ) ) + \tau . \tau . ( ( a | b ) + \tau . ( c | d ) ) + \tau . ( ( a . b + d . c . ) ) ,$ $b . a ) + ( c . d + d . c ) ) + \tau . \tau$ . This holds under ≈ thanks to the expansion law, and since τ and τ.τ are weakly bisimilar. However, the latter are not semi-weakly bisimilar, hence <sup>∼</sup>= fails. Also, this holds under $\approx _ { m } .$ , since it abstracts away from τ actions. $\approx _ { d }$ fails as well, for the same reason as in item 3.

10. ${ \approx } _ { m } ~ \backslash \approx : ~ ( a , ~ a + \tau )$ . This is well-known not to hold under ≈. Instead under $\approx _ { m }$ the τ step can be mimicked by staying idle since the a action remains enabled also after the $\tau ,$ since the τ can be undone to do a.

$$
1 1 . ~ ( \cong \cap \approx _ { m } ) \backslash ( \sim \cup \approx _ { d } ) ~ : ~ ( \tau . ( a . b + b . a + \tau + \tau . \tau ) + ( a \mid b ) , \tau . ( a . b + b . a + ( a \mid b ) + \tau . \tau ) ) .
$$

This fails under ∼ since $\tau \cdot \tau$ is not matched on the right, since afterwards a third $\tau$ would be enabled. This fails under $\approx _ { d }$ since after the first τ we can still go to $( a | b )$ on the right, but not on the left. This succeeds under $\approx _ { m }$ since τs are abstracted away, and terms without τ s are identical. This succeeds under $\cong$ since τ s can always be matched, and thanks to the expansion law. □

If instead of considering only pairs of standard processes we consider only pairs of non-standard processes, then all the areas remain non-empty.

Proposition 4.3. Each of the areas in $F i g .$ . 3 contains at least a pair of non-standard processes.

Proof. One can take the witnesses from the proof of Proposition 4.2 add ${ \mathrm { ~ a ~ } } \tau [ n ]$ prefix in front of both processes to obtain witnesses made of non-standard processes. □

Finally, if we consider pairs made of a non-standard process and a standard one, then ${ \sim } _ { F R }$ becomes empty. The other areas remain non-empty.

Proposition 4.4. Each of the areas in $F i g .$ 3 contains at least a pair made of a nonstandard process and a standard one, but for the area $^ { 1 , }$ corresponding $t o \sim _ { F R }$

Proof. The area for ${ \sim } _ { F R }$ is empty since $\sim _ { F R }$ can always distinguish a non-standard process, that can make a backward move, from a standard one, that cannot. For the others, one can take the witnesses from the proof of Proposition 4.2 add ${ \mathrm { ~ a ~ } } \tau [ n ]$ prefix in front of only one of the two components. This preserves all the CCS bisimilarities, which cannot observe the history, as well as the weak CCSK bisimilarities, where the backward $\tau [ n ]$ can be matched by the other process by staying idle. □

## 5 Congruence Properties of Bisimilarities

We first discuss whether the considered equivalences are congruences or not, namely in the case of strong CCS bisimilarity whether $P \sim Q \implies C [ P ] \sim C [ Q ]$ . Note that this implication makes sense only if all the involved processes are well-formed, hence we only consider this case.

Strong bisimilarity is a congruence in CCS [15]. Somehow surprisingly, its extension to CCSK is not a congruence, as shown in the counterexample below.

Example 5.1 (∼ is not a congruence).

$$
\tau [ n ] \sim 0 \implies a + \tau [ n ] \sim a + 0
$$

Processes on the left are strongly bisimilar since ∼ abstracts away from the history. Processes on the right are not strongly bisimilar since $a + \tau [ n ]$ cannot execute any forward move, while $a + 0$ can execute a.

The key point here is that forward equivalences abstract away from the history, but adding a choice where the added branch is a non-standard process disables the other

branch (which needs to be standard to ensure well-formedness). Thus, the same issue also occurs for the other forward equivalences we consider, namely weak (≈) and semi-weak (<sup>∼</sup>=) bisimilarities, as stated below.

Proposition 5.2 (Forward equivalences are not congruences in CCSK). None $o f \sim , \approx a n d \cong$ are congruences on CCSK terms.

Proof. Counterexample 5.1 proves the thesis for all the equivalences.

Note that for ≈ the usual problem of CCS that τ.a ≈ $a \implies \tau . a + b \approx a + b$ remains. However, $\cong$ is a congruence in CCS [16], but not in CCSK for the reason above.

Concerning reversible equivalences, ∼<sub>FR</sub> has been proved to be a congruence in [12, Proposition 4.9]. Instead, for weak directional bisimilarity a problem similar to the one above occurs.

Proposition 5.3 (Weak directional bisimilarity is not a congruence). $\approx _ { d }$ is not a congruence on CCSK terms.

Proof. The following counterexample proves the thesis:

$$
\tau [ n ] \approx _ { d } 0 \implies a + \tau [ n ] \approx _ { d } a + 0
$$

This is not the case for weak mixed bisimilarity, which, somehow surprisingly is a congruence. In order to clarify why this is the case, we first show a property of $\approx _ { m }$ which rules out counterexamples as the ones above.

Proposition 5.4. Assume $P \approx _ { m } Q$ where P is standard. Then $Q \ \Rightarrow _ { m } \ Q ^ { \prime }$ with $Q ^ { \prime }$ standard.

Proof. Assume towards a contradiction that there is no such $Q ^ { \prime } .$ . By definition, we have $Q ( \stackrel { \theta } { \to } _ { r } ) ^ { * } { \sf t o S t d } ( Q )$ . At least one of the steps is not a τ , otherwise we would have proven the thesis. Let us take the first such action. Then $Q$ can perform a backward non-τ action. However, such an action cannot be matched by $P ,$ , since $P$ is standard (and executing τ steps only enables backward τ steps, while we need to match a non-τ backward action).

We also show that indeed weak mixed bisimilarity completely abstracts away from τ steps.

Proposition 5.5 (Weak mixed bisimilarity abstracts away from τ steps). $P \Rightarrow _ { m } Q$ implies $P \approx _ { m } Q$

Proof. Thanks to the Loop Lemma (cf. [18, Prop. 5.1]), we also have $Q \Rightarrow _ { m } P$ . Hence, any challenge from $P$ can be matched by Q by first reducing to $P ,$ and vice versa.

Note that even if $P \Rightarrow _ { m } Q$ and $P \approx _ { m } Q$ are both equivalence relations, they do not coincide. $\mathrm { E . g . , } a + a \approx _ { m } a .$ , but $a + a \not \to _ { m } a$

Theorem 5.6 (Weak mixed bisimilarity is a congruence). $\approx _ { m }$ is a congruence on CCSK terms.

Proof. We prove the thesis by induction on the structure of the context, with a case for each operator. Cases for prefix and restriction are trivial. Let us consider choice and parallel composition.

Choice: we have to show that if $P \approx _ { m } Q$ then $P + R \approx _ { m } Q + R$ . Note that at most one among P and R can be non-standard due to well-formedness. Assume first both P and R are standard. If the challenge is from P, then Q inside $Q + R$ can match the challenge with the same sequence of moves used in $Q$ alone, all lifted thanks to rule (CHOICE). If R moves, and Q is standard, then the very same moves can be performed in both the cases, again using rule (CHOICE) to lift them. If Q is not standard, thanks to Proposition 5.4 above, we can first reduce $Q$ to toStd(Q), and then match the moves from R as above. Note that $Q$ and $\tt t o S t d ( Q )$ are mixed bisimilar thanks to Proposition 5.5, hence the reduction preserves mixed bisimilarity. Assume now P is non-standard. Then only P can move, and transitions can be lifted using rule (CHOICE) since by well-formedness R is standard. Assume now R is nonstandard. Analogously to the above, only R can move, in both the cases since P and Q need to be standard due to well-formedness.

Parallel composition: we have to show that if $P \approx _ { m } Q$ then $P \left| R \approx _ { m } Q \right| R$ . Assume $P \mid R \xrightarrow { \alpha } _ { f }$ . There are three subcases depending on which component contributes to the transition.

Transition from $P \colon Q$ can match the transition, and the matching computation can be lifted to $Q \mid R$ thanks to rule (PAR).

Transition from R: the same transitions can be done on both the sides, remaining in the relation.

Synchronization: Q can match transitions from P by hypothesis, and transitions from R can be performed on both the sides. This includes the components of the transitions that give rise to the synchronization. Hence we stay in the relation.   
The case of backward transitions is analogous.

We believe that the notion of mixed bisimilarity is very relevant. Indeed it provides a notion of bisimilarity which completely abstracts away from τ actions, which is coinductive (since it can be formulated as a bisimulation), and which is a congruence. We are not aware of any other notion of bisimilarity which has all these properties.

While leaving a more detailed analysis of this equivalence for future work, we discuss here some relevant axioms enabling to axiomatically reason on this equivalence. Notice that it makes sense to discuss about axioms since mixed bisimilarity is a congruence.

Various correct axioms for $\sim _ { F R }$ have been proposed in [12, Theorem 4.10]. While trivially all these axioms are correct also for $\approx _ { m }$ , we focus here on axioms which are specific of weak mixed bisimilarity.

The axioms in Fig. 4 hold for weak mixed bisimilarity.

$$
\tau . P = P\tag{TAU-PREF-M}
$$

$$
\tau + P = P\tag{TAU-CH-M}
$$

$$
\tau | P = P\tag{TAU-PAR-M}
$$

$$
\tau [ n ] . P = P
$$

$$
\tau [ n ] + P = P\tag{TAU-PREF-K}
$$

$$
\tau [ n ] \mid P = P\tag{TAU-CH-K}
$$

(TAU-PAR-K)

Fig. 4. CCSK axioms for weak mixed bisimilarity $\approx _ { m }$

Theorem 5.7. The axioms in Figure $\it 4$ are correct w.r.t. weak mixed bisimilarity.

Proof. τ moves can always be matched by the other process by staying idle, while moves from P can be matched by first doing or undoing τ steps as needed. Note that executing τ-steps moves from the top-3 rows to the bottom ones. □

The axioms in Fig. 4 are aligned with Proposition 5.5 in showing that mixed bisimilarity completely abstracts away from τ steps.

We remark that as shown in Figure $^ { 3 , }$ axioms which hold for weak bisimilarity ≈ do not necessarily hold for $\approx _ { m }$ . Let us now discuss the well-known Milner τ-laws of weak bisimilarity (collected in Fig. 5). The laws (TAU-CH) and (TAU-SEQ) follow directly from (TAU-PREF-M), and idempotence of + for the former.

Instead (TAU-DUPL-CH) fails, as shown below.

Example 5.8 ((TAU-DUPL-CH) does not hold). Consider the right-hand side challenge:

$$
\alpha . ( P + \tau . Q ) + \alpha . Q \stackrel { \alpha [ n ] } { \longrightarrow } _ { f } \alpha . ( P + \tau . Q ) + \alpha [ n ] . Q
$$

There are two possible answers from the left-hand side, namely:

$$
\begin{array} { r l } { \alpha . ( P + \tau . Q ) } & { \xrightarrow { \alpha [ n ] } _ { f } \alpha [ n ] . ( P + \tau . Q ) } \\ & { \xrightarrow { \alpha [ n ] } _ { f } \underline { { \tau [ m ] } } _ { f } \alpha [ n ] . ( P + \tau [ m ] . Q ) } \end{array}
$$

(We can also reach the same states after having undone and redone multiple times the $\tau \ \mathrm { s t e p . } )$ In both the cases, actions from P are enabled, directly in the first case, and by first undoing $\tau [ m ]$ in the second case. However, no action from P can be executed in the right-hand side above, since we need to undo one $\alpha [ n ]$ and do α on the other side.

## 6 Conclusion and Future Work

In this paper, we contrasted diferent notions of bisimilarity for CCSK processes, including two definitions of weak bisimilarities not previously discussed in the literature. We also proved that none of these notions are equivalent, not even if we restrict to standard

$$
P + \tau . P = \tau . P\tag{TAU-CH}
$$

$$
\alpha . \tau . P = \alpha . P\tag{TAU-SEQ}
$$

$$
\alpha . ( P + \tau . Q ) = \alpha . ( P + \tau . Q ) + \alpha . Q\tag{TAU-DUPL-CH}
$$

Fig. 5. τ laws for weak bisimilarity ≈

processes only. Notably, weak mixed bisimilarity turns out to be coinductive, to be a congruence, and to completely abstract away from τ-steps, making it a very interesting equivalence.

We hope that these results can be the basis of a more detailed study of bisimilarities in CCSK. We remark that such a deeper understanding may also impact classical concurrency theory, since there are strong relations [1,2] between reversible strong bisimilarities and history-preserving [21] and hereditary history-preserving [3] bisimilarities. Also, weak mixed bisimilarity induces an equivalence on CCS, which is a congruence and abstracts away from τ steps. Notice however that its current definition is not coinductive in CCS, since it relies on CCSK terms.

We present now a few other items for future research. First, we remark that while giving correct axioms for weak mixed bisimilarity, we have not provided a complete axiomatization. This is definitely a relevant item for future work (given the interesting properties of weak mixed bisimilarity) but not easy. Indeed, there are no complete axiomatizations for forward-reverse bisimilarity either, which we expect to be needed as first step before tackling the mixed case.

Another interesting item would be to understand how other bisimilarities, and more in general behavioral equivalences, can be extended from CCS to CCSK. Given that reversibility provides quite a strong observational power (as shown by the relations with history-preserving and hereditary history-preserving bisimilarities), it may be the case that some of them collapse.

## References

1. Cl´ement Aubert and Ioana Cristescu. How reversibility can solve traditional questions: The example of hereditary history-preserving bisimulation. In Igor Konnov and Laura Kov´acs, editors, 31st International Conference on Concurrency Theory, CONCUR 2020, Vienna, Austria (Virtual Conference), September 1-4, 2020, volume 171 of LIPIcs, pages 7:1–7:23. Schloss Dagstuhl - Leibniz-Zentrum f¨ur Informatik, 2020.

2. Cl´ement Aubert, Iain Phillips, and Irek Ulidowski. Bisimulations and reversibility. In Claudio Antares Mezzina and Alan Schmitt, editors, Components Operationally: Reversibility and System Engineering: Essays Dedicated to Jean-Bernard Stefani on the Occasion of His 65th Birthday, volume 16065 of Lecture Notes in Computer Science, pages 46–67. Springer, 2026.

3. M.A. Bednarczyk. Hereditary history preserving bisimulations or what is the power of the future perfect in program logics. Technical report, Polish Academy of Sciences, 1991.

4. Luca Cardelli and Cosimo Laneve. Reversibility in massive concurrent systems. Scientific Annals of Computer Science, 21(2):175, 2011.

5. Christopher D Carothers, Kalyan S Perumalla, and Richard M Fujimoto. Eficient optimistic parallel simulations using reverse computation. ACM Transactions on Modeling and Computer Simulation (TOMACS), 9(3):224–253, 1999.

6. Ioana Cristescu, Jean Krivine, and Daniele Varacca. A compositional semantics for the reversible π-calculus. In 28th Annual ACM/IEEE Symposium on Logic in Computer Science, LICS 2013, New Orleans, LA, USA, June 25-28, 2013, pages 388–397. IEEE Computer Society, 2013.

7. Vincent Danos and Jean Krivine. Reversible communicating systems. In International Conference on Concurrency Theory, pages 292–307. Springer, 2004.

8. Jakob Engblom. A review of reverse debugging. In Proceedings of the 2012 System, Software, SoC and Silicon Debug Conference, pages 1–6. IEEE, 2012.

9. Rolf Landauer. Irreversibility and heat generation in the computing process. IBM journal of research and development, 5(3):183–191, 1961.

10. Ivan Lanese, Naoki Nishida, Adri´an Palacios, and Germ´an Vidal. CauDEr: a causalconsistent reversible debugger for Erlang. In International Symposium on Functional and Logic Programming, pages 247–263. Springer, 2018.

11. Ivan Lanese, Naoki Nishida, Adri´an Palacios, and Germ´an Vidal. A theory of reversibility for Erlang. J. Log. Algebraic Methods Program., 100:71–97, 2018.

12. Ivan Lanese and Iain Phillips. Forward-reverse observational equivalences in CCSK. In RC, Lecture Notes in Computer Science, pages 126–143. Springer, 2021.

13. James McNellis, Jordi Mola, and Ken Sykes. Time travel debugging: Root causing bugs in commercial scale software. CppCon talk, 2017.

14. Robin Milner. A Calculus of Communicating Systems, volume 92 of Lecture Notes in Computer Science. Springer, 1980.

15. Robin Milner. Communication and concurrency. Prentice-Hall, Inc., 1989.

16. Ugo Montanari and Vladimiro Sassone. CCS dynamic bisimulation is progressing. In Andrzej Tarlecki, editor, Mathematical Foundations of Computer Science 1991, 16th International Symposium, MFCS’91, Kazimierz Dolny, Poland, September 9-13, 1991, Proceedings, volume 520 of Lecture Notes in Computer Science, pages 346–356. Springer, 1991.

17. Shunya Oguchi, Shoji Yuen, and Nobuko Yoshida. RevMiGo: Reversible channel-based communication in Go language. In Robert Gl¨uck and Robin Kaarsgaard, editors, Reversible Computation - 17th International Conference, RC 2025, Odense, Denmark, July 3-4, 2025, Proceedings, volume 15716 of Lecture Notes in Computer Science, pages 119–127. Springer, 2025.

18. Iain Phillips and Irek Ulidowski. Reversing algebraic process calculi. The Journal of Logic and Algebraic Programming, 73(1-2):70–96, 2007.

19. Iain Phillips, Irek Ulidowski, and Shoji Yuen. A reversible process calculus and the modelling of the ERK signalling pathway. In International Workshop on Reversible Computation, pages 218–232. Springer, 2012.

20. Davide Sangiorgi. Introduction to Bisimulation and Coinduction. Cambridge University Press, 2012.

21. Rob J. van Glabbeek and Ursula Goltz. Refinement of actions and equivalence notions for concurrent systems. Acta Informatica, 37(4/5):229–327, 2001.