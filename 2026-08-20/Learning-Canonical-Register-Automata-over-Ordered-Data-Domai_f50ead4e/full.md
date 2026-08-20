# Learning Canonical Register Automata over Ordered Data Domains

Yong Li<sup>1,2</sup> , Qiyi Tang<sup>3</sup> , and Di-De Yen<sup>3</sup>

<sup>1</sup> Key Laboratory of System Software (Chinese Academy of Sciences), Beijing, China 2 Institute of Software, Chinese Academy of Sciences, Beijing, China liyong@ios.ac.cn

<sup>3</sup> School of Computer Science and Informatics, University of Liverpool, UK {qiyi.tang, d.d.yen}@liverpool.ac.uk

Abstract. Register automata are finite automata equipped with memory that recognize data languages over infinite alphabets. In this work, we investigate active learning algorithms for deterministic register automata (DRAs) over ordered data domains—covering both dense domains, such as the rationals, and non-dense domains such as the integers. We show that the active learning problem for DRAs over both dense and nondense ordered domains can be treated within a single unified framework. More specifically, we develop and implement a polynomial-time active learning procedure for DRAs over ordered domains, using oracles for membership, equivalence and memorability queries. The memorability queries were originally introduced for learning DRAs over domains with identity tests. Our unified framework also leads to a new consequence: minimization of DRAs over the non-dense ordered domain of integers is decidable, extending a result previously known only for dense domains. Finally, we give improved complexity bounds of several decision problems for DRAs over ordered domains that are closely related to the queries used in active learning.

## 1 Introduction

The L<sup>∗</sup> algorithm, introduced by Angluin [2] within the Minimally Adequate Teacher (MAT) framework, enables a learner to infer an automaton representing an unknown system by interacting with a teacher through two types of queries: membership queries and equivalence queries. This query-based paradigm is often referred to as active learning [2, 38]. A membership query asks whether a word u belongs to the target language L, while an equivalence query asks whether a proposed automaton recognizes L. After collecting enough information from membership queries, the learner constructs a conjectured automaton and asks an equivalence query. The teacher either confirms that the conjecture is correct or returns a counterexample, which is then used by the learner to refine the conjecture. This iterative process continues until the learner successfully converges to the target automaton.

Active learning has been widely applied, including learning assumptions for compositional verification [15], software testing [16], detecting errors in network protocol implementations [36], regular model checking [13], extracting automata models for recurrent neural networks [40], verifying binarized neural networks [37], and explaining machine learning models [34].

In this paper, we consider active automata learning for deterministic register automata (DRAs) over ordered data domains. Register automata (RAs) are automata over infinite alphabets and have been extensively studied [1,6,7,14,24,33] motivated by real-world problems where data may have an unbounded domain. An RA stores data symbols in a finite number of registers, which can later be compared against new inputs. RAs are far more expressive than classical finite automata; however, they lose many desirable closure and decidability properties—language inclusion and equivalence become undecidable for nondeterministic RAs [24]. Fortunately, these problems are decidable for DRAs [33]. A key challenge in learning DRAs is that the learner must simultaneously reason about control-state behavior and data constraints, unlike in classical active learning.

In particular, comparisons between input data and register values come in two forms: (1) identity tests, which check equality; and (2) ordered comparisons, which in addition to equality also test whether the input is greater or smaller than a stored value. Most existing work focuses on DRAs restricted to identity tests [9,23,24,33], which limits their ability to capture ordering-based properties. For instance, checking whether an input sequence is sorted (i.e. in non-decreasing order) is recognizable by a DRA with ordered comparison but not by any RA using identity tests only. There has been prior work [10,12] on learning DRAs over ordered domains; however, it relies on tree-based queries over abstract symbolic words, whereas our approach operates directly on concrete words using standard membership queries together with memorability information.

We focus on a family of DRAs introduced in [7] that support ordered comparisons. These automata prohibit duplicate register values and, after each transition, permute registers so that their ordering reflects the order of last occurrence of their stored data. The languages recognized by such DRAs—often called regular data languages—admit a Myhill–Nerode-style characterization and have a canonical minimal representation that is simultaneously minimal in both the number of states and the number of registers [7, 25]. This property is crucial for active learning, which relies on a canonical representation of the target formalism, analogous to the Myhill–Nerode theorem [30, 32] that guarantees a unique minimal DFA for regular languages.

A subtle issue arises when comparing dense data domains (e.g. Q) and nondense domains (e.g. Z). For RAs restricted to identity tests, this distinction is irrelevant. However, for DRAs with ordered comparisons, non-dense domains introduce additional complications.

Contributions. We show that minimal DRAs over Q and minimal DRAs over Z that recognize the same language when restricted to integers are structurally equivalent. Consequently, the decidable complexity result of the minimization algorithm of [25], originally developed for dense domains, extends to non-dense domains as well. Accordingly, we obtain a unified active learning framework that applies uniformly to both dense and non-dense data domains. As a related result, we establish that the configuration reachability problem over Z is decidable. Based on the canonical form of DRAs, we develop an active learning algorithm for DRAs over ordered data domains. This algorithm uses polynomially many membership, equivalence and memorability queries, which precisely identify which symbols must be stored in registers when processing a data word<sup>4</sup>. The main technical challenge is that, unlike learning DRAs with identity tests [9,23], which rely on equality-based reasoning to capture freshness of values, ordered comparisons require reasoning modulo order-preserving renamings of data values. Such renamings are not unique, but we show that a canonical representative sufices to decide the relevant equivalence relations between words (cf. Lemma 12), making the learning independent of the choice of mapping. This breaks standard observation-table techniques based on sufix reasoning, which compare rows via equality of abstract future behavior.

To address this, table entries here are compared modulo canonical renamings and the subset of values stored in memory, yielding our notion S-consistency (cf. Section 4), which is not an equivalence relation and requires refined counterexample handling that updates symbolic transitions rather than only separating states. We implement this learning algorithm and demonstrate its efectiveness on a family of data languages and randomly generated DRAs.

We further study the complexity of decision problems for DRAs over ordered data domains, which are closely related to the queries arising in learning frameworks. We show PSPACE-completeness of language inclusion/intersection non-emptiness for DRAs over equality and ordered domains. For dense ordered domains, we additionally show PSPACE upper bounds for configuration equivalence, language equivalence, and memorability. For the memorability problem, our PSPACE upper bounds improve the previously known NEXPTIME upper bounds from [7, 25].

Related Work. Active learning of RAs has been investigated in [6, 9, 18, 23]. In [6], Permutation Deterministic Register Automata (PDRAs) were introduced, and our DRAs restricted to identity tests correspond to LAR-DRAs, a subclass of RAs with a fixed permutation policy. It was shown in [6] that minimization of LAR-DRAs is in PTIME, as both language equivalence and memorability are tractable. Moreover, they provide a PTIME active learning algorithm for LAR-DRAs using memorability queries. The learning algorithms in [18,23] consider the DRA model of [11] and do not use memorability queries; their runtime is exponential in both the number of registers and the length of the longest counterexample. Additionally, there is work [28] on learning nominal automata, which are equivalent to RAs in expressiveness [8]. These automata group infinite states into orbits under permutations; specifically, [28] provides an active learning algorithm for a subclass of nondeterministic nominal automata—including deterministic ones—over infinite domains with equality tests. The authors of [28] briefly note that their framework is intended to accommodate both equality and ordered domains, although working out the extension to ordered domains is left open. Venhoek, Moerman and Rot [39] further study computation over ordered nominal sets, contributing techniques for reasoning about infinite ordered structures in nominal settings. A passive learning algorithm for LAR-DRAs is given in [5].

Active learning for DRAs over ordered domains is studied in [10, 12], where data words consist of symbols from both a finite and an infinite set. Such data words are also considered in [23] and can be categorized into diferent patterns based on their finite components. This approach relies on tree queries instead of standard membership queries: for a given concrete prefix and a set S of abstract sufixes, the teacher returns a symbolic decision tree. This tree captures the behavior of the unknown automaton’s runs on the prefix followed by all possible concrete sufixes that satisfy the patterns in S. This tree-query technique is also used in [18] to reduce query complexity. Compared with memorability queries, tree queries provide richer information, since the resulting symbolic decision trees can expose both relevant register values and transition guards. In contrast, a memorability query directly returns the values of a prefix that may afect future behavior, independently of a particular sufix, and thus provides a more specialized oracle for identifying the required register contents. In [10,12], ordered domains with a ternary operation sum $( \ell , m , n )$ (signifying $\ell + m = n )$ and those with an operation inc $( m , n )$ (signifying $m + 1 = n )$ are also studied. Furthermore, the equivalence and emptiness problems are addressed for the more expressive model in [14], which supports ordered data domains and allows certain linear arithmetic constraints.

On the other hand, our DRAs over ordered data domains were first introduced in [7,25]. In this model, the duplication of register values is prohibited; this restriction does not reduce expressiveness and ensures the uniqueness of minimal automata. In our learning algorithm, in addition to equivalence and membership queries, the teacher also answers memorability queries. These queries were introduced in [6] to improve the complexity of learning DRAs over domains restricted to equality tests.

Symbolic finite automata (SFAs) constitute another automata model for infinite alphabets. Unlike RAs, SFAs retain the classical closure and decidability properties of finite automata but have strictly less expressive power [17]. There is extensive work on active learning for general and restricted classes of SFAs, see, for example, [3, 4, 19, 20, 22, 26, 27].

## 2 Preliminaries

We use R (resp. Q, Z, N) to denote the sets of real (resp. rational, integer, and non-negative) numbers. A set $S \subseteq \mathbb { R }$ is dense in R if, for every $s < t ,$ there exists $r \in S$ with $s < r < t [ 3 5 ]$ . Accordingly, Q is dense in R, whereas Z is not. For simplicity, we say $\mathbb { Q }$ and R are dense and $\mathbb { Z }$ is non-dense throughout the paper.

Let Σ be an alphabet and R a binary relation on $\Sigma ,$ where Σ may be finite or infinite. A (data) word is a finite sequence over Σ. For words $u = a _ { 1 } \ldots a _ { m } .$ $v = b _ { 1 } \ldots b _ { n }$ in $\varSigma ^ { * }$ and a symbol (or letter) $d \in \Sigma$ , we write d u if $d = a _ { i }$ for some $1 \leq i \leq m$ , and $u \cdot v$ denotes the concatenation $a _ { 1 } \dots a _ { m } b _ { 1 } \dots b _ { n }$ . The length of u is denoted by u . The relation R induces an equivalence relation on $\varSigma ^ { * }$ : for $u = a _ { 1 } \ldots a _ { m }$ and $v = b _ { 1 } \ldots b _ { n }$ , we write $u \sim _ { R }$ v if and only if (1) $m = n$ , and ${ \bf ( 2 ) } \left( a _ { i } , a _ { j } \right) \in {  R }$ if and only if $( b _ { i } , b _ { j } ) \in R$ for all $1 \leq i , j \leq n$

Each equivalence class τ of $\sim _ { R }$ is called a <sub>R</sub>-word type, or simply a type. We say τ has length n if every word u of type τ has $| u | = n$ . A language $L \subseteq \Sigma ^ { * }$ is a data language over $( \Sigma , R )$ if, for all words u, v of the same <sub>R</sub>-type, $u \in L \Leftrightarrow v \in L$ . In this paper, we consider only alphabet $\mathcal { Z } \in \{ \mathbb { R } , \mathbb { Q } , \mathbb { Z } \}$ and R being either < (the total order) or = (the identity relation) on Σ.

Let σ be a mapping on Σ<sup>∗</sup> with $\sigma ( u \cdot v ) = \sigma ( u ) \cdot \sigma ( v )$ for all $u , v \in \Sigma ^ { * }$ We call σ R-preserving if $\sigma ( u ) \sim _ { R }$ u for all $u \in \Sigma ^ { * }$ ; when R is $< ,$ , it is orderpreserving. For any $u = a _ { 1 } \ldots a _ { n } , v = b _ { 1 } \ldots b _ { n } \in \Sigma ^ { * }$ with $u \sim _ { R } v ,$ , there exists an R-preserving partial mapping σ such that $\sigma ( u ) = v$ . Moreover, if Σ is dense, σ can be chosen bijective as follows. For $( \Sigma , R ) = ( \mathbb { Q } , < )$ , let $u ^ { \prime } = a _ { 1 } ^ { \prime } \ldots a _ { n } ^ { \prime }$ and $v ^ { \prime } = b _ { 1 } ^ { \prime } \ldots b _ { n } ^ { \prime }$ be the sorted sequences of u and v, respectively. Then $\sigma _ { u  v }$ is defined as follows: (1) $\begin{array} { r } { \sigma _ { u \to v } ( c ) = \frac { ( c - a _ { i } ^ { \prime } ) ( b _ { i + 1 } ^ { \prime } - b _ { i } ^ { \prime } ) } { a _ { i + 1 } ^ { \prime } - a _ { i } ^ { \prime } } + b _ { i } ^ { \prime } } \end{array}$ , for each $a _ { i } ^ { \prime } \neq a _ { i + 1 } ^ { \prime }$ and $a _ { i } ^ { \prime } \leq c < a _ { i + 1 } ^ { \prime } ; ( 2 ) \sigma _ { u  v } ( a _ { i } ^ { \prime } ) = b _ { i } ^ { \prime } ,$ for each $a _ { i } ^ { \prime } = a _ { i + 1 } ^ { \prime } .$ where $i = 1 , \ldots , n - 1 ;$ $( 3 ) \ \sigma _ { u \to v } ( c ) ^ { \prime } = ( c - a _ { 1 } ^ { \prime } ) + b _ { 1 } ^ { \prime } , \mathrm { ~ i f ~ } c < a _ { 1 } ^ { \prime } ; \mathrm { ~ } ( 4 ) \ \sigma _ { u \to v } ( c ) ^ { \prime } = ( c - a _ { n } ^ { \prime } ) + b _ { n } ^ { \prime } , \mathrm { ~ i f ~ } c \geq a _ { n } ^ { \prime }$ By definition, $\sigma _ { u \to v }$ is order-preserving and bijective. Since $\sigma _ { u  v }$ is bijective, its inverse $\sigma _ { u  v } ^ { - 1 }$ also exists.

Register automata (RAs), also known as finite-memory automata, extend finite-state automata to infinite alphabets and recognize data languages. Several equivalent variants exist in the literature [6, 7, 24, 33]. In this paper, we adopt the definition of RAs from [7].

Definition 1. Given $k \in \mathbb N$ , an alphabet Σ, and a binary relation R on $\Sigma ,$ a k-register automaton $\left( k { - } R A \right) \ A$ over (Σ, R) is a tuple $( Q , q _ { 0 } , F , \varDelta )$ where: (1) Q is a set of states, partitioned into disjoint subsets $Q _ { 0 } , \ldots , Q _ { k } , ( 2 )$ q<sub>0</sub> $\in \mathrm { ~ \it Q _ { 0 } ~ }$ is the initial state, $( { \mathcal { 3 } } ) ~ F \subseteq Q$ is the set of final states, and $( 4 ) ~ \varDelta$ is a finite set of transitions of the form $( p , \tau , E , q )$ , where $p \in Q _ { i }$ and $q \in Q _ { j }$ for some $0 \leq i , j \leq k ,$ τ is a <sub>R</sub>-word type of length i + 1, $E \subseteq \{ 1 , \dots , i + 1 \}$ , and $j = i + 1 - | E |$

An RA is deterministic (DRA) if, for any two transitions $( p , \tau , E , q )$ and $( p ^ { \prime } , \tau ^ { \prime } , E ^ { \prime } , q ^ { \prime } )$ , we have $( E , q ) = ( E ^ { \prime } , \dot { q } ^ { \prime } )$ whenever $( p , \tau ) = ( p ^ { \prime } , \tau ^ { \prime } )$

A configuration of is a pair $( q , v )$ , where $q \in Q _ { i }$ and v is a word over Σ of length i, for some $0 \leq i \leq k$ . For configurations $( p , u )$ and $( q , v )$ and a symbol $a \in \Sigma$ , we write $( p , u ) \stackrel { a } {  } _ { \cal A } ( q , v ) \mathrm { o r } ( p , u ) \stackrel { \tau : { \cal E } } { \longrightarrow } _ { \cal A } ( q , v )$ if there exists a transition $\delta = ( p , \tau , E , q )$ such that $u \cdot a = a _ { 1 } \ldots a _ { n }$ is of type τ and v is obtained from $u \cdot a$ by removing all $a _ { i }$ with $i \in E$ , subject to the following two conditions: (1) ${ \mathrm { i f ~ } } | u | = k$ , then $E \neq \emptyset$ to prevent memory overflow; and $( 2 ) ~ j ~ \in ~ E$ whenever $1 \leq j < n$ and $a _ { j } = a _ { n } = a$ , which ensures no data duplication by requiring that

$$
 ( \begin{array} { c c c c c c c } { { q _ { 0 } } } \end{array} ) \xrightarrow { 0 : \emptyset } ( \begin{array} { c c c c c c } { { q _ { 1 } } } \end{array} ) \xrightarrow { 0 . 1 : \emptyset } ( \begin{array} { c c c c c c } { { q _ { 2 } } } \end{array} ) ^ { 0 \cdot 2 \cdot 1 : \{ 1 , 2 , 3 \} } ( \begin{array} { c c c c c c } { { q _ { 3 } } } \end{array} )
$$

Fig. 1: A well-typed DRA $A _ { m i d }$ recognizing $L _ { m i d }$ where sink state and its related transitions are omitted.

if the current input a is already stored, it must be removed. The subscript is omitted when clear from context.

Let $w = b _ { 1 } \dots b _ { m } \in \Sigma ^ { * } . \mathrm { A }$ sequence of configurations $\pi = \left( p _ { 0 } , u _ { 0 } \right) \ldots \left( p _ { m } , u _ { m } \right)$ is a run on w, written $( p _ { 0 } , u _ { 0 } ) \stackrel { w } { \longrightarrow } ( p _ { m } , u _ { m } )$ , if $\left( p _ { i } , u _ { i } \right) \ \xrightarrow { b _ { i } } \ \left( p _ { i + 1 } , u _ { i + 1 } \right)$ for all $0 \leq i < m$ . It is accepting if $( p _ { 0 } , u _ { 0 } ) = ( q _ { 0 } , \varepsilon )$ and $p _ { m } \in F$ . We say is complete if every word $u \in \Sigma ^ { * }$ has a run in from $( q _ { 0 } , \varepsilon )$ . The language recognized by is $L ( A ) = \{ w \in \Sigma ^ { * } \mid ( q _ { 0 } , \varepsilon ) \stackrel { w } { \longrightarrow } ( q _ { f } , u ) , q _ { f } \in F , u \in \Sigma ^ { * } \}$ , which is a data language over (Σ, R). Two RAs and <sup>′</sup> are equivalent if $L ( \mathcal { A } ) = L ( \mathcal { A } ^ { \prime } )$

Consider $L _ { m i d } = \{ a _ { 1 } a _ { 2 } a _ { 3 } \ | \ a _ { 1 } < a _ { 3 } < a _ { 2 } \}$ over $( \mathbb { Z } , < ) { : }$ : the DRA in Fig. 1 has states $q _ { 0 } , \ldots , q _ { 3 }$ , with $q _ { 0 }$ being the initial state and $q _ { 3 }$ a final state, and recognizes $L _ { m i d }$ . Each transition is labeled with $\tau : E .$ . For example, the transition from $q _ { 2 }$ to $q _ { 3 }$ is labeled with $0 \cdot 2 \cdot 1 : \{ 1 , 2 , 3 \}$ , where 0 2 1 is the word type $\tau$ and $\{ 1 , 2 , 3 \}$ is E. On the word $w = 3 \cdot 9 \cdot 7$ , the DRA has the accepting run $\pi = ( q _ { 0 } , \varepsilon ) { \overset { 3 } {  } } ( q _ { 1 } , 3 ) { \overset { 9 } {  } } ( q _ { 2 } , 3 \cdot 9 ) { \overset { 7 } {  } } ( q _ { 3 } , \varepsilon )$ , so $w \in L _ { m i d }$

We denote by the number of states in . An RA is state-minimal if $| { \mathcal { A } } | \leq | { \mathcal { A } } ^ { \prime } |$ for every equivalent RA <sup>′</sup>, and data-minimal if it is a k-RA and every equivalent $\mathcal { A } ^ { \prime }$ is a k<sup>′</sup>-RA with $k ^ { \prime } \geq k .$ is minimal if it is both state- and dataminimal. A state-minimal or data-minimal RA recognizing $L ( \mathcal { A } )$ always exists, but a minimal RA (even a DRA) need not. Consider the following example.

Example 2. For each $n \in \mathbb { N } ,$ , let $L _ { n }$ be the language of words w over Q that form a strictly increasing or decreasing sequence of length n. For example, when $n = 3 .$ the word $w = 2 \cdot - 1 \cdot - 3 . 5$ belongs to $L _ { n }$ , whereas $u = 1 \cdot 2$ and $v = 2 \cdot - 3 \cdot 0$ do not. Consider a 1-DRA $A _ { n }$ that recognizes $L _ { n }$ . It has two fashions—increasing and decreasing—and always stores the most recent input in its single register. Given $w = a _ { 1 } \ldots a _ { m } , \mathcal { A } _ { n }$ enters the increasing (resp. decreasing) fashion if $a _ { 2 } ~ > ~ a _ { 1 }$ (resp. $a _ { 2 } < a _ { 1 } )$ . For $2 < i < n$ , it remains in the current fashion provided $a _ { i }$ is larger (resp. smaller) than the value stored in the register; otherwise it rejects. Length $m = n$ is verified using control states. Since both fashions require $n - 1$ intermediate states (plus initial and final states), $A _ { n }$ has 2n states.

While $A _ { n }$ is data-minimal for $L _ { n }$ , it is not state-minimal since an equivalent $( n - 1 ) { \mathrm { - D R A } } \ A _ { n } ^ { \prime }$ with $| { \mathcal { A } } _ { n } ^ { \prime } | < | { \mathcal { A } } _ { n } |$ exists. This DRA stores the first $n - 1$ symbols of the input in its registers and checks on the $\mathrm { \textmu { y } }$ whether these values form a strictly increasing or strictly decreasing sequence. It needs $n + 2$ states: $n + 1$ states to verify the word length and one rejecting state to make $\mathcal { A } _ { n } ^ { \prime }$ complete.

As shown in Example 2, a minimal DRA need not exist for every data language L. To address this, we introduce well-typeness, which ensures the existence of a minimal well-typed DRA for every DRA-recognizable language. Notably, the canonical DRAs of [7] also satisfy this condition. Restricting DRAs to be welltyped preserves expressive power, though it may increase the number of states.

Definition 3. Let be an RA over $( \Sigma , R )$ . We call well-typed if for every two configuration transitions $( p , u ) \ { \overset { a } { \to } } _ { A } \ ( q , v )$ and $( p ^ { \prime } , u ^ { \prime } ) \stackrel { a ^ { \prime } } { \longrightarrow } _ { \cal A } ( q , v ^ { \prime } )$ ending in the same state q, we have $v \sim _ { R } v ^ { \prime }$

Intuitively, an RA is well-typed if the <sub>R</sub>-type of its registers is uniquely determined by the current state. In the remainder of the paper, all DRAs are assumed to be well-typed unless stated otherwise.

## 3 DRAs over Non-Dense Ordered Domains

In this section, we will present the Myhill–Nerode theorem for DRAs over ordered domains, which forms the basis for learning and minimization. We then solve open problems for DRAs over non-dense domains—most notably minimization and configuration reachability. Together, these results lay the foundation for a unified learning framework for DRAs over both dense and non-dense settings.

As with other state machines, minimization for DRAs is tightly related to the Myhill–Nerode theorem, as shown in [7]. We now introduce some notions needed to state this theorem for DRAs. For a language $L$ and two words $u , v ,$ , a word w is called an L-distinguishing extension for u and v if exactly one of the words $u \cdot w$ and $v \cdot w$ belongs to L. For regular languages $L ,$ two words are Nerode-congruent when no L-distinguishing extension exists. For data languages, however, word classification additionally requires the notion of memorability introduced in $\left[ 7 \right]$ Consider the language $L _ { 7 }$ in Example 2. Any DRA  recognizing $L _ { 7 }$ , after reading $u = 2 \cdot 3 \cdot 6 .$ , must store the last value 6 in a register, since $\mathcal { A }$ must check whether the next input value is greater than 6. Thus, 6 is L-memorable in $u ,$ whereas 2 and 3 are not. The formal definition is as follows:

Definition 4. Let L be a language over $( \Sigma , R )$ . A symbol a $\in \Sigma$ is L-memorable in a word $u \in \Sigma ^ { * }$ if there exist a word $u ^ { \prime }$ of the same -type as $u ,$ symbols $a ^ { \prime } , b ^ { \prime } \in \Sigma$ , a word $w \in \Sigma ^ { * }$ and an R-preserving mapping σ with $\sigma ( a ^ { \prime } ) = b ^ { \prime }$ and $\sigma ( d ) = d \ f o r$ all other $d \in u ^ { \prime } \cdot w$ , such that: $( \varLambda ) \ u \cdot a \sim _ { R } u ^ { \prime } \cdot a ^ { \prime } \sim _ { R } \sigma ( u ^ { \prime } \cdot a ^ { \prime } )$ ; and (2) w is an L-distinguishing extension for u<sup>′</sup> and $\sigma ( u ^ { \prime } )$

Intuitively, only memorable symbols determine the membership of future extended words. If Σ is dense, we can let $u ^ { \prime } = u$ , and replace a in u with a nearby symbol b, obtaining a word $\sigma ( u )$ with $\sigma ( u ) \sim _ { R }$ u and check whether this change afects the membership of its extensions in $L , { \mathrm { i . e . } }$ , there is an L-distinguishing extension w for u and $\sigma ( u )$ . In the non-dense setting, such a replacement b may not exist. We therefore need to construct a word $u ^ { \prime }$ of the same type whose data values have suficient space to allow a valid replacement $b ^ { \prime }$ and then test on $u ^ { \prime }$

We write $m e m _ { L } ( u ) = a _ { 1 } \ldots a _ { k }$ for the sequence of L-memorable symbols in $u ,$ ordered so that for every $a _ { i } , a _ { j } , j > i$ if the last occurrence of $a _ { j }$ in u occurs after the last occurrence of $a _ { i } .$ Let $\mathcal { A }$ be a DRA recognizing $L . \operatorname { I f } \left( q _ { 0 } , \varepsilon \right) \stackrel { u } { \to } _ { A } \left( q , v \right)$

for some configuration $( q , v )$ , then by definition, $\begin{array} { r } { m e m _ { L } ( u ) } \end{array}$ is a subsequence of v [7]. We are now ready to define the Nerode congruence for data languages.

Definition 5. (cf. [7]) Given a data language L over $( \Sigma , R )$ , we define an equivalence relation $\cong _ { L }$ on $\varSigma ^ { * }$ , where for every $u , v \in \Sigma ^ { * }$ , we write $\boldsymbol { u } \cong _ { L } \boldsymbol { v } \ i f ;$

$- \ m e m _ { L } ( u ) \sim _ { R }$ mem<sub>L</sub>(v), and

$$
\begin{array} { r l } & { - \ f o r \ a l l \ x , y \in \Sigma ^ { * } , \ i f m e m _ { L } ( u ) \cdot x \sim _ { R } m e m _ { L } ( v ) \cdot y , \ t h e n \ u \cdot x \in L \Leftrightarrow v \cdot y \in L . } \end{array}
$$

Intuitively, equivalent words u and v must have memorable words of the same type. Furthermore, when their memorable-word extensions share the same type, extending u and v by the corresponding sufixes yields the same membership in $L ,$ since only memorable words afect future membership.

Let L be a data language over $( \Sigma , R )$ and $k = \operatorname* { m a x } \left\{ | m e m _ { L } ( u ) \mid u \in \Sigma ^ { * } \right\}$ Let $[ u ] _ { \cong L }$ denote the equivalence class defined by $\cong _ { L }$ where $u \in \Sigma ^ { * }$ belongs to. We define the canonical k-RA $\mathcal { C } _ { L } = ( Q , q _ { 0 } , F , \varDelta )$ for L as follows:

$\textstyle -  Q = \bigcup _ { i = 0 } ^ { k } Q _ { i }$ where $Q _ { i } = \{ [ u ] \cong _ { L } \subseteq \Sigma ^ { * } \mid u \in \Sigma ^ { * } , | m e m _ { L } ( u ) | = i \}$

$- \ q _ { 0 } = [ \varepsilon ] _ { \cong _ { L } } , F = \{ [ u ] _ { \cong _ { L } } \in Q \ | \ u \in L \}$ , and

$- \ ( [ u ] _ { \cong _ { L } } , \tau , E , [ u ^ { \prime } ] _ { \cong _ { L } } ) \in \varDelta$ if there exists a symbol $a \in \Sigma$ such that:

$u \cdot a \cong _ { L } u ^ { \prime } .$ mem<sub>L</sub>(u)  a has type τ , and

E is the set of indices i in the sequence $\ a _ { 1 } \dots a _ { d } = m e m _ { L } ( u ) \cdot a$ such that either $a _ { i } \notin$ mem<sub>L</sub>(u  a) or $a _ { i } = a$ for $i < d$

In general, $\mathcal { C } _ { L }$ may have infinitely many states or registers and may be nondeterministic. When $L$ is DRA-recognizable, $\mathcal { C } _ { L }$ is exactly the minimal DRA for L. This Myhill–Nerode theorem for DRAs was established in [7]:

Theorem 6. (cf. [7]) Let L be a data language. Then L is recognized by a DRA $i f \cong _ { L }$ has finite index. Moreover, every DRA-recognizable language L has a canonical automaton $\mathcal { C } _ { L }$ which is the unique minimal DRA up to isomorphism.

With Theorem $^ { 6 , }$ the standard DFA minimization algorithm extends to DRAs. Given a DRA , we merge states p and $q$ whenever the residual languages of the configurations $( p , u )$ and $( q , v )$ , for some $u , v ,$ coincide. The existence of such u and v also ensures that $p$ and $q$ are reachable from the initial configuration, with u and v serving as witnesses. Checking this language equivalence reduces to solving the configuration reachability problem in the product automaton. Consequently, the minimization algorithm<sup>5</sup> depends on solving the configuration reachability problem, which is interesting in its own right: Given a configuration $( p , u )$ and a state $q ,$ determine whether there exists a word $w _ { q }$ such that a run from $( p , u )$ reaches some configuration $( q , w )$ on input word $w _ { q }$

As shown in $[ 7 ] ;$ the configuration reachability problem is trivially decidable for dense domains. Given a configuration $( p , u )$ and a transition $( p , \tau , E , q )$ , the density property ensures that there always exists a value a such that $u \cdot a$ has type τ. Hence, with a complete DRA, checking whether a configuration $( p , u )$ can reach a state q reduces to deciding whether there exists a sequence of transitions from $( p , u )$ that leads to the state q. This can be checked easily.

The decidability of the configuration reachability problem in the non-dense setting, however, remains open since the above approach fails. Consider the DRA $A _ { m i d }$ over $( \mathbb { Z } , < )$ in Fig. 1: we have $( q _ { 2 } , u ) \stackrel { a } {  } ( q _ { 3 } , \varepsilon )$ when $u = 1 { \cdot } 5$ and $a = 3$ , but for $u = 1 \cdot 2$ , no value of a produces such a run. To address this gap, we provide the first decidable complexity result of this problem over non-dense domains.

Theorem 7. The configuration reachability problem for DRAs over $( \mathbb { Z } , < )$ is decidable.

As an immediate result of Theorem 7, we obtain the following:

Corollary 8. The minimization problem for DRAs over $( \mathbb { Z } , < )$ is decidable.

In fact, a simpler route to achieve Corollary 8 is available. The next theorem shows that minimization and language equivalence for DRAs over both dense and non-dense domains fit into a single unified framework. That is, we can reduce these problems over non-dense domains to the problems over dense domains which are already known to be decidable [7]. However, this unification does not extend to the configuration reachability problem.

Theorem 9. Let  be a DRA, and let $L _ { \mathbb { Q } }$ and $L _ { \mathbb { Z } }$ be the data languages recognized by over $( \mathbb { Q } , < )$ and $( \mathbb { Z } , < )$ , respectively. For any $w \in \mathbb { Q } ^ { * }$ and $w ^ { \prime } \in \mathbb { Z } ^ { * }$ with w $\sim _ { < } \ w ^ { \prime } { } _ { ; }$ , it holds that $w \in L _ { \mathbb { Q } }$ if $w ^ { \prime } \in L _ { \mathbb { Z } }$ . Furthermore, a DRA over $( \mathbb { Q } , < )$ is the minimal DRA recognizing $L _ { \mathbb { Q } }$ if the same DRA over $( \mathbb { Z } , < )$ is the minimal DRA recognizing $L _ { \mathbb { Z } }$ . This correspondence remains valid when either $\mathbb { Q }$ or $\mathbb { Z }$ is replaced by R.

Proof. Given a DRA , we use $\mathcal { A } _ { \mathbb { Q } }$ and $\mathcal { A } _ { \mathbb { Z } }$ to represent the DRAs over $( \mathbb { Q } , < )$ and $( \mathbb { Z } , < )$ , respectively. For every w $\in \mathbb { Q } ^ { * }$ , there exists $w ^ { \prime } \in \mathbb { Z } ^ { * }$ s.t. $w ^ { \prime } \sim _ { < } w$ and vice versa. Therefore, $L ( A _ { \mathbb { Z } } ) = L ( A _ { \mathbb { Q } } ) \cap \mathbb { Z } ^ { * }$ and $L ( A _ { \mathbb { Q } } ) = \{ w | \exists w ^ { \prime } \in$ $L ( \mathcal { A } _ { \mathbb { Z } } ) \wedge w ^ { \prime } \sim _ { < } w \big \}$ . This implies that $\scriptstyle A _ { \mathbb { Q } }$ is the minimal DRA recognizing $L ( \mathcal { A } _ { \mathbb { Q } } )$ if and only if $\mathcal { A } _ { \mathbb { Z } }$ is the the minimal DRA recognizing $L ( \mathcal { A } _ { \mathbb { Z } } )$ . The argument remains valid if the ordered domain $\mathbb { Q }$ or $\mathbb { Z }$ is replaced by R. □

## 4 Active Learning of DRAs

In this section, we present an algorithm for learning a DRA of an unknown data language L using polynomially many queries in the active learning framework. In this framework, a learner aims to infer an automaton recognizing L, while a teacher knows L and answers membership queries $\mathsf { M Q } ( u )$ (checking whether $u \in L )$ and equivalence queries $\mathsf { E Q } ( \mathcal { H } )$ (checking whether  recognizes L).

The learner begins by issuing membership queries to populate an observation table $\tau _ { \ast }$ from which it constructs a conjectured automaton . An equivalence query then checks its correctness: if the teacher replies ${ } ^ { 6 6 } \mathrm { Y e s } ^ { 5 }$ , learning terminates;

![](images/c8557c29a1f135f3a4074b6971d1f48dcedf7085dbb991c08bed6e17acc99d70.jpg)  
Fig. 2: The example DRA where the missing transitions are labeled with , and the observation table where the extension set X are omitted.

otherwise a counterexample $w \in L \ominus L ( \mathcal { H } )$ is returned and used to refine . This process repeats until a correct automaton is found. While originally developed for DFAs, we adapt this framework to DRAs over ordered domains. By Theorem $^ { 9 , }$ it sufices to learn DRAs over $( \mathbb { Q } , < )$ , since the case of $( \mathbb { Z } , < )$ reduces to it; See Section C for more details on the reduction.

Unlike DFA learning, data languages lack known polynomial-time learning algorithms with respect to the number of queries, even for only equality tests on data [23]. This barrier can be overcome by using also memorability queries [6], Mem(u), which return the memorable word $\begin{array} { r } { m e m _ { L } ( u ) } \end{array}$ of a word u.

We first present a DRA learner using membership, equivalence, and memorability queries, and then give improved complexity results for some decision problems of DRAs that are closely related to solving queries.

## 4.1 Learning DRAs with Memorability Queries

We fix a target DRA-recognizable language L over $( \Sigma , R ) = ( \mathbb { Q } , < )$ in the whole section. The learner is assumed to know $( \Sigma , R )$ and maintains an observation table to record its knowledge of L.

Definition 10 (Observation Table). An observation table for a DRA over $( \Sigma , R )$ is a tuple $( U , X , S , M , f )$ , where $U , X$ , and S are finite sets of data words: U is the set of prefixes, X the extensions, and S the sufixes. The function $M : U \cup X \to \Sigma ^ { * }$ records the memorable word of each word $u \in U \cup X$ $i . e . , M ( u ) = m e m _ { L } ( u )$ . The classification function $f : U \times S \to \{ + , - \}$ labels each u w as in L or not, i.e, $f ( u , w ) = M Q ( u \cdot w )$ . We require $( 1 ) U \cup X$ to be prefix-closed and $( { \mathcal { Q } } ) \varepsilon \in U$ and $\varepsilon \in S$

Table in Fig. 2 shows a final observation table for $L _ { r e p } = \{ x y x y \in \mathbb { Q } ^ { 4 } \mid x < y \}$ The rows correspond to the prefix set $U = \{ \varepsilon , 0 , 0 \cdot \bar { 1 } , 0 \cdot 1 \cdot 0 , 0 \cdot 1 \cdot 0 \cdot 1 , 0 \cdot - 1 \}$ and the columns contain the memorable function M and the sufix set $S =$ $\{ \varepsilon , 0 , 0 \cdot 1 \cdot 0 \cdot 1 \}$ . The set X is omitted here, as it is only used internally when constructing conjectured DRAs. From this table, we can obtain the minimal DRA  (with respect to $\cong _ { L } )$ , also shown in Fig. 2. Each row $u \in U$ naturally corresponds to a state of : u serves as the representative of its $\cong _ { L }$ -equivalence class, and reaches the associated state exactly after reading u. For instance, the empty word ε is the initial state, whereas 0 1 0 1 is the only final state.

In the remainder of the paper, we abusively use the word representatives in U as state names. While constructing the observation table, as in the classic $L ^ { * }$ algorithm for DFAs, we ensure that for every state $u \in U$ , each one-letter extension $u \cdot b$ appears in the table, i.e., in X. To compute the one-letter extensions of u $( \mathrm { i . e . , } u \cdot b$ for $b \in \Sigma )$ , we observe that (i) only the memorable word $M ( u )$ is relevant, since the types of future words depend solely on its memorable letters; and (ii) we do not need to enumerate all $b \in \Sigma$ , but only a finite set that covers all possible types of $M ( u ) \cdot b .$

Let $a _ { 1 } < \cdots < a _ { d }$ be the ordered sequence of distinct letters in $M ( u )$ . Then b can take one of the four types of values:

1. $b \in M ( u )$ , i.e., it means that b occurs also in $M ( u )$

2. $b = a _ { 1 } - 1 , \mathrm { i . e . }$ , we have that $b < a _ { i }$ for all $1 \leq i \leq d ;$

3. $b = a _ { d } + 1 , \mathrm { i . e . }$ , it gives that $b > a _ { i }$ for all $1 \leq i \leq d ;$ and finally

4. $b = ( a _ { i } + a _ { i + 1 } ) / 2$ for some $1 \leq i < d , \mathrm { i . e . , } a _ { j } < b < a _ { k }$ for all $j \le i$ and $k \geq i + 1$

Note that the fourth type of values will be ignored $\mathrm { i f } \ | M ( u ) | = 1$ and we set $b = 0$ when $u = \varepsilon$ . If R is the identity test, it sufices to add values $b \in M ( u )$ or a fresh letter $b \notin M ( u )$ . For example, if $M ( u ) = 0 \cdot 1$ , then $b \in \{ 0 , 1 , 2 \}$

To ensure that U matches the state set of the target DRA, we must determine when two words $u _ { 1 } , u _ { 2 } \in U$ are nonequivalent under the Nerode congruence $\cong _ { L }$ (Definition 5). Recall that $u _ { 1 } \not \equiv _ { L }$ u holds precisely when (i) their memorable word types difer, i.e., mem $\mathbf { \Omega } _ { L } ( u _ { 1 } ) \mathbf { \Omega } \not \sim \mathbf { \Omega } _ { R }$ mem<sub>L</sub>(u<sub>2</sub>), or (ii) there exist $x , y \in \Sigma ^ { * }$ such that mem $\mathbf { \Omega } _ { L } ( u _ { 1 } ) \cdot x \sim _ { R }$ mem $\mathbf { \Omega } _ { L } ( u _ { 2 } ) \cdot y$ but $u _ { 1 } \cdot x \in L \not \circledast u _ { 2 } \cdot y \in L$ . While testing whether two words have the same type is straightforward, finding witness words x and y is more challenging. In contrast, the classical $L ^ { * }$ algorithm for DFAs [2] needs only a single witness to distinguish two words.

Fortunately, since Σ is dense, it sufices to find an order-preserving mapping σ with $\sigma ( m e m _ { L } ( u _ { 1 } ) ) = m e m _ { L } ( u _ { 2 } )$ and a single L-distinguishing sufix w that separates $\sigma ( u _ { 1 } )$ from u . The following lemma justifies this reduction:

Lemma 11. Let Σ be a dense set. For two words $u _ { 1 } , u _ { 2 } \in \Sigma ^ { * }$ with $m e m _ { L } ( u _ { 1 } ) \sim _ { R }$ mem $\mathsf { \Omega } _ { \mathsf { C } } ( u _ { 2 } ) , u _ { 1 } \notin \mathsf { \Gamma } _ { L }$ u if and only if there exist a bijective orderpreserving mapping σ and a word w such that $\sigma ( m e m _ { L } ( u _ { 1 } ) ) = m e m _ { L } ( u _ { 2 } )$ and $\sigma ( u _ { 1 } ) \cdot w \in L \not \circledast u _ { 2 } \cdot w \in L$

A further challenge is that infinitely many order-preserving mappings σ may exist, making enumeration impossible. Nevertheless, we prove that it sufices to find one suitable mapping. Notably, such a bijective order-preserving mapping σ with $\sigma ( m e m _ { L } ( u _ { 1 } ) ) = m e m _ { L } ( u _ { 2 } )$ can be obtained easily, as described in Sect. 2.

Lemma 12. Let Σ be a dense set, and let $u _ { 1 } , u _ { 2 }$ be distinct words. $I f \sigma$ is a bijective order-preserving mapping with $\sigma ( m e m _ { L } ( u _ { 1 } ) ) = m e m _ { L } ( u _ { 2 } )$ , then $u _ { 1 } \not \cong L$ u if there exists $w \in \Sigma ^ { * }$ such that $\sigma ( u _ { 1 } ) \cdot w \in L \not \circledast u _ { 2 } \cdot w \in L$

![](images/5cd559ef4ae8a5c76b4e48e35ac2da99d76f37352cf43ccc5a96a19e3ec89e75.jpg)  
Fig. 3: Unclosed table ${ \mathcal { T } } _ { 0 } .$ closed table $\mathcal { T } _ { 1 }$ and its induced DRAs $\mathcal { H } _ { 1 }$

To derive a conjectured DRA from the observation table $\tau _ { \ast }$ , the table must be closed. We introduce the notion of S-consistency: a word $v \in X$ is said to be S-consistent with a prefix $u \in U$ if $M ( v ) \sim _ { R } M ( u )$ and, for every $w \in S$ we have $\mathsf { M } \mathsf { Q } ( \sigma ( v ) \cdot w ) = f ( u , w )$ , where σ is an order-preserving mapping on $\Sigma \ \mathrm { s . t . } \ \sigma ( M ( v ) ) = M ( u )$ . By Lemma 12, the table $\tau$ is said to be closed if, for each extension ua $\in { X }$ of a word $u \in U$ , there exists a prefix $u ^ { \prime } \in U$ s.t. ua is S-consistent to $u ^ { \prime } .$ . Intuitively, the target DRA moves to state $u ^ { \prime }$ at state u when reading $a \in \Sigma$ if ua is S-consistent with $u ^ { \prime } .$ . Note that S-consistency is not an equivalence relation, as it does not need to be symmetric nor transitive.

Our algorithm relies on two key properties of S-consistency: (i) if u is not S-consistent with $v ,$ then u $\cong _ { L } \ v ; \ ( \mathrm { i i } )$ if u is not S-consistent with $v ,$ then u is not $S ^ { \prime } .$ -consistent with v whenever $S \subseteq S ^ { \prime }$

If is not closed, the learner invokes CloseTable( , MQ, Mem), which repeatedly: (i) finds a word $u a \in X$ with $u \in U$ that has no S-consistent $u ^ { \prime } \in U ;$ (ii) adds ua to $U$ and adds also all its one-letter extensions uab to $X ; ~ ( \mathrm { i i i } )$ fills the new row labeled with ua by setting $f ( u a , w ) = \mathsf { M Q } ( u a \cdot w )$ for all $w \in S$ and updates $M ( u a b ) = \mathsf { M e m } ( u a b )$ for all new extensions $u a b \in X$

Consider the initial table $\mathcal { T } _ { 0 }$ in Fig. 3 for learning $L = \{ x y x y \in \mathbb { Q } ^ { 4 } \mid x < y \}$ with $X = \{ 0 \}$ and $M ( 0 ) = 0$ . To close the table, the learner checks whether $\varepsilon \cdot 0$ is S-consistent with ε. Since $M ( \varepsilon \cdot 0 ) = 0$ difers in word type from $M ( \varepsilon ) = \varepsilon ,$ the check fails, and 0 is added to U. We update $f$ with $f ( 0 , \varepsilon ) = \mathsf { M Q } ( 0 \cdot \varepsilon ) = -$ for the new row 0 and column ε. Further, the one-letter extensions of 0 are $\{ 0 \cdot 0 , 0 \cdot - 1 , 0 \cdot 1 \}$ , so X becomes $\{ 0 , 0 \cdot 0 , 0 \cdotp { - } 1 , 0 \cdotp { - } 1 \}$ . Using memorability queries, we also update M as $M ( 0 \cdot 0 ) = \varepsilon , M ( 0 \cdot - 1 ) = \varepsilon , M ( 0 \cdot 1 ) = 0 \cdot 1$ . Continuing in this way, the learner adds 0 1 to U and obtains the closed table $\mathcal { T } _ { 1 }$ in Fig. 3.

With all extensions computed for each new state, the conjectured DRA we construct will be complete. Once $\tau$ is closed, we can induce using Definition 13.

Definition 13 (DRA Construction). Let $\mathcal { T } = ( U , X , S , M , f )$ be a closed observation table and k = max $\{ | M ( u ) | \mid u \in U \}$ . We construct a $k { \cdot } D R A \ \mathcal { H } =$ $( Q , q _ { 0 } , F , \varDelta )$ from as follows:

$\textstyle Q \ = \bigcup _ { i = 0 } ^ { k } Q _ { i }$ where $Q _ { i } \ = \ \{ u \in U \mid | M e m ( u ) | = i \} , \ q _ { 0 } \ = \ \varepsilon \ \in \ Q _ { 0 } , \ F \ =$ $\{ u \in U \mid f ( u , \varepsilon ) = + \}$ , and

$- \ ( u , \tau , E , u ^ { \prime } ) \in \varDelta \ i f$ there is an extension $u { \cdot } a \in X \ s . t . \ ( 1 ) \ u ^ { \prime } \in U$ is the first found word with which u a is S-consistent, $\quad ( 2 ) ~ \tau = M ( u ) \cdot a = a _ { 1 } \cdot \cdot \cdot a _ { d } $ , and E contains indices i where $a _ { i } \notin M ( u \cdot a ) \ o r \ a _ { i } = a _ { d } = a$ with $i < d ,$

```postscript
Algorithm 1: Function cexAnalyze $( \mathcal { T } , \mathcal { H } , w , \mathsf { M Q } ( \cdot ) )$ that analyses w and
returns a sufix v. The prerequisite is that w $\in L \odot L ( { \mathcal { H } } )$
1 $u \gets \varepsilon , r \gets \varepsilon , m q \gets \mathsf { M Q } ( w )$
2 while True do
3 Let $a \cdot z : = w$ where $a \in \Sigma ; ~ / / w = \varepsilon$ is impossible;
4 $/ / \mathrm { I N V A R I A N T } \colon \ r = M ( \boldsymbol { u } ) ;$
5 Let $( u , r ) \xrightarrow { M ( u ) \cdot b : E } ( u ^ { \prime } , r ^ { \prime } )$ be the transition taken in when reading a;
6 $/ / \mathrm { I N V A R I A N T } \colon \ r \cdot a = M ( u ) \cdot a \sim _ { R } M ( u ) \cdot b , \ M ( u ^ { \prime } ) \sim _ { R } M ( u b ) = r ^ { \prime } ;$
7 Let $\sigma _ { 1 }$ be an order-preserving mapping from $M ( u ) \cdot a$ to $M ( u ) \cdot b ;$
8 $/ / { \tt S i }$ nce $r = M ( u ) , \ r ^ { \prime } = M ( u \cdot b )$ is obtained by removing letters
in positions E from $r \cdot b ;$
9 Let $\sigma _ { 2 }$ be an order-preserving mapping from $M ( \boldsymbol { u } \cdot \boldsymbol { b } )$ to $M ( u ^ { \prime } )$
10 if $M Q ( u ^ { \prime } { \cdot } \sigma _ { 2 } ( \sigma _ { 1 } ( z ) ) ) \ne $ mq then
11 return new sufix $\sigma _ { 2 } ( \sigma _ { 1 } ( z ) ) ;$
12 end
13 $u  u ^ { \prime } , r  M ( u ^ { \prime } ) , w  \sigma _ { 2 } ( \sigma _ { 1 } ( z ) ) ; \ / / { \mathsf { M Q } } ( u ^ { \prime } \cdot \sigma _ { 2 } ( \sigma _ { 1 } ( z ) ) ) = m q$
14 end
```

## It immediately follows from Theorem 6 that:

Lemma 14. If U contains exactly the set of word representatives defined $b y \cong _ { L }$ then is a minimal DRA recognizing the target language L.

Consider constructing the DRA $\mathcal { H } _ { 1 }$ from the closed table $\mathcal { T } _ { 1 }$ in Fig. 3. Since the maximal length of a memorable word is $| 0 \cdot 1 | = 2 , \mathcal { H } _ { 1 }$ is a 2-DRA. The set of states is $Q = \{ \varepsilon \} \not  \{ 0 \} \not  \{ 0 \cdot 1 \}$ , with $q _ { 0 } = \varepsilon$ , and $F = \emptyset$ because no $u \in U$ satisfies $f ( u , \varepsilon ) = +$ . Here, we focus on the transition from 0 to 0 1 in $\varDelta$ . The one-letter extension 0 1 of state 0 is S-consistent with 0 1, so reading letter 1 at 0 leads to state 0 1. We then set $\tau = { \cal M } ( 0 ) \cdot 1 = 0 \cdot 1$ and $E = \emptyset$ , since both 0 and 1 belong to $M ( 0 \cdot 1 )$ and no letters need to be removed.

Once the conjectured DRA is constructed, the learner poses an equivalence query EQ( ) to the teacher. If the answer is positive, is returned as the result. Otherwise, the teacher provides a counterexample $w \in L ( \mathcal { H } ) \ominus L$ . The learner analyzes w to update the observation table, allowing it to correct by adding new states or redirecting transitions. The counterexample analysis procedure is described in Algorithm 1. An example of counterexample analysis can be found in Section B.

In Algorithm 1, the counterexample w is processed letter by letter. In each iteration, we assume the current configuration is $( u , r )$ and $\mathsf { M } \mathsf { Q } ( u \cdot a z ) = m q .$ where $w = a \cdot z .$ . Reading letter a at state u, transitions to $u ^ { \prime }$ . However, this transition may have been constructed using a diferent letter b such that ub is S-consistent with $u ^ { \prime }$ and $a \neq b ,$ , where $M ( u ) \cdot a \sim _ { R } M ( u ) \cdot b .$

Our goal is to identify an extension ub wrongly classified as equivalent to $u ^ { \prime } .$ We first normalize the remaining sufix z for ub via the mapping $\sigma _ { 1 }$ from $M ( u ) \cdot a$ to $M ( u ) \cdot b ,$ , preserving membership results: ${ \mathsf { M Q } } ( u b \cdot \sigma _ { 1 } ( z ) ) = m q = { \mathsf { M Q } } ( u a \cdot z )$ 2 since both runs visit the same state sequence in the target DRA of L. To verify whether $u b \cong _ { L } u ^ { \prime } .$ we further normalize $\sigma _ { 1 } ( z )$ for $u ^ { \prime }$ using $\sigma _ { 2 }$ from $M ( u b )$ to $M ( u ^ { \prime } )$ , yielding $v = \sigma _ { 2 } ( \sigma _ { 1 } ( z ) )$ . If $\mathsf { M Q } ( u ^ { \prime } \cdot v )$ = mq, then $u ^ { \prime } \ncong _ { L } u b .$ , according to Lemma 12 since $\sigma _ { 2 } ( u b \cdot \sigma _ { 1 } ( z ) ) \sim _ { R } \sigma _ { 2 } ( u b ) \cdot v$ , giving $\mathsf { M } \mathsf { Q } ( \sigma _ { 2 } ( u b ) \cdot v ) = m q .$ Otherwise, we proceed to the next iteration with $u  u ^ { \prime } , r  M ( u ^ { \prime } )$ , and w $\ulcorner  v$ Since $w \in L \ominus L ( \mathcal { H } )$ , before reaching the last state $u ^ { \prime \prime }$ where $\mathsf { M } \mathsf { Q } ( u ^ { \prime \prime } \cdot \varepsilon ) \neq m q .$ the membership results must change at some point. Hence, the if condition will eventually be satisfied, and a sufix v will be returned.

Algorithm 2: The DRA learner   
1 Initialize table $\mathcal { T } = ( U , X , S , M , f )$ where $U = S = \{ \varepsilon \} , X = \{ 0 \} , M ( \varepsilon ) = \varepsilon ,$   
$M ( 0 ) = { \mathsf { M e m } } ( 0 ) ,$ and $f ( \varepsilon , \varepsilon ) = \mathsf { M } \mathsf { Q } ( \varepsilon ) ;$   
2 CloseTable( , MQ( ), Mem( )) and let be the DRA constructed from $\tau ;$   
3 Let (res, w) be the response from the teacher on $\mathsf { E Q } ( \mathcal { H } ) ;$   
4 while $r e s = N o$ do   
5 $v  \mathsf { c e x A n a l y z e } ( T , \mathcal { H } , w , \mathsf { M Q } ( \cdot ) ) ;$   
6 Add v to the sufix set S in and set $f ( u , v ) = \mathsf { M Q } ( u \cdot v )$ for all $u \in U ;$   
7 CloseTable( , MQ( ), Mem( )) and let be the DRA constructed from $\tau { ; }$   
8 Let (res, w) be the response from the teacher on $\mathsf { E Q } ( \mathcal { H } ) ;$   
9 end   
10 return $\mathcal { H } ;$

The correctness of Algorithm 1 is guaranteed by the following lemma.

Lemma 15. Let w be the counterexample provided by the teacher. Algorithm 1 terminates after at most w iterations and returns a sufix v such that ub $\in X$ is not S v -consistent with a word $u ^ { \prime } \in U$ with which ub is currently S-consistent.

We now formally define the learner in Algorithm 2. To initialize the observation table , the learner inserts the empty word ε into U. The extension set is initialized as $X = \{ 0 \}$ , since 0 has the same type as all one-letter words. It fills entries via membership queries ${ \mathsf { M Q } } ( \cdot )$ and memorability queries Mem( ), so $M ( \varepsilon ) = \varepsilon$ and $M ( 0 ) = \mathsf { M e m } ( 0 )$ . The function CloseTable closes $\tau$ with the help of ${ \mathsf { M Q } } ( \cdot )$ and Mem( ). Conjectured DRAs are then derived from $\tau$ following Definition 13. On receiving a counterexample w, cexAnalyze (Algorithm 1) returns a sufix v that is added to S. New entries are filled via MQ( ). The refinement loop of terminates once the teacher returns a positive answer.

The soundness and completeness of Algorithm 2 directly follow from Lemmas 12, 14 and 15.

Theorem 16. Let  be a complete canonical minimal DRA of L with n states and m transitions defined under $\cong _ { L }$ , and let d be the maximum length of any counterexample returned by the teacher. It holds that:

– Algorithm 2 terminates and returns a DRA  that is isomorphic to ${ \mathcal { A } } .$

Algorithm 2 terminates after at most m $\times \left( n - 1 \right)$ equivalence queries.

– Algorithm 2 requires m memorability queries and $\mathcal { O } ( d \times m \times n + m ^ { 3 } \times n ^ { 3 } )$ membership queries.

The returned DRA recognizes $L ,$ as confirmed by the teacher. Algorithm 2 terminates because each counterexample either corrects a transition or adds a new state (Lemma 15), and the number of equivalence classes under $\cong _ { L }$ and transitions in target DRA is finite. Now we show that our algorithm runs in polynomial time in the number of queries.

In DFA learning, two words $u _ { 1 }$ and $u _ { 2 }$ are considered equivalent with respect to a set of sufixes S if $\mathsf { M } \mathsf { Q } ( u _ { 1 } w ) = \mathsf { M } \mathsf { Q } ( u _ { 2 } w )$ for all $w \in S$ . Thus, when a transition ub is misclassified as consistent with some $u ^ { \prime }$ , it can safely be added to $U .$ . In our setting, however, S-consistency does not guarantee this property, as it is not an equivalence relation. When Algorithm 1 returns a distinguishing sufix v separating ub from the state $u ^ { \prime }$ with which ub is currently S-consistent, we add v to the sufix set $S$ to correct the transition from u over b. Consequently, ub becomes not $S  \{ v \}$ -consistent with $u ^ { \prime }$ . Each transition can be corrected at most $n - 1$ times; that is, the target state of a transition over b from u may be misclassified no more than $n - 1$ times, as once ub is inconsistent with a state $u ^ { \prime \prime }$ , it remains inconsistent with $u ^ { \prime \prime }$ thereafter. Hence, the learner asks at most $m \cdot ( n - 1 )$ equivalence queries in total. In practice, new states are typically discovered immediately after adding the sufix returned by the counterexample analysis. The reasoning for membership and memorability queries can be found in Section D.

## 4.2 Complexity of Decision Problems

This section studies the complexity of some key decision problems for DRAs over ordered data domains. Understanding these complexities is essential, as they closely correspond to the queries required by our active learning algorithm, particularly when the teacher has the target data language in the form of a DRA. All results in this section apply to both well-typed and non-well-typed DRAs.

Let $\mathcal { A } _ { 1 } = ( Q ^ { 1 } , q _ { 0 } ^ { 1 } , F ^ { 1 } , \varDelta ^ { 1 } )$ be a $k _ { \mathrm { 1 } } \mathrm { - D R A }$ and $\mathcal { A } _ { 2 } = ( Q ^ { 2 } , q _ { 0 } ^ { 2 } , F ^ { 2 } , \varDelta ^ { 2 } )$ be a $k _ { 2 ^ { - } }$ DRA, respectively. The language intersection non-emptiness asks whether there is a word $w \in L ( \mathcal { A } _ { 1 } ) \cap L ( \mathcal { A } _ { 2 } )$ . The language inclusion problem asks whether $L ( \mathcal { A } _ { 1 } ) \subseteq L ( \mathcal { A } _ { 2 } )$ , equivalently whether $L ( \mathcal { A } _ { 1 } ) \cap \overline { { L ( \mathcal { A } _ { 2 } ) } } = \emptyset$ , where $\overline { { L ( \mathcal { A } _ { 2 } ) } }$ is obtained by complementing $\boldsymbol { A } _ { 2 }$ via swapping accepting and non-accepting states since $\boldsymbol { A } _ { 2 }$ is deterministic.

Theorem 17. For DRAs over $( \Sigma , = )$ and over $( \Sigma , < )$ , the language intersection non-emptiness problem is PSPACE-complete. Consequently, the language inclusion problem is also PSPACE-complete.

For the PSPACE upper bound, intersection non-emptiness is reduced to reachability in the product of the two DRAs, where it sufices to distinguish register valuations by their relative data types. If an accepting run exists, there is one of length at most $| Q ^ { 1 } | \cdot | Q ^ { 2 } | \cdot ( k _ { 1 } + k _ { 2 } ) !$ , so it can be guessed symbol by symbol while storing only the current product configuration and a counter. PSPACE-hardness already holds for DRAs with identity tests, via a reduction from linear-bounded deterministic Turing machines.

Note that by Theorem 9, the above result holds for both dense and non-dense domains.

Given a DRA over $( \Sigma , R )$ and a configuration $( q , v )$ of ${ \mathcal { A } } ,$ we define the language of $\mathcal { A } ^ { ( q , v ) }$ as $L ( \mathcal { A } ^ { ( q , v ) } ) ~ = ~ \{ w ~ \in ~ \Sigma ^ { * } ~ | ~ \exists q _ { f } ~ \in ~ F , u ~ \in ~ \Sigma ^ { * } . ~ ( q , v ) ~ \overset { w } { \longrightarrow } _ { A }$ $\left( q _ { f } , u \right) ]$ . Let $( q _ { 1 } , v _ { 1 } )$ and $( q _ { 2 } , v _ { 2 } )$ be two configurations of $\boldsymbol { A } _ { 1 }$ and $\boldsymbol { A } _ { 2 }$ respectively. The configuration equivalence problem asks whether $L ( \mathcal { A } _ { 1 } ^ { ( q _ { 1 } , v _ { 1 } ) } ) = L ( \mathcal { A } _ { 2 } ^ { ( q _ { 2 } , v _ { 2 } ) } )$ . The language equivalence problem asks whether $L ( \mathcal { A } _ { 1 } ) = L ( \mathcal { A } _ { 2 } )$ . It is a special case of the configuration equivalence problem, that ${ \mathrm { i s } } ,$ it is equivalent to the configuration equivalence problem whether $L ( \mathcal { A } _ { 1 } ^ { ( q _ { 0 } ^ { 1 } , \varepsilon ) } ) = L ( \mathcal { A } _ { 2 } ^ { ( q _ { 0 } ^ { 2 } , \bar { \varepsilon } ) } )$ where $( q _ { 0 } ^ { 1 } , \varepsilon )$ and $( q _ { 0 } ^ { 2 } , \varepsilon )$ are the initial configurations of $\mathcal { A } _ { 1 }$ and $\boldsymbol { A } _ { 2 }$ respectively.

Theorem 18. Both the configuration equivalence problem and the language equivalence problem are in PSPACE for DRAs over dense domain $( \Sigma , < )$

The configuration equivalence problem is referred to as the residual equivalence problem in [6], and is shown to be in PTIME for well-typed DRAs over unordered (identity) data domains $( \Sigma , = ) ~ [ 6$ , Theorem $^ { 8 ] }$

Given a k-DRA over $( \Sigma , < )$ , a word $u \in \Sigma ^ { * }$ and $a \in u$ , the memorability problem asks whether a is L-memorable in u, that is, whether $a \in m e m _ { L } ( u )$ We show this problem is in PSPACE, which improves upon the NEXPTIME upper-bound for the memorability problem established in [7, Proposition 3].

Theorem 19. The memorability problem is in PSPACE for DRAs over dense domain $( \Sigma , < )$

Over a dense domain, memorability reduces to configuration equivalence: replacing a by a fresh value while preserving the order type yields two configurations that are inequivalent exactly when a is memorable.

Experiments. We implemented the proposed active learning algorithm in Python<sup>6</sup> and evaluated it on the running examples in this paper as well as randomly generated DRAs. The results confirm that the numbers of membership, equivalence, and memorability queries grow polynomially with the size of the target automaton, in line with the theoretical bounds, with membership queries dominating in practice. We also compared our approach with RaLib [12]; RaLib uses tree-based queries with black-box approximate equivalence checking, whereas our method uses word-based queries with white-box exact equivalence checking. The experiments show that both approaches are efective in their respective settings. See Section E and Section F for more details.

## 5 Conclusion and Discussion

In this paper, we studied DRAs over ordered domains. We showed that, with respect to minimization, DRAs over dense and non-dense domains can be treated within a unified framework. This yields decidability of the minimization problem for DRAs over non-dense domains, resolving a previously open question [7].

Building on this unified framework, we developed and implemented an active learning procedure for DRAs that makes use of membership, equivalence, and memorability queries. Our experimental results demonstrate that the learning framework presented in this paper complements existing approaches for DRAs over ordered domains.

A natural question arises: can memorability queries be removed to obtain an algorithm similar to the classic $L ^ { \ast 2 }$ We believe this is possible by using the notion of abstract sufixes introduced in [23]. Instead of treating the sufix v returned by Algorithm 1 as a concrete word, we treat it as an abstract sufix representing all possible sufixes z for a prefix u over a set $D _ { u , v }$ with $z \sim R$ v. Let v contain ℓ distinct letters. For each prefix $u = b _ { 1 } \dots b _ { m } \in U$ , let $b _ { 1 } ^ { \prime } < \cdots < b _ { m } ^ { \prime }$ be the sorted values in u, $b _ { 0 } ^ { \prime } = b _ { 1 } ^ { \prime } - 1$ , and $b _ { m + 1 } ^ { \prime } = b _ { m } ^ { \prime } + 1$ . Then, $D _ { u , v } \ =$ $\textstyle \bigcup _ { i = 0 } ^ { m } \bigcup _ { j = 1 } ^ { \ell + 1 } \{ b _ { i } ^ { \prime } + \frac { j } { \ell + 1 } \times ( b _ { i + 1 } ^ { \prime } - b _ { i } ^ { \prime } ) \}$ . The sufixes z cover all possible extensions of u that have the same word type as v, analogous to our previous computation for the one-letter extensions of the memorable word $M ( u )$ . Since $M ( u )$ is unknown, we have to consider all possible sufixes of the same word type to infer $M ( u )$ instead of a single sufix v from a counterexample.

Checking S-consistency and the corresponding counterexample analysis procedures also become more involved. We leave a full learning algorithm using only membership and equivalence queries as future work. Unsurprisingly, this approach requires an exponential number of membership queries, as also in [23], to fill the observation table since all concrete sufixes z over $D _ { u , v }$ must be enumerated and added to the sufix set for u.

We also note that the register automata underlying RaLib operate on data words over $A \times D ,$ , where A is a finite action alphabet and D is an infinite data domain, whereas our model considers words directly over D. Hence, the two models are not directly equivalent, since the action-labelled model can distinguish inputs carrying the same data value but diferent actions. A natural extension of the model considered here is to $A \times D$ , with word types preserving the finite action sequence while comparisons and renamings apply only to the data components. We conjecture that this action-extended variant has the same expressive power as the corresponding action-labelled register automata. Establishing a formal correspondence between the two models, and determining whether our learning and complexity bounds carry over to this extension, are interesting directions for future work.

Acknowledgements. We would like to thank the anonymous reviewers for their suggestions that helped improve the paper. This work was supported in part by the National Key R&D Program of China (Grant No. 2025YFE0220300), the Beijing Natural Science Foundation Project No. IS26039, CAS Project for Young Scientists in Basic Research (Grant No. YSBR-040), and the Engineering and Physical Sciences Research Council (EPSRC) through grant EP/X042596/1.

This project is part of the European Union’s Horizon Europe programme under the Marie Skłodowska-Curie grant agreement No. 101208673.

## References

1. Alur, R., DAntoni, L., Deshmukh, J., Raghothaman, M., Yuan, Y.: Regular functions and cost register automata. In: LICS. pp. 13–22 (2013)

2. Angluin, D.: Learning regular sets from queries and counterexamples. Inf. Comput. 75(2), 87–106 (1987)

3. Argyros, G., D’Antoni, L.: The learnability of symbolic automata. In: Chockler, H., Weissenbacher, G. (eds.) Computer Aided Verification. pp. 427–445. Springer International Publishing, Cham (2018)

4. Argyros, G., Stais, I., Kiayias, A., Keromytis, A.D.: Back in black: Towards formal, black box analysis of sanitizers and filters. In: S&P. pp. 91–109 (2016)

5. Balachander, M., Filiot, E., Gentilini, R.: Passive learning of regular data languages in polynomial time and data. In: Majumdar, R., Silva, A. (eds.) CONCUR. LIPIcs, vol. 311, pp. 10:1–10:21. Schloss Dagstuhl - Leibniz-Zentrum für Informatik (2024)

6. Balachander, M., Filiot, E., Gentilini, R., Tzevelekos, N.: Register automata with permutations. In: Gawrychowski, P., Mazowiecki, F., Skrzypczak, M. (eds.) MFCS. LIPIcs, vol. 345, pp. 14:1–14:18. Schloss Dagstuhl - Leibniz-Zentrum für Informatik (2025)

7. Benedikt, M., Ley, C., Puppis, G.: Minimal memory automata. Tech. rep. (2010), long version of ‘What You Must Remember When Processing Data Words’.

8. Bojańczyk, M., Klin, B., Lasota, S.: Automata theory in nominal sets. Logical Methods in Computer Science Volume 10, Issue 3, 4 (2014)

9. Bollig, B., Habermehl, P., Leucker, M., Monmege, B.: A fresh approach to learning register automata. In: Béal, M.P., Carton, O. (eds.) DLT. pp. 118–130. Springer Berlin Heidelberg, Berlin, Heidelberg (2013)

10. Cassel, S., Falk, H., Jonsson, B.: Ralib : A learnlib extension for inferring efsms. In: DIFTS (2015)

11. Cassel, S., Howar, F., Jonsson, B., Merten, M., Stefen, B.: A succinct canonical register automaton model. Journal of Logical and Algebraic Methods in Programming 84(1), 54–66 (2015)

12. Cassel, S., Howar, F., Jonsson, B., Stefen, B.: Active learning for extended finite state machines. Formal Aspects Comput. 28(2), 233–263 (2016)

13. Chen, Y., Hong, C., Lin, A.W., Rümmer, P.: Learning to prove safety over parameterised concurrent systems. In: Stewart, D., Weissenbacher, G. (eds.) FMCAD. pp. 76–83. IEEE (2017)

14. Chen, Y.F., Lengal, O., Tan, T., Wu, Z.: Register automata with linear arithmetic . In: LICS. pp. 1–12. IEEE Computer Society, Los Alamitos, CA, USA (Jun 2017)

15. Cobleigh, J.M., Giannakopoulou, D., Pasareanu, C.S.: Learning assumptions for compositional verification. In: Garavel, H., Hatclif, J. (eds.) TACAS. LNCS, vol. 2619, pp. 331–346. Springer (2003)

16. Czerny, M.X.: Learning-based software testing: Evaluation of Angluin’s L<sup>∗</sup> algorithm and adaptations in practice (2014)

17. D’Antoni, L., Veanes, M.: The power of symbolic automata and transducers. In: Majumdar, R., Kunčak, V. (eds.) Computer Aided Verification. pp. 47–67. Springer International Publishing, Cham (2017)

18. Dierl, S., Fiterau-Brostean, P., Howar, F., Jonsson, B., Sagonas, K., Tåquist, F.: Scalable tree-based register automata learning. In: Finkbeiner, B., Kovács, L. (eds.) TACAS. LNCS, vol. 14571, pp. 87–108. Springer (2024)

19. Drews, S., D’Antoni, L.: Learning symbolic automata. In: Legay, A., Margaria, T. (eds.) TACAS. pp. 173–189. Springer, Berlin, Heidelberg (2017)

20. Fisman, D., Frenkel, H., Zilles, S.: Inferring symbolic automata. Logical Methods in Computer Science Volume 19, Issue 2, 5 (2023)

21. Goles, E., Montealegre, P., Salo, V., Törmä, I.: Pspace-completeness of majority automata networks. Theor. Comput. Sci. 609, 118–128 (2016)

22. Grinchtein, O., Jonsson, B., Leucker, M.: Learning of event-recording automata. Theoretical Computer Science 411(47), 4029–4054 (2010)

23. Howar, F., Stefen, B., Jonsson, B., Cassel, S.: Inferring canonical register automata. In: Kuncak, V., Rybalchenko, A. (eds.) VMCAI. LNCS, vol. 7148, pp. 251–266. Springer (2012)

24. Kaminski, M., Francez, N.: Finite-memory automata. Theoretical Computer Science 134(2), 329–363 (1994)

25. Ley, C.: Forward looking logics and automata (2011)

26. Maler, O., Mens, I.E.: A Generic Algorithm for Learning Symbolic Automata from Membership Queries, pp. 146–169. Springer International Publishing (2017)

27. Mens, I.E., Maler, O.: Learning regular languages over large ordered alphabets. Logical Methods in Computer Science Volume 11, Issue 3, 13 (2015)

28. Moerman, J., Sammartino, M., Silva, A., Klin, B., Szynwelski, M.: Learning nominal automata. In: Castagna, G., Gordon, A.D. (eds.) POPL. pp. 613–625. ACM (2017)

29. Murawski, A.S., Ramsay, S.J., Tzevelekos, N.: Polynomial-time equivalence testing for deterministic fresh-register automata. In: Potapov, I., Spirakis, P.G., Worrell, J. (eds.) MFCS. LIPIcs, vol. 117, pp. 72:1–72:14. Schloss Dagstuhl - Leibniz-Zentrum für Informatik (2018)

30. Myhill, J.: Finite automata and the representation of events. In: Technical Report WADD TR-57-624. p. 112–137 (1957)

31. Neider, D., Smetsers, R., Vaandrager, F., Kuppens, H.: Benchmarks for Automata Learning and Conformance Testing, pp. 390–416. Springer International Publishing, Cham (2019)

32. Nerode, A.: Linear automaton transformations. In: American Mathematical Society. p. 541–544 (1958)

33. Neven, F., Schwentick, T., Vianu, V.: Finite state machines for strings over infinite alphabets. ACM Trans. Comput. Logic 5(3), 403–435 (2004)

34. Ozaki, A.: Actively learning from machine learning models with queries and counterexamples (extended abstract). In: Endriss, U., Melo, F.S., Bach, K., Diz, A.J.B., Alonso-Moral, J.M., Barro, S., Heintz, F. (eds.) ECAI. Frontiers in Artificial Intelligence and Applications, vol. 392, pp. 25–26. IOS Press (2024)

35. Royden, H., Fitzpatrick, P.: Real Analysis. Prentice Hall (2010)

36. de Ruiter, J., Poll, E.: Protocol state fuzzing of TLS implementations. In: Jung, J., Holz, T. (eds.) USENIX. pp. 193–206. USENIX Association (2015)

37. Shih, A., Darwiche, A., Choi, A.: Verifying binarized neural networks by Angluinstyle learning. In: Janota, M., Lynce, I. (eds.) SAT. LNCS, vol. 11628, pp. 354–370. Springer (2019)

38. Vaandrager, F.W., Garhewal, B., Rot, J., Wißmann, T.: A new approach for active automata learning based on apartness. In: Fisman, D., Rosu, G. (eds.) TACAS. LNCS, vol. 13243, pp. 223–243. Springer (2022)

39. Venhoek, D., Moerman, J., Rot, J.: Fast computations on ordered nominal sets. Theor. Comput. Sci. 935, 82–104 (2022)

40. Weiss, G., Goldberg, Y., Yahav, E.: Extracting automata from recurrent neural networks using queries and counterexamples. In: Dy, J.G., Krause, A. (eds.) ICML. PMLR, vol. 80, pp. 5244–5253. PMLR (2018)

## A Missing Proofs of Section 3

## A.1 Proof of Theorem 7

Theorem 7. The configuration reachability problem for DRAs over $( \mathbb { Z } , < )$ is decidable.

Proof. Let $\boldsymbol { \mathcal { A } } = ( Q , q _ { 0 } , F , \varDelta )$ be a well-typed DRA over $( \mathbb { Z } , < )$ with κ registers, $( p , v )$ be a configuration and q a state, and let min(v) and max(v) denote the smallest and largest integers in v, respectively. Suppose $\pi = ( p , v ) \ { \stackrel { u } { \to } } \ ( q , w )$ for some words u, w, where the step-by-step transitions of π are:

$$
( p _ { 0 } , v _ { 0 } ) \stackrel { a _ { 1 } } { \longrightarrow } ( p _ { 1 } , v _ { 1 } ) \stackrel { a _ { 2 } } { \longrightarrow } \ldots \stackrel { a _ { n } } { \longrightarrow } ( p _ { n } , v _ { n } ) .
$$

The statement of the theorem holds if the following claims are true:

(C1) We can obtain a run $\pi _ { \mathrm { s q u e e z e } }$ from $\pi ,$ , where

$$
\pi _ { \mathrm { s q u e e z e } } = ( p _ { 0 } , x _ { 0 } ) \xrightarrow { b _ { 1 } } ( p _ { 1 } , x _ { 1 } ) \xrightarrow { b _ { 2 } } \ldots \xrightarrow { b _ { n } } ( p _ { n } , x _ { n } )
$$

for some $x _ { 0 } , \ldots , x _ { n }$ and $b _ { 1 } , \ldots , b _ { n }$ . Moreover, the range of the values occurring in $x _ { 0 } , \ldots , x _ { n }$ is bounded between min $. ( v ) - 3 \kappa \cdot n$ and max $( v ) + 3 \kappa \cdot n$ (C2) If $\begin{array} { r } { n > ( \sum _ { i = 0 } ^ { \kappa } ( \operatorname* { m a x } ( v ) - \operatorname* { m i n } ( v ) + 2 ) ^ { i } ) \cdot | Q | \equiv l e n ( A , v ) } \end{array}$ , then there exists a run $\pi _ { \mathrm { s h o r t } }$ from $( p , v )$ to $( q , w ^ { \prime } )$ for some $w ^ { \prime }$ with $| \pi _ { \mathrm { s h o r t } } | < | \pi |$

With (C1) and (C2), to verify whether there exists a run from $( p , v )$ to $( q , w )$ for some $w ,$ it sufices to examine runs of length at most $\boldsymbol { l e n } ( \boldsymbol { A } , \boldsymbol { v } )$ and over configurations in $Q \times X$ , where X consists of words of length between 0 and κ and over symbols in $\{ \operatorname* { m i n } ( v ) - l e n ( A , v ) , \ldots , \operatorname* { m a x } ( v ) + l e n ( A , v ) \} \subset \mathbb { Z }$ . Since there are only finitely many such runs, the verification process is decidable.

To derive (C1) and (C2), we show that one can efectively obtain an orderpreserving mapping σ such that $\sigma ( v ) = v$ and

$$
\pi ^ { \prime } = \left( p , \sigma ( v ) \right) \xrightarrow { \sigma ( u ) } \left( q , \sigma ( w ) \right)
$$

is a valid run satisfying the following two properties: (P1) If a value between $\operatorname* { m i n } ( v ) - 3 \kappa \cdot n$ and ma $\mathfrak { c } ( v ) + 3 \kappa \cdot n$ is missing in u, then max $( \sigma ( u ) ) - \operatorname* { m i n } ( \sigma ( u ) ) ~ < ~$ $\operatorname* { m a x } ( u ) - \operatorname* { m i n } ( u ) . ( \mathrm { P 2 } )$ If $n > l e n ( \mathcal { A } , v )$ , then a subrun of $\pi ^ { \prime }$ can be removed while the remaining part is still a valid run, a property analogous to the pumping lemma for DFAs.

Consider (C1). Suppose there exist three values $a , \ a ^ { \prime } ,$ , and $a ^ { \prime \prime }$ , satisfying max $\begin{array} { r } { \mathrm { \Lambda } : ( v ) < a < a + 1 = a ^ { \prime } < a ^ { \prime } + 1 = a ^ { \prime \prime } \le \operatorname* { m a x } ( v ) + 3 \kappa \cdot n , } \end{array}$ , such that none of $a , a ^ { \prime }$ , or $a ^ { \prime \prime }$ appear in any $v _ { i }$ for $i \geq 1$ . Let $\sigma _ { a } ^ { s h i f t }$ be the order-preserving partial mapping that maps each value b to itself if $b < a ,$ and to $b - d \mathrm { i f } \ b > a ^ { \prime \prime }$ , where $d$ is the diference between $a ^ { \prime \prime }$ and the minimum value greater than $a ^ { \prime \prime }$ among all components in $v _ { 1 } , \ldots , v _ { n }$ and $a _ { 1 } , \ldots , a _ { n }$ . To ensure that $\sigma _ { a } ^ { s h i f t }$ is well-defined, we also require $\sigma _ { a } ^ { s h i f t } ( b ) = a ^ { \prime }$ for $b \in \{ a , a ^ { \prime } , a ^ { \prime \prime } \}$ . Then $\sigma _ { a } ^ { s h i f t } ( v ) = v$ , and

$$
\begin{array} { r } { \left( p _ { 0 } , \sigma _ { a } ^ { s h i f t } ( v _ { 0 } ) \right) \xrightarrow { \sigma _ { a } ^ { s h i f t } ( a _ { 1 } ) } \dots \xrightarrow { \sigma _ { a } ^ { s h i f t } ( a _ { n } ) } \left( p _ { n } , \sigma _ { a } ^ { s h i f t } ( v _ { n } ) \right) } \end{array}
$$

is still a valid run from $( p , v )$ to $( q , w ^ { \prime } )$ for some $w ^ { \prime } .$ . Moreover, the range of the values occurring in $\sigma _ { a } ^ { s h i f t } ( v _ { 1 } ) , \ldots , \sigma _ { a } ^ { s h i f t } ( v _ { n } )$ is strictly smaller than the range of $v _ { 1 } , \ldots , v _ { n }$ . This technique applies symmetrically when mi $\begin{array} { r } { \mathrm { ~ \ i ~ } ( v ) - 3 \kappa \cdot n \le a < } \end{array}$ min(v). By repeatedly applying this process at most $\kappa \cdot n$ times, we obtain the run π<sub>squeeze</sub> satisfying (C1).

Next, consider (C2). By the pigeonhole principle, ${ \mathrm { i f ~ } } | \pi | > | Q |$ , then along the run there must be a repeated state. Hence, there exists a decomposition $u = u _ { 1 } u _ { 2 } u _ { 3 }$ and a state r such that

$$
\pi = ( p , v ) \stackrel { u _ { 1 } } { \longrightarrow } ( r , x ) \stackrel { u _ { 2 } } { \longrightarrow } ( r , y ) \stackrel { u _ { 3 } } { \longrightarrow } ( q , w )\tag{1}
$$

for some words $x , y$ . We could immediately delete the subrun $( r , x ) \xrightarrow { u _ { 2 } } ( r , y )$ if $x = y$ , and the remaining run $( p , v ) \stackrel { u _ { 1 } } { \longrightarrow } ( r , x ) \stackrel { u _ { 3 } } { \longrightarrow } ( q , w )$ would still be valid. However, in general we do not have $x = y .$

More generally, for each $\ell \in \mathbb { N } , \mathrm { i f } | \pi | > \ell \cdot | Q |$ , we obtain a decomposition

$$
\pi = \left( p , v \right) { \overset { u _ { 1 } } { \longrightarrow } } \left( r , x _ { 1 } \right) { \overset { u _ { 2 } } { \longrightarrow } } \left( r , x _ { 2 } \right) \ldots { \overset { u _ { \ell } } { \longrightarrow } } \left( r , x _ { \ell } \right) { \overset { u _ { \ell + 1 } } { \longrightarrow } } \left( q , w \right)
$$

for some $x _ { 1 } , \ldots , x _ { \ell }$ and $u _ { 1 } , \ldots , u _ { \ell + 1 }$ . For each $i ,$ let $y _ { i }$ be the sorted version of $x _ { i }$ Then $y _ { 1 } , \ldots , y _ { \ell }$ are strictly increasing sequences of the same length θ for some $\theta .$ Let $y _ { i } = a _ { 1 } ^ { [ i ] } \ldots a _ { \theta } ^ { [ i ] }$ for each $i = 1 , \ldots , \ell$ . If there exist two distinct indices $i < j$ such that, for every $s = 1 , \ldots , \theta - 1$

$$
a _ { s + 1 } ^ { [ i ] } - a _ { s } ^ { [ i ] } \geq a _ { s + 1 } ^ { [ j ] } - a _ { s } ^ { [ j ] } ,\tag{2}
$$

then by letting σ be the mapping $\sigma ( a ) =$

$$
\bullet a + ( a _ { 1 \atop - } ^ { [ i ] } - a _ { \underline { { { 1 } } } \atop - } ^ { [ j ] } ) { \mathrm { ~ i f ~ } } a < a _ { 1 } ^ { [ j ] } .
$$

$$
\bullet a + ( a _ { s } ^ { [ i ] } - a _ { s } ^ { [ j ] } ) \mathrm { ~ i f ~ } a _ { s } ^ { [ j ] } \leq a \leq a _ { s + 1 } ^ { [ j ] } \mathrm { ~ f o r ~ } s = 1 , \ldots , \theta - 1 .
$$

$$
\bullet a + ( a _ { \theta } ^ { [ i ] } - a _ { \theta } ^ { [ j ] } ) \mathrm { { i f } } a \geq a _ { \theta } ^ { [ j ] } .
$$

Hence, we have that $\sigma$ is order-preserving, and

$$
\left( r , \sigma ( x _ { j } ) \right) \xrightarrow { \sigma ( u _ { j } ) } \left( r , \sigma ( x _ { j + 1 } ) \right) \xrightarrow { \sigma ( u _ { j + 1 } ) } \ldots \left( r , \sigma ( x _ { \ell } ) \right) \xrightarrow { \sigma ( u _ { \ell + 1 } ) } \left( q , \sigma ( w ) \right)
$$

is a valid run from $( r , \sigma ( x _ { j } ) )$ to $( q , \sigma ( w ) )$ on $\sigma ( u _ { j } \cdot \cdot \cdot u _ { \ell + 1 } )$ , where $\sigma ( x _ { j } ) = x _ { i }$ Accordingly,

$$
( p , v ) \xrightarrow { u _ { 1 } } \ldots ( r , x _ { i } ) { \xrightarrow { \sigma ( u _ { j } ) } } ( r , \sigma ( x _ { j + 1 } ) ) { \xrightarrow { \sigma ( u _ { j + 1 } ) } } \ldots ( r , \sigma ( x _ { \ell } ) ) { \xrightarrow { \sigma ( u _ { \ell + 1 } ) } } ( q , \sigma ( w ) )
$$

is a shorter valid run from $( p , v )$ to $( q , \sigma ( w ) )$

The following property of finite sequences will be used later:

Lemma 20. Let $\delta _ { 1 } , \ldots , \delta _ { \ell }$ be ℓ pairwise distinct strictly increasing sequences over N, where each $\delta _ { i } = d _ { 1 } ^ { [ i ] } \ldots d _ { \theta } ^ { [ i ] }$ for some fixed θ. Given a constant $d \in \mathbb { N } _ { \mathrm { \Sigma } }$ , if $\ell > ( d + 2 ) ^ { \theta }$ , then there exist two distinct indices $i , j$ such that for every $s ,$ both of values $d _ { s } ^ { [ i ] }$ and $d _ { s } ^ { [ j ] }$ are greater than d whenever they are not equal.

Proof. Consider the sequences over $\{ 0 , 1 , \ldots , d , d + 1 \}$ of length θ. There are exactly $( d + 2 ) ^ { \theta }$ such sequences. Among those sequences, there is at least one over exactly $\{ 0 , d + 1 \}$ . Given two distinct $\delta _ { i }$ and $\delta _ { j } .$ , we let $e _ { i , j } = f _ { 1 } \ldots f _ { \theta }$ a sequence over $\{ 0 , 1 , \ldots , d , d + 1 \}$ , where $f _ { s } = 0$ if $\bar { d } _ { s } ^ { \bar { [ { i } ] } } = \delta _ { s } ^ { [ { j } ] } ; = t$ if $d _ { s } ^ { [ i ] } \neq \delta _ { s } ^ { [ j ] }$ and min $( d _ { s } ^ { [ i ] } , \delta _ { s } ^ { [ j ] } ) = t ; = d + 1$ , otherwise. If $\ell > ( d + 2 ) ^ { \theta }$ , there exists $e _ { i , j }$ over exactly $\{ 0 , d + 1 \}$ . Accordingly, if $\ell > ( d + 2 ) ^ { \theta }$ there are at least two $\delta _ { i }$ and $\delta _ { j }$ such that $\delta _ { i } \neq \delta _ { j }$ and for each $s ,$ both of values $d _ { s } ^ { [ i ] }$ and $d _ { s } ^ { [ j ] }$ are greater than d whenever they are not equal. □

Now assume that for every distinct pair of i and j, there exists s such that

$$
a _ { s + 1 } ^ { [ i ] } - a _ { s } ^ { [ i ] } \ \neq \ a _ { s + 1 } ^ { [ j ] } - a _ { s } ^ { [ j ] } .\tag{3}
$$

By Lemma 20, if $\ell > ( \operatorname* { m a x } ( v ) - \operatorname* { m i n } ( v ) + 2 ) ^ { \theta }$ , then there exist indices $i , j$ such that both diferences $a _ { s + 1 } ^ { [ i ] } - a _ { s } ^ { [ i ] }$ and $a _ { s + 1 } ^ { [ j ] } - a _ { s } ^ { [ j ] }$ exceed max(v) min(v) whenever, for every distinct pair $i , j$ , there exists an index s satisfying Eq. 3.

We define ι as an index between 1 and $\theta ,$ where:

ι = 1 if $a _ { 1 } ^ { [ i ] } > \operatorname* { m a x } ( v )$ ，

ι = θ if $a _ { \theta } ^ { [ i ] } < \operatorname* { m i n } ( v )$

otherwise, ι is the smallest index satisfying min $( v ) \leq a _ { \iota } ^ { [ i ] } \leq \operatorname* { m a x } ( v )$

Let $d _ { 1 } \dots d _ { \theta - 1 }$ be the sequence defined by

$$
d _ { s } = \operatorname* { m a x } \{ a _ { s + 1 } ^ { [ i ] } - a _ { s } ^ { [ i ] } , a _ { s + 1 } ^ { [ j ] } - a _ { s } ^ { [ j ] } \}
$$

for all $s = 1 , \ldots , \theta - 1$ . We define σ as follows:

$$
\bullet \sigma ( a ) = a + a _ { \iota } ^ { [ i ] } + ( d _ { \iota } + \cdot \cdot \cdot + d _ { s - 1 } ) \mathrm { ~ i f ~ } a _ { s } ^ { [ i ] } \leq a \leq a _ { s + 1 } ^ { [ i ] } \mathrm { ~ a n d ~ } s \geq \iota ,
$$

• <sup>σ(a)</sup> <sup>=</sup> <sup>a</sup> <sup>+</sup> <sup>a[i]</sup>ι − <sup>(d</sup>ι−1 <sup>+</sup> · · · <sup>+</sup> <sup>d</sup>s+1<sup>)</sup> <sup>if</sup> <sup>a[i]</sup>s ≤ <sup>a</sup> ≤ <sup>a[i]</sup>s+1 <sup>and</sup> [i] $s < \iota$

Then the following properties hold:

– σ is order-preserving,

$\sigma ( a _ { s + 1 } ^ { [ i ] } ) - \sigma ( a _ { s } ^ { [ i ] } ) = d _ { s }$ for all s,

$\sigma ( v ) = v$ , since there is at most one index s satisfying Eq. 3 and for that index we have min $( v ) \leq a _ { s } ^ { [ i ] } \leq \operatorname* { m a x } ( v )$

Accordingly,

$$
\pi _ { \mathrm { p r e } } = ( p , \sigma ( v ) ) { \xrightarrow { \sigma ( u _ { 1 } ) } } \dots { \xrightarrow { \sigma ( u _ { i } ) } } ( r , \sigma ( x _ { i } ) )
$$

is a valid run from $( p , v )$ to $( r , \sigma ( x _ { i } ) )$ on $\sigma ( u _ { 1 } ) \cdot \cdot \cdot \sigma ( u _ { i } )$ . Hence, $\pi _ { \mathrm { p r e } }$ together with the run $\left( r , x _ { j } \right) \xrightarrow { u _ { j + 1 } \cdots u _ { \ell + 1 } } \left( q , w \right)$ satisfies Eq. 2. □

Algorithm 3: Minimization algorithm for DRAs   
input: A DRA $\mathcal { A } = ( Q , q _ { 0 } , F , \varDelta )$ recognizing L over $( \Sigma , R )$   
1 for $q \in Q$ do   
2 Obtain a word $w _ { q }$ such that $\left( q _ { 0 } , \epsilon \right) \xrightarrow { w _ { q } } _ { A } \left( q , w \right)$ for some w;   
3 end   
4 $f l a g \gets \top ;$   
5 while flag do   
6 $f l a g \gets \bot ;$   
7 for $p \in Q$ do   
8 for $q \in Q \backslash$ {<sup>p</sup>} <sup>do</sup>   
9 if $w _ { p } \cong _ { L } w _ { q }$ then   
10 Merge p and q.;   
11 $f l a g \gets \top ;$   
12 end   
13 end   
14 end   
15 end   
16 for $q \in Q$ do   
17 Modify transitions starting from q.;   
18 end   
19 return $\mathcal { A } ;$

## A.2 Minimization Algorithm for DRAs

By Theorem 6, the minimization problem for DRAs is decidable, provided that the canonical automaton of every given DRA can be efectively constructed. For DFAs, the canonical automaton can be obtained by merging states that correspond to the same residual language. The same idea applies to DRAs, as guaranteed by the following lemma:

Lemma 21 ( [7]). Let  be a DRA recognizing L. For any two words $u , v _ { \mathrm { { i } } }$ $i f \left( q _ { 0 } , \epsilon \right) \stackrel { u } { \longrightarrow } \left( q , x \right)$ and $( q _ { 0 } , \epsilon ) \stackrel { v } {  } ( q , y )$ for some configurations $( q , x )$ and $( q , y )$ then u $\cong _ { L } v$

The pseudocode for DRA minimization is given in Algorithm 3. In the algorithm, we first compute a representative word $w _ { q }$ for each state $q .$ . Then, for any pair of states $p , q ,$ we merge them whenever $w _ { p } \cong _ { L } w _ { q }$ . After this step, the automaton is already state-minimal. However, it may still fail to be data-minimal. Therefore, in a final step, we use $m e m _ { L } ( w _ { q } )$ for each state $q$ to adjust transitions outgoing from q so as to achieve data-minimality.

The decidability of the following three problems ensures that Algorithm $3$ is an efective minimization procedure:

Representative problem: Given a state q, compute a representative word $w _ { q }$ such that $( q _ { 0 } , \epsilon ) \xrightarrow { w _ { q } } ( q , w )$ for some word $w ,$ whenever such a representative exists.

Memorability problem: Given a word $u ,$ compute $\begin{array} { r } { m e m _ { L } ( u ) } \end{array}$

Word-equivalence problem: Given two words $u , v ,$ verify whether $u \cong _ { L } v$

Clearly, the representative problem is decidable if the configuration reachability problem is decidable and a witness word $w _ { q }$ can be efectively extracted.

As for DFAs, the equivalence problem of two DRAs reduces to the reachability problem for their product automaton.

Definition 22. Given two DRA $\mathcal { A } _ { 1 }$ and $\boldsymbol { A } _ { 2 }$ , the product automaton $\mathcal { A } _ { 1 } \times \mathcal { A } _ { 2 }$ is a tuple $\left( Q ^ { \times } , q _ { 0 } ^ { \times } , F ^ { \times } , \varDelta ^ { \times } \right)$ where:

$Q ^ { \times } = Q ^ { 1 } \times Q ^ { 2 }$ is a set of states.

$( q _ { 0 } ^ { 1 } , q _ { 0 } ^ { 2 } ) \in Q ^ { \times } \ i s$ the initial state.

$F ^ { \dot { \times } } = F ^ { 1 } \times F ^ { 2 }$ is the set of final states.

$\varDelta  { \boldsymbol { \times } } \quad \varDelta  { \boldsymbol { s } } \ \boldsymbol { a }$ finite set of transitions of the form $\left( ( p _ { 1 } , p _ { 2 } ) , ( \tau _ { 1 } , \tau _ { 2 } ) , ( E _ { 1 } , E _ { 2 } \right)$ $( q _ { 1 } , q _ { 2 } ) )$ , where $( p _ { i } , \tau _ { i } , E _ { i } , q _ { i } ) \in \varDelta ^ { i } \ f o r \ i \in \{ 1 , 2 \}$

This product register automaton difers from the RAs in Definition 1: it is not well-typed, since its registers may contain duplicate values. However, it can be translated into a DRA satisfying Definition 1 by introducing additional states that simulate the register configuration so as to prevent duplicate values. Thus, the equivalence problem for DRAs from given configurations is decidable whenever the configuration reachability problem is decidable.

Given a word $u ,$ if $( q _ { 0 } , \epsilon ) \stackrel { u } {  } _ { \cal A } ( q , w )$ , then $m e m _ { L } ( u )$ is a subsequence of $w ,$ obtained by deleting all non-memorable symbols. If every extension run $( q , w ) \stackrel { s } {  } ( r , x )$ from $( q , w )$ satisfies $r \not \in F$ , then mem $\mathbf { \Omega } _ { L } ( u ) = \epsilon$ . Otherwise, suppose $( q , w ) \stackrel { s } {  } ( r , x )$ for some $r \in F ,$ and let $w ^ { \prime } = \sigma ( w )$ for some order preserving mapping $\sigma$ which maps a to a diferent value b. If the DRA started from $( q , w )$ is not equivalent to that started from $( q , w ^ { \prime } )$ , then a is memorable; otherwise, it is not. When $\varSigma$ is dense, such $\mathrm { ~ a ~ } w ^ { \prime }$ always exists. This need not hold for non-dense alphabets. For instance, if $w = 2 \cdot 3 \cdot 4$ over $\mathbb { Z }$ and $a = 3$ , then there is no $b \neq 3$ such that σ preserves the ordering relations in $w .$ . In contrast, if $w = 1 \cdot 3 \cdot 5$ and $a = 3 .$ , we may choose $b = 4$

Recall that w arises from a run $( q _ { 0 } , \epsilon ) \stackrel { u } {  } ( q , w )$ , and if $u \cong _ { L }$ u<sup>′</sup> with $( q _ { 0 } , \epsilon ) \stackrel { u ^ { \prime } } {  }$ $( q , w ^ { \prime } )$ , then the ith symbol of w is memorable in u if the ith symbol of $w ^ { \prime }$ is memorable in $u ^ { \prime } .$ Since $L$ is a data language, for every $u \sim _ { R } u ^ { \prime }$ , we have u $\cong _ { L }$ u<sup>′</sup>. For any sequence $u ,$ we can efectively compute a sequence $u ^ { \prime }$ such that $u ^ { \prime } \sim _ { R }$ u and any two distinct symbols in $u ^ { \prime }$ difer by more than 1. For example, if $u = 1 \cdot 2 \cdot 3 .$ , we may take $u ^ { \prime } = 1 \cdot 3 \cdot 5$ . Hence the memorability problem reduces to the configuration equivalence problem, and thereby to the configuration reachability problem.

Suppose there exists an R-preserving mapping σ such that $\sigma ( m e m _ { L } ( u ) ) =$ mem $\mathbf { \Omega } _ { L } ( v )$ for given words u and $v ,$ and let $u ^ { \prime } = \sigma ( u )$ . Then $u \cong _ { L }$ v if for all words s, $u ^ { \prime } \cdot s \in L \Longleftrightarrow v \cdot s \in L$ . Let $( q _ { u ^ { \prime } } , w _ { u ^ { \prime } } )$ and $( q _ { v } , w _ { v } )$ be the configurations reached from $( q _ { 0 } , \epsilon )$ upon reading $u ^ { \prime }$ and $v ,$ respectively. Therefore, u $\cong _ { L }$ v if the DRA  starting from $( q _ { u ^ { \prime } } , w _ { u ^ { \prime } } )$ is equivalent to the one starting from $( q _ { v } , w _ { v } )$ . Consequently, the word-equivalence problem, and thus the minimization problem for DRAs, is decidable because the configuration reachability problem is decidable.

## B More Examples on the Learning Algorithm

![](images/e059645d682227edde6ace9cfb51fd63d53369648e6030c0117c8213ac8a899e.jpg)  
Fig. 4: Closed Table $\mathcal { T } _ { 2 }$ and its derived DRA $\mathcal { H } _ { 2 }$

Example 23. Let us reconsider the examples in Fig. 3. Assume that the teacher returns counterexample $w = 4 { \cdot } 5 { \cdot } 4 { \cdot } 5$ to the learner on equivalence query $\mathsf { E Q } ( \mathcal { H } _ { 1 } )$ Clearly, $w \in L$ but w / $L ( \mathcal { H } _ { 1 } )$ . Initially, we have $u = \varepsilon , r = \varepsilon$ and $m q = +$ . In the first iteration, we decompose $w = a \cdot z$ with $a = 4$ and $z = 5 { \cdot } 4 { \cdot } 5$ . Reading $a = 4$ the transition $( \varepsilon , \varepsilon ) \xrightarrow { 0 : \varnothing } ( 0 , 4 )$ is taken when reading a, so $b = 0$ . By applying $\sigma _ { 1 }$ which maps 4 to 0, we obtain $\sigma _ { 1 } ( z ) = 1 { \cdot } 0 { \cdot } 1$ . Furthermore, σ<sub>2</sub> maps $M ( u \cdot b ) = 0$ to $M ( u ^ { \prime } ) = M ( 0 ) = 0$ , hence $\sigma$ is an identity map. Thus, $\sigma _ { 2 } ( \sigma _ { 1 } ( z ) ) = 1 \cdot 0 \cdot 1$ and $\mathsf { M Q } ( u ^ { \prime } \cdot \sigma _ { 2 } ( \sigma _ { 1 } ( z ) ) ) = \mathsf { M Q } ( 0 \cdot 1 \cdot 0 \cdot 1 ) = +$ . We then set $u = 0 , r = 0$ and $w = 1 \cdot 0 \cdot 1$ and go to next iteration. In the second iteration, we again write $w = a \cdot z$ with $a = 1$ and $z = 0 { \cdot } 1$ . The transition $( 0 , 0 ) \xrightarrow { 0 \cdot 1 : \emptyset } ( 0 \cdot 1 , 0 \cdot 1 )$ is taken. Since $M ( u ) \cdot a = M ( u ) \cdot b = M ( u ^ { \prime } )$ , σ and $\sigma _ { 2 }$ are both identity mappings. Thus, $\sigma _ { 2 } ( \sigma _ { 1 } ( z ) ) = 0 . 1$ and $\mathsf { M Q } ( u ^ { \prime } \cdot \sigma _ { 2 } ( \sigma _ { 1 } ( 0 \cdot 1 ) ) ) = \mathsf { M Q } ( 0 \cdot 1 \cdot 0 \cdot 1 ) = +$ . We set $u = 0 . 1 , r = 0 . 1$ and $w = 0 1$ and go to the next iteration. In this last iteration, the same reasoning applies: $a = 0$ and $z = 1$ and the transition $( 0 \cdot 1 , 0 \cdot 1 ) \xrightarrow { 0 1 0 : \{ 1 , 3 \} } ( 0 , 1 )$ is taken; so $b = 0$ . Since $a = b$ and thus $M ( u ) \cdot a = M ( u ) \cdot b , \sigma _ { 1 }$ is again the identity mapping. We have $M ( u \cdot b ) = M ( 0 \cdot 1 \cdot 0 ) = 1 { \mathrm { ~ a n d ~ } } M ( u ^ { \prime } ) = M ( 0 ) = 0$ , giving $v = \sigma _ { 2 } ( \sigma _ { 1 } ( z ) ) = 0$ . Thus, ${ \sf M Q } ( u ^ { \prime } \cdot v ) = { \sf M Q } ( 0 \cdot 0 ) = -$ . Therefore the returned word is $v = 0$ which will be added into S. After closing the updated table, we will obtain the table $\mathcal { T } _ { 2 }$ and derive a new conjecture DRA $\mathcal { H } _ { 2 }$ , as shown in Fig. 4. After refining $\mathcal { H } _ { 2 }$ with counterexample $1 { \cdot } 0 { \cdot } 1 { \cdot } 2 { \cdot } 1 { \cdot } 2$ , we will get the final table $\tau$ and construct from $\tau$ the correct conjecture DRA in Fig. 2. □

## C More Details on Learning DRAs over Non-dense Domains

We mentioned in the main content that one can learn DRAs over $( \mathbb { Z } , < )$ with our DRA learner for $( \mathbb { Q } , < )$ . Here we provide more details on that.

Assume that the teacher answers only membership and memorability queries on words over $\mathbb { Z } ,$ and equivalence query over a DRA over $\mathbb { Z } .$ . Let $L _ { \mathbb { Z } }$ be our target language over $\mathbb { Z }$ and $L _ { \mathbb { Q } }$ be its corresponding language over $\mathbb { Q }$ . We can just add an interface between our DRA learner of dense domains and the teacher for non-dense domains.

For asking a membership query for a word u over $\mathbb { Q } .$ , we can use a bijective order-preserving mapping $\sigma$ that maps a word over $\mathbb { Q }$ to a word over $\mathbb { Z } .$ For instance, when $u = 1 \cdot 2 \cdot { \textstyle { \frac { 1 } { 2 } } }$ , we have $\sigma ( u ) = 2 \cdot 4 \cdot 1$ . We then return the response on ${ \sf M Q } ( \sigma ( u ) )$ . Since u $\sim _ { R } \ \sigma ( u )$ , they should have the same membership in the target language $L _ { \mathbb { Q } }$

– For asking a memorability query for a word u over $\mathbb { Q } .$ , we also ask the memorability query $\mathsf { M e m } ( \sigma ( u ) )$ instead and return $\sigma ^ { - 1 } ( \mathsf { M e m } ( \sigma ( u ) ) )$ . Since $u \sim _ { R } \sigma ( u )$ over the dense domains, we know that the memorable words for $\sigma ( u )$ have the same positions as the memorable words for u.

– To resolve the equivalence query on the conjectured DRA  over $\mathbb { Q } .$ we convert it to a DRA $\mathcal { H } ^ { \prime }$ over $\mathbb { Q }$ for all word types of transitions using σ and ask equivalence qeury $\mathsf { E Q } ( \mathcal { H } ^ { \prime } )$ . If the answer is ${ } ^ { \mathfrak { s o y } } \mathrm { e s } ^ { \mathfrak { n } }$ , we output the DRA $\mathcal { H } ^ { \prime }$ over $\mathbb { Z } ,$ otherwise, just return the counterexample w to our DRA learner, since it is also a valid counterexample over $\mathbb { Q }$

By Theorem $^ { 9 , }$ this learning algorithm will output a correct DRA of the target language $L _ { \mathbb { Z } }$

## D Missing Proofs of Section 4.1

Lemma 11. Let $\varSigma$ be a dense set. For two words $u _ { 1 } , u _ { 2 } \in \Sigma ^ { * }$ with mem $_ L ( u _ { 1 } ) \sim _ { R } m e m _ { L } ( u _ { 2 } ) , u _ { 1 } \not \cong _ { L } u _ { 2 }$ $i f$ and only $i f$ there exist a bijective orderpreserving mapping σ and a word w such that $\sigma ( m e m _ { L } ( u _ { 1 } ) ) = m e m _ { L } ( u _ { 2 } )$ and $\sigma ( u _ { 1 } ) \cdot w \in L \not \circledast u _ { 2 } \cdot w \in L$

Proof. We first prove the “only $\mathrm { i f } ^ { \dag }$ direction. Suppose that $u _ { 1 } \nsimeq _ { L } u _ { 2 }$ . Since mem<sub>L</sub> $_ { \mathit { \Pi } _ { \prime } } ( u _ { 1 } ) \sim _ { R } \ m e m _ { L } ( u _ { 2 } )$ , by Definition $5 ,$ there exist words $x , y$ such that mem<sub>L</sub> $, ( u _ { 1 } ) \cdot x \sim _ { R } m e m _ { L } ( u _ { 2 } ) \cdot y _ { ! }$ , but exactly one of the words $u _ { 1 } \cdot x$ and $u _ { 2 } \cdot y$ is in $L .$ . Because $\varSigma$ is dense, the relation mem ${ \bf \nabla } _ { \cdot } L ( u _ { 1 } ) \cdot x \sim _ { R } m e m _ { L } ( u _ { 2 } ) \cdot y$ implies the existence of an order-preserving mapping σ such that $\tau ( m e m _ { L } ( u _ { 1 } ) \cdot x ) =$ mem $\mathbf { \Psi } _ { L } ( u _ { 2 } ) \cdot y$ . By letting $w = \sigma ( x ) = y$ , we obtain that either $\sigma ( u _ { 1 } ) \cdot w \in L$ or $u _ { 2 } \cdot w \in L$

Next, we prove the $ { \mathrm { ^ 6 6 } } _ { 1 }  { \mathrm { f } } ^ { \flat }$ direction. Suppose there exist a word w and an orderpreserving mapping $\sigma$ such that σ(mem $_ L ( u _ { 1 } ) ) = m e m _ { L } ( u _ { 2 } )$ and $\sigma ( u _ { 1 } ) \cdot w \in$ $L \not \Leftrightarrow u _ { 2 } \cdot w \in L$ . Let $x = \sigma ^ { - 1 } ( w )$ and $y = w$ . Since $\sigma$ is order-preserving and $\varSigma$ is dense, we have mem ${ \bf \Phi } _ { \cdot L } ( u _ { 1 } ) \cdot x \sim _ { R } m e m _ { L } ( u _ { 2 } ) \cdot y$ . Moreover, exactly one of the words $u _ { 1 } \cdot x$ and $u _ { 2 } \cdot y$ is in L. Hence, $u _ { 1 } \not \equiv _ { L } u _ { 2 }$ □

To order to prove Lemma 12, it sufices to prove the following Lemma 24, since Lemma 12 is a direct consequence of Lemma 24.

Lemma 24. Let L be a data language over $( \Sigma , R )$ where Σ is dense. For a word $u \in \Sigma ^ { * }$ and two bijective order-preserving mappings $\sigma , \sigma ^ { \prime }$ with σ(mem $\mathbf { \Psi } _ { L } ( u ) ) =$ $\sigma ^ { \prime } ( m e m _ { L } ( u ) )$ , it holds that $\sigma ( u ) \cdot w \in L \Leftrightarrow \sigma ^ { \prime } ( u ) \cdot w \in L$ for all $v \in \Sigma ^ { * }$

Proof. Since σ and $\sigma ^ { \prime }$ are bijective, their inverses $\sigma ^ { - 1 }$ and $\sigma ^ { \prime - 1 }$ exist and are also order-preserving. Let u be a word. By the assumption $\sigma ( m e m _ { L } ( u ) ) = \sigma ^ { \prime } ( m e m _ { L } ( u ) )$ , it follows that for every word w $\in \Sigma ^ { * }$ , mem<sub>L</sub>(u) $\sigma ^ { - 1 } ( w ) \sim _ { R } \sigma ( m e m _ { L } ( u ) \cdot \sigma ^ { - 1 } ( w ) ) = \sigma ( m e m _ { L } ( u ) ) \cdot w = \sigma ^ { \prime } ( m e m _ { L } ( u ) ) \cdot w \sim _ { R } $ $\sigma ^ { \prime - 1 } ( \sigma ^ { \prime } ( m e m _ { L } ( u ) ) { \cdot } w ) = m e m _ { L } ( u ) { \cdot } \sigma ^ { \prime - 1 } ( w )$ holds. That is, mem $\scriptstyle { \boldsymbol { L } } ( u ) \cdot \sigma ^ { - 1 } ( w )$ ∼R mem $\boldsymbol { \mathbf { \ell } } _ { L } ( u ) \cdot \boldsymbol { \sigma } ^ { \prime - 1 } ( w )$ holds for every w $\in \Sigma ^ { * }$ . By definition of $\cong _ { L } .$ , since mem $\mathbf { \Psi } _ { L } ( u )$ $\sigma ^ { - 1 } ( w ) \sim _ { R }$ mem $\dot { \mathbf { \Omega } _ { L } } ( u ) \cdot \boldsymbol { \sigma } ^ { \prime - 1 } ( w )$ , we have $u \cdot \sigma ^ { - 1 } ( w ) \in L \Leftrightarrow u \cdot \sigma ^ { \prime - 1 } ( w ) \in L$ for all w $\ u _ { 1 } \in \Sigma ^ { * }$ because u $\cong _ { L }$ u holds. It then follows that, $\sigma ( u ) \cdot w \in L \Leftrightarrow \sigma ^ { \prime } ( u ) \cdot w \in L$ for all w $\in \Sigma ^ { * }$ since σ and $\sigma ^ { \prime }$ are order-preserving mappings. □

Lemma 12. Let Σ be a dense set, and let $u _ { 1 } , u _ { 2 }$ be distinct words. If σ is a bijective order-preserving mapping with σ(mem $_ L ( u _ { 1 } ) ) = m e m _ { L } ( u _ { 2 } )$ , then $u _ { 1 } \not \equiv _ { L }$ u<sub>2</sub> if there exists $w \in \Sigma ^ { * }$ such that $\sigma ( u _ { 1 } ) \cdot w \in L \not \circledast u _ { 2 } \cdot w \in L$

Proof. It is a direct consequence of Lemma 24.

The following lemma, which captures the key properties of S-consistency, follows directly from its definition.

## Lemma 25. Let $S \subseteq \Sigma ^ { * }$ be a set of words. We have:

1. if u is not S-consistent with v, then u $\ncong _ { L } v ;$

2. if u is not S-consistent with v, then u is not S-consistent with v whenever $S \subseteq S ^ { \prime }$

We present examples below to show that S-consistency is not an equivalence relation: it is neither symmetric nor transitive.

Example 26. Consider the language $L _ { 4 }$ where $n = 4$ for the language in Example 2, that is, $L _ { 4 } = \{ w \in \mathbb { Q } \mid | w | = 4$ and w is strictly increasing or decreasing .

<table><tr><td></td><td>M|</td><td>ε</td><td>1·2</td></tr><tr><td>ε 0</td><td>ε 0</td><td>一 一</td><td>一 一</td></tr><tr><td>0·1</td><td>1</td><td></td><td>一</td></tr><tr><td>0·1·2</td><td>2</td><td>—</td><td>一</td></tr></table>

During the learning process, the above observation table may arise. The prefixes of the first three rows belong to $U .$ , whereas the last row, $0 \cdot 1 \cdot 2$ , belongs to $X$ , the only one-letter extension in X shown in this table. We denote by $u _ { i }$ the prefix corresponding to the ith row. This table witnesses non-symmetry: u<sub>2</sub> is S-consistent with $u _ { 3 }$ , but $u _ { 3 }$ is not S-consistent with $u _ { 2 }$ . In addition, the oneletter extension 0 1 2 (in X) is S-consistent with both $u _ { 2 }$ and $u _ { 3 }$ , witnessing non-transitivity of S-consistency.

In order to prove Lemma 15, we first prove the following Lemma.

Lemma 27. Let $w = a _ { 1 } \cdot \cdot \cdot a _ { d }$ and $\pi = ( u _ { 0 } , r _ { 0 } ) \stackrel { a _ { 1 } } { \longrightarrow } \cdots \stackrel { a _ { d } } { \longrightarrow } ( u _ { d } , r _ { d } )$ be the run of  on w. Let $u _ { 0 } ^ { \prime } , \cdot \cdot \cdot , u _ { d } ^ { \prime }$ be the sequence of states visited in Algorithm 1 if we remove the $_ { i f }$ statement $( i . e .$ , we do not use the early return). It then holds that $u _ { i } ^ { \prime } = u _ { i }$ for all $0 \leq i \leq d .$

Proof. Let $r _ { 0 } ^ { \prime } , \cdot \cdot \cdot , r _ { d } ^ { \prime }$ and $a _ { 1 } ^ { \prime } z _ { 0 } , \cdots , a _ { d } ^ { \prime } z _ { d - 1 }$ be the corresponding register values and the remaining sufixes visited in Algorithm 1 where $z _ { d } = \varepsilon$ and $a _ { 1 } ^ { \prime } z _ { 0 } = w$ Let $r _ { 0 } ^ { \prime } b _ { 1 } , \cdot \cdot \cdot , r _ { d } ^ { \prime } b _ { d }$ be the selected word type visited in Algorithm 1. We first prove the following invariants in the whole loop by induction:

$$
\mathrm { ( i ) } ~ r _ { i } ^ { \prime } \cdot a _ { i + 1 } ^ { \prime } \sim _ { R } M ( u _ { i } ^ { \prime } ) \cdot b _ { i + 1 } ~ \mathrm { f o r ~ a l l } ~ 0 \leq i < d .
$$

$$
\mathrm { ( i i ) } \ r _ { i } ^ { \prime } a _ { i + 1 } ^ { \prime } z _ { i } \sim _ { R } r _ { i } ^ { \prime } b _ { i + 1 } \sigma _ { 1 } ( z _ { i } ) \ \mathrm { f o r \ a l l } \ 0 \le i < d .
$$

For the base case, obviously, when $i = 0$ , we have $u _ { 0 } ^ { \prime } = r _ { 0 } ^ { \prime }$ and $z _ { 0 } = w [ 2 \cdots ]$ . It follows that $M ( u _ { 0 } ^ { \prime } ) = \varepsilon$ and thus $r _ { 0 } ^ { \prime } \cdot a _ { 1 } ^ { \prime } \sim _ { R } M ( u _ { 0 } ^ { \prime } ) \cdot b _ { 1 }$ holds directly. Moreover, $\sigma _ { 1 }$ is in fact an order-preserving mapping from $a _ { 1 } ^ { \prime } = a _ { 1 }$ to $b _ { 1 }$ . Hence, $\sigma _ { 1 } ( a _ { 1 } ^ { \prime } z _ { 0 } ) =$ $b _ { 1 } \cdot \sigma _ { 1 } ( z _ { 0 } ) \sim _ { R } a _ { 1 } ^ { \prime } z _ { 0 }$

For the induction step, assume that it holds for all $i \leq \ell .$ According to Algorithm 1, $r _ { \ell + 1 } ^ { \prime } = M ( u _ { \ell + 1 } ^ { \prime } )$ and $z _ { \ell + 1 } = \sigma _ { 2 } ( \sigma _ { 1 } ( z _ { \ell } ) )$ . Since when reading $a _ { \ell + 2 } ^ { \prime }$ at configuration $( u _ { \ell + 1 } ^ { \prime } , r _ { \ell + 1 } ^ { \prime } = M ( u _ { \ell + 1 } ^ { \prime } ) )$ , the transition with the word type $M ( u _ { \ell + 1 } ^ { \prime } ) \cdot b _ { \ell + 2 }$ is selected, it indicates that $\boldsymbol { r } _ { \ell + 1 } ^ { \prime } \cdot \boldsymbol { a } _ { \ell + 2 } ^ { \prime } = \boldsymbol { M } ( \boldsymbol { u } _ { \ell + 1 } ^ { \prime } ) \cdot \boldsymbol { a } _ { \ell + 2 } ^ { \prime } \sim _ { R }$ $M ( u _ { \ell + 1 } ^ { \prime } ) \cdot b _ { \ell + 2 }$ . Recall that current $\sigma _ { 1 }$ is an order-preserving mapping from $r _ { \ell + 1 } ^ { \prime } \cdot \stackrel { \cdot } { a } _ { \ell + 2 } ^ { \prime }$ to $r _ { \ell + 1 } ^ { \prime } \cdot b _ { \ell + 2 }$ . Therefore, it holds that $r _ { \ell + 1 } ^ { \prime } \cdot a _ { \ell + 2 } ^ { \prime } \cdot z _ { \ell + 1 } \sim _ { R } \sigma _ { 1 } ( r _ { \ell + 1 } ^ { \prime } .$ $a _ { \ell + 2 } ^ { \prime } \cdot z _ { \ell + 1 } \big ) = r _ { \ell + 1 } ^ { \prime } \cdot b _ { \ell + 2 } \cdot \sigma _ { 1 } ( z _ { \ell + 1 } )$

Since $r _ { i } ^ { \prime } a _ { i + 1 } ^ { \prime } z _ { i } \sim _ { R } r _ { i } ^ { \prime } b _ { i + 1 } \sigma _ { 1 } ( z _ { i } )$ for all $0 \leq i < d .$ , we have that $u _ { i } ^ { \prime } a _ { i + 1 } ^ { \prime } z _ { i } \in L$ if and only i $\mathrm { f } , u _ { i } ^ { \prime } b _ { i + 1 } \sigma _ { 1 } ( z _ { i } ) \in L$ . The reason is the following. Assume that is the minimal well-typed DRA of L defined by $\cong _ { L }$ . After reading $u _ { i } ^ { \prime } , A$ would reach a configuration $( q , r )$ where $r = r _ { i } ^ { \prime } .$ . Since $r _ { i } ^ { \prime } a _ { i + 1 } ^ { \prime } z _ { i } \sim _ { R } r _ { i } ^ { \prime } b _ { i + 1 } \sigma _ { 1 } ( z _ { i } )$ , if from $( q , r )$ $a _ { i + 1 } ^ { \prime } z _ { i }$ is accepted, also accepts $b _ { i + 1 } \sigma _ { 1 } ( z _ { i } )$

$$
u _ { i } ^ { \prime } = u _ { i }
$$

$$
r _ { i } ^ { \prime } a _ { i + 1 } ^ { \prime } z _ { i } \sim _ { R } r _ { i } w [ i + 1 \cdot \cdot \cdot ]
$$

$$
0 \leq i < d
$$

For the base case, obviously, when $i = 0$ , we have $u _ { 0 } ^ { \prime } = u _ { 0 } = \varepsilon$ . Moreover, $a _ { 1 } ^ { \prime } z _ { 1 } = w$ . Since $r _ { 0 } ^ { \prime } = r _ { 0 } = M ( u _ { 0 } ^ { \prime } ) = \varepsilon .$ , the claim holds.

Now, for the induction step, assume that $u _ { \ell } ^ { \prime } = u _ { \ell } , r _ { \ell } ^ { \prime } { \cdot } a _ { \ell + 1 } ^ { \prime } z _ { \ell } \sim _ { R } r _ { \ell } { \cdot } w [ \ell + 1 { \cdot } { \cdot } \cdot ]$ We want to prove that the claim also holds at $\ell { + 1 }$ . Let $\left( u _ { \ell } , M ( u _ { \ell } ) { \cdot } a _ { \ell + 1 } , E _ { \ell } , u _ { \ell + 1 } \right)$ be the transition taken in  when reading $a _ { \ell + 1 } ~ = ~ w [ \ell + 1 ]$ at $( u _ { \ell } , r _ { \ell } )$ and $( u _ { \ell } ^ { \prime } , M ( u _ { \ell } ^ { \prime } ) \cdot b _ { \ell + 1 } , E _ { \ell } ^ { \prime } , u _ { \ell + 1 } ^ { \prime } )$ be the transition taken in Algorithm 1 when reading $a _ { \ell + 1 } ^ { \prime } \mathrm { a t } ( u _ { \ell } ^ { \prime } , r _ { \ell } ^ { \prime } )$ . By induction hypothesis, we know that $u _ { \ell } = u _ { \ell } ^ { \prime }$ and $r _ { \ell } ^ { \prime } a _ { \ell + 1 } ^ { \prime } z _ { \ell } \sim _ { R }$ $r _ { \ell } \cdot w [ \ell + 1 \cdot \cdot \cdot ]$ . It follows that $r _ { \ell } ^ { \prime } \cdot a _ { \ell + 1 } ^ { \prime } \sim _ { R } r _ { \ell } \cdot w [ \ell + 1 ] = r _ { \ell } \cdot a _ { \ell + 1 }$ (the prefixes also have the same type). On the other hand, $M ( u _ { \ell } ) \cdot a _ { \ell + 1 } = r _ { \ell } \cdot a _ { \ell + 1 } \sim _ { R } r _ { \ell } ^ { \prime } \cdot a _ { \ell + 1 } ^ { \prime } \sim _ { R }$ $r _ { \ell } ^ { \prime } \cdot b _ { \ell + 1 }$ hold. So, $M ( u _ { \ell } ) \cdot a _ { \ell + 1 } \sim _ { R } M ( u _ { \ell } ^ { \prime } ) \cdot b _ { \ell + 1 }$ . We then know that the transition $( u _ { \ell } , M ( u _ { \ell } ) \cdot a _ { \ell + 1 } , E _ { \ell } , u _ { \ell + 1 } )$ and the transition $( u _ { \ell } ^ { \prime } , M ( u _ { \ell } ^ { \prime } ) \cdot b _ { \ell + 1 } , E _ { \ell } ^ { \prime } , u _ { \ell + 1 } ^ { \prime } )$ are the same transition since $u _ { \ell } ^ { \prime } = u _ { \ell }$ and $\mathcal { H }$ is deterministic. Therefore, $u _ { \ell + 1 } = u _ { \ell + 1 } ^ { \prime } . $ In fact, by Claim (ii), we have $\begin{array} { r } { r _ { \ell } ^ { \prime } b _ { \ell + 1 } \sigma _ { 1 } ( z _ { \ell } ) \sim _ { R } r _ { \ell } ^ { \prime } \cdot a _ { \ell + 1 } ^ { \prime } \cdot z _ { \ell } } \end{array}$ . Since $r _ { \ell } ^ { \prime } \cdot a _ { \ell + 1 } ^ { \prime } z _ { \ell } \stackrel { . } { \sim } _ { R }$ $r _ { \ell } \cdot w [ \ell + 1 \cdot \cdot \cdot ]$ , we also have that $r _ { \ell } ^ { \prime } b _ { \ell + 1 } \sigma _ { 1 } ( z _ { \ell } ) \sim _ { R } r _ { \ell } \cdot w [ \ell + 1 \cdot \cdot \cdot ]$ . By removing the letters in the same positions from $r _ { \ell } ^ { \prime } \cdot b _ { \ell + 1 }$ and $r _ { \ell } \cdot a _ { \ell + 1 }$ , we know that $M ( r _ { \ell } ^ { \prime } \cdot b _ { \ell + 1 } ) \cdot \sigma _ { 1 } ( z _ { \ell } ) \sim _ { R } M ( r _ { \ell } \cdot a _ { \ell + 1 } ) \cdot w [ \ell + 2 \cdot \cdot \cdot ]$ . Further, we have $M ( \boldsymbol { r } _ { \ell } ^ { \prime } \cdot \boldsymbol { b } _ { \ell + 1 } )$ $\sigma _ { 1 } ( z _ { \ell } ) \sim _ { R } \sigma _ { 2 } ( M ( r _ { \ell } ^ { \prime } \cdot b _ { \ell + 1 } ) \cdot \sigma _ { 1 } ( z _ { \ell } ) ) = M ( u _ { \ell + 1 } ^ { \prime } ) \cdot \sigma _ { 2 } ( \sigma _ { 1 } ( z _ { \ell } ) ) = r _ { \ell + 1 } ^ { \prime } \cdot a _ { \ell + 2 } ^ { \prime } z _ { \ell + 1 }$ and $M ( r _ { \ell } \cdot a _ { \ell + 1 } ) \cdot w [ \ell + 2 \cdot \cdot \cdot ] = r _ { \ell + 1 } \cdot w [ \ell + 2 \cdot \cdot \cdot ]$ . Therefore, it holds that $u _ { i } ^ { \prime } = u _ { i }$ and $r _ { i } ^ { \prime } a _ { i + 1 } ^ { \prime } z _ { i } \sim _ { R } r _ { i } \cdot w [ i + 1 ]$ for all $0 \leq i < d .$

With similar argument, we can also prove that $u _ { d } ^ { \prime } = u _ { d }$ . Therefore, we have completed the proof. □

Lemma 15. Let w be the counterexample provided by the teacher. Algorithm 1 terminates after at most w iterations and returns a sufix v such that ub $\in X$ is not $S { \uplus \{ \ v \ v \{ v \} } $ -consistent with a word $u ^ { \prime } \in U$ with which ub is currently S-consistent.

Proof. By Lemma 27, the state sequence over w and that in Algorithm 1 coincide. Therefore, for the last state $u _ { d }$ , we have $\mathsf { M Q } ( u _ { d } ) \ne m q .$ Therefore, the membership results must flip at some point before reaching the last state $u _ { d }$ and the if condition will be satisfied. Hence, Algorithm 1 terminates after at most w iterations.

Assume it returns at the ith iteration with $v = \sigma _ { 2 } ( \sigma _ { 1 } ( z _ { i - 1 } ) )$ and some $u _ { i } ^ { \prime }$ such that

$$
\mathsf { M Q } ( u _ { i } ^ { \prime } \cdot v ) \neq \mathsf { M Q } ( u _ { i - 1 } ^ { \prime } \cdot a _ { i } ^ { \prime } z _ { i - 1 } ) ,
$$

where the one-letter extension $u _ { i - 1 } ^ { \prime } b _ { i }$ is currently S-consistent with $u _ { i } ^ { \prime }$ . Since

$$
\sigma _ { 2 } ( \sigma _ { 1 } ( M ( u _ { i - 1 } ^ { \prime } a _ { i } ^ { \prime } ) ) ) = \sigma _ { 2 } ( M ( u _ { i - 1 } ^ { \prime } b _ { i } ) ) = M ( u _ { i } ^ { \prime } ) ,
$$

we obtain

$$
\sigma _ { 2 } ( \sigma _ { 1 } ( M ( u _ { i - 1 } ^ { \prime } a _ { i } ^ { \prime } ) ) ) \cdot \sigma _ { 2 } ( \sigma _ { 1 } ( z _ { i - 1 } ) ) = M ( u _ { i } ^ { \prime } ) \cdot v .
$$

Moreover, by proof of Lemma $2 7 .$

$$
u _ { i - 1 } ^ { \prime } a _ { i } ^ { \prime } z _ { i - 1 } \in L \iff u _ { i - 1 } ^ { \prime } b _ { i } \sigma _ { 1 } ( z _ { i - 1 } ) \in L ,
$$

and hence,

$$
\mathsf { M Q } ( u _ { i - 1 } ^ { \prime } a _ { i } ^ { \prime } z _ { i - 1 } ) = \mathsf { M Q } ( u _ { i - 1 } ^ { \prime } b _ { i } \sigma _ { 1 } ( z _ { i - 1 } ) ) = \mathsf { M Q } ( \sigma _ { 2 } ( u _ { i - 1 } ^ { \prime } b _ { i } ) \cdot v )
$$

Thus, $\mathsf { M Q } ( \sigma _ { 2 } ( u _ { i - 1 } ^ { \prime } b _ { i } ) \cdot v ) \neq \mathsf { M Q } ( u _ { i } ^ { \prime } \cdot v )$ , and $u _ { i - 1 } ^ { \prime } b _ { i }$ is not $S  \{ v \}$ -consistent with $u _ { i } ^ { \prime }$ , witnessed by the word v and the mapping $\sigma _ { 2 }$ □

We now provide the missing proof for Theorem 16.

Theorem 16. Let  be a complete canonical minimal DRA of L with n states and m transitions defined under $\cong _ { L }$ , and let d be the maximum length of any counterexample returned by the teacher. It holds that:

– Algorithm 2 terminates and returns a DRA  that is isomorphic to .

– Algorithm 2 terminates after at most $m \times ( n - 1 )$ equivalence queries.

– Algorithm 2 requires m memorability queries and $\mathcal { O } ( d \times m \times n + m ^ { 3 } \times n ^ { 3 } )$ membership queries.

Proof. When Algorithm 1 returns a distinguishing sufix v separating ub from the state $u ^ { \prime }$ with which ub is currently S-consistent, we add v to the sufix set S to correct the transition from u over b. Consequently, ub becomes not $S \uplus \{ v \} .$ consistent with $u ^ { \prime } .$ Each transition can be corrected at most $n - 1$ times; that is, the target state of a transition over b from u may be misclassified no more than $n - 1$ times, as once ub is inconsistent with a state $u ^ { \prime \prime }$ , it remains inconsistent with $u ^ { \prime \prime }$ thereafter by Lemma 25. Each counterexample will correct at least one transition. Hence, the learner asks at most $m \cdot \left( n - 1 \right)$ equivalence queries in total. It is worth noting that in our experiments, new states will usually be discovered after adding the sufix returned from the counterexample analysis procedure.

For memorability queries, we only need it for every outgoing transition. The reason is as follows: first, for the initial state ε, $M ( \varepsilon )$ is trivially ε, so there is no need to ask memorability queries for the empty word. The only place we need to ask memorability queries is when we obtain a new one-letter extension and add it to the extension set X. (All new states come from X, so their memorable words have been queried when first added into X.) Since the number of all one-letter extensions is equivalent to the number of transitions, the number of memorability queries is m. So, the learner needs m memorability queries.

For membership queries, we need (1) $\mathcal { O } ( n \times n \times m )$ queries to fill the table entries since there are at most m n counterexamples/sufixes, (2) $\mathcal { O } ( d \times n \times m )$ membership queries to analyze the counterexamples where each analysis costs at most d membership queries and we need to analyze counterexamples at most m n times, and $( 3 ) \bar { \mathcal { O } } ( \bar { m ^ { 3 } } \times n ^ { 3 } )$ membership queries to look for the S-consistency class for each extension in X when constructing the conjectured DRA where $| X | \le m$ , each transition will cost at most $n \times n \times m$ times during a DRA construction with n being the maximum row number and $m \times n$ being maximum column number and we need to construct conjectured DRAs at most m n times. Therefore, we need in total $\mathcal { O } ( d \times m \times n + \dot { m } ^ { 3 } \times n ^ { 3 } )$ membership queries.

## E Preliminary Experiments

We implemented the proposed active learning algorithm in Python<sup>7</sup> and evaluated it on all examples presented in this paper, as well as on randomly generated DRAs. All experiments were conducted on a machine running Ubuntu 24.04 LTS, equipped with an Intel i7-11800H CPU and 16 GB of RAM.

Query Complexity. Due to space constraints, we report results only for the family of DRAs introduced in Example $2 ,$ namely the languages $L _ { n }$ for $n = 1 , \ldots , 2 5$ . The minimal DRA recognizing $L _ { n }$ has three states and three transitions when $n = 1$ , and 2n states and 6n 6 transitions when $n > 1$ . As shown in Fig. 5, the numbers of all query types grow polynomially with the size of the target DRA (measured by 2n states and 6n 6 transitions). Among them, the number of MQs grows the fastest, followed by EQs and then Mem queries, in line with the bounds in Theorem 16. Additional results are reported in Section F. Evaluation of RaLib and our approach. We also conducted experiments on $L _ { n }$ using RaLib [10,12], using the iosimulator backend and default settings on $L _ { n }$ for $2 \leq n \leq 1 0$ , without imposing a time limit. However, a direct comparison is not straightforward: ${ \sf R a l i b } ^ { 8 }$ targets register extended finite state machines (EFSMs) [12], which incorporate both inputs and outputs, whereas our model does not support outputs. Moreover, EFSMs characterize safety languages and do not explicitly distinguish between accepting and rejecting states.

![](images/61e5daf146cf838219c429aba69df7357ea2c42bc134ae23ff36eef792dbc01d.jpg)

![](images/39ebdcb303e52e83db0b1b474b3a2733136d534b14d814304486ec4189588291.jpg)  
Fig. 5: The x-axis is the the length of word $n = 1 , \ldots , 2 5$ for $L _ { n } .$ . Left: the number of EQs and Mems for each n. Right: the number of ${ \sf M Q s }$

To enable comparison, we translate our register automata into EFSMs by introducing outputs: OOK for transitions that can reach accepting states and ONO for transitions reaching a rejecting sink. Since EFSMs separate input and output states, this translation increases the number of states, as reflected in Table 1, where the input models of RaLib are larger. Thus, the comparison with RaLib is not meant as a direct performance evaluation, but as a demonstration that our approach efectively learns models in a complementary setting. While RaLib learns EFSMs using tree queries (TQs) and approximate equivalence checking, our method uses word-based membership queries and exact equivalence checking.

Table 1 summarizes the results. RaLib requires fewer TQs because tree queries compactly encode multiple words, whereas our method operates on individual words for MQs. As a result, our total runtime is dominated by MQs, while RaLib spends more time on EQs. Despite the larger number of MQs, our approach scales predictably with the size of the target model, as also reflected by the linear growth in memorability (Mem) queries. Furthermore, RaLib employs approximate equivalence checking with probabilistic guarantees, resulting in a small and constant number of EQs. Consequently, the learned models produced by

Table 1: Experimental results for RaLib and our approach on $L _ { n } .$ . DRA sizes are reported as numbers of states, and runtimes are given in seconds. All runtimes are total operation times.
<table><tr><td rowspan="2">Model</td><td colspan="4">Input DRA size Learned DRA size</td><td colspan="2">#TQ/MQ</td><td colspan="2">#EQ</td><td colspan="2">TQ/MQ Time</td><td colspan="2">(s) )EQ Time (s)</td><td colspan="2">Our Mem</td></tr><tr><td>RaLib</td><td>Ours</td><td>RaLib</td><td>Ours</td><td>RaLib</td><td>Ours</td><td>RaLib Ours</td><td></td><td>RaLib</td><td>Ours</td><td>RaLib</td><td>Ours</td><td>#Mem Time</td><td></td></tr><tr><td> $L _ { 2 }$ </td><td>9</td><td>4</td><td>8</td><td>4</td><td>34</td><td>39</td><td>2</td><td>2</td><td>&lt;1</td><td>&lt;1</td><td>2.19</td><td>&lt;1</td><td></td><td>&lt;1</td></tr><tr><td> $L _ { 3 }$ </td><td>16</td><td>6</td><td>10</td><td>6</td><td>49</td><td>156</td><td>2</td><td>4</td><td>&lt;1</td><td>&lt;1</td><td>2.57</td><td>&lt;1</td><td>12</td><td>&lt;1</td></tr><tr><td> $L _ { 4 }$ </td><td>22</td><td>8</td><td>12</td><td>8</td><td>68</td><td>662</td><td>2</td><td>8</td><td>&lt;1</td><td>&lt;1</td><td>2.91</td><td>&lt;1</td><td>18</td><td>&lt;1</td></tr><tr><td> $L _ { 5 }$ </td><td>28</td><td>10</td><td>14</td><td>10</td><td>91</td><td>2,173</td><td>2</td><td>14</td><td>&lt;1</td><td>&lt;1</td><td>3.07</td><td>&lt;1</td><td>24</td><td>&lt;1</td></tr><tr><td> $L _ { 6 }$ </td><td>34</td><td>12</td><td>16</td><td>12</td><td>118</td><td>5,885</td><td>2</td><td>22</td><td>&lt;1</td><td>&lt;1</td><td>3.37</td><td>&lt;1</td><td>30</td><td>&lt;1</td></tr><tr><td> $L _ { 7 }$ </td><td>40</td><td>14</td><td>18</td><td>14</td><td></td><td>149 13,653</td><td>2</td><td>32</td><td>&lt;1</td><td>&lt;1</td><td>3.59</td><td>&lt;1</td><td>36</td><td>&lt;1</td></tr><tr><td> $L _ { 8 }$ </td><td>46</td><td>16</td><td>20</td><td>16</td><td></td><td>184 28,192</td><td>2</td><td>44</td><td>&lt;1</td><td>1.25</td><td>3.76</td><td>&lt;1</td><td>42</td><td>&lt;1</td></tr><tr><td> $L _ { 9 }$ </td><td>52</td><td>18</td><td>22</td><td>18</td><td></td><td>223 53,312</td><td>2</td><td>58</td><td>&lt;1</td><td>2.58</td><td>4.04</td><td>&lt;1</td><td>48</td><td>&lt;1</td></tr><tr><td> $L _ { 1 0 }$ </td><td>58</td><td>20</td><td>24</td><td>20</td><td></td><td>266 94,412</td><td>2</td><td>74</td><td>&lt;1</td><td>5.11</td><td>3.70</td><td>&lt;1</td><td>54</td><td>&lt;1</td></tr></table>

RaLib do not always accept exactly the same language as the target DRAs, as observed in our experiments. In contrast, our exact equivalence checking ensures correctness but requires more EQs as the model size increases. Moreover, the well-typed register model used in our work appears to enable fast equivalence checking in practice.

We also considered register benchmarks from the RaLib GitHub repository, which contains a collection of models related to those in the Automata Wiki [31]<sup>9</sup>. However, to the best of our knowledge, there is no established translation from EFSMs to our register automata model. To adapt these benchmarks, we combined each EFSM input transition and its corresponding output transition into a single transition. Since our model does not support multiple parameters as in EFSMs, we instead introduced additional states to process parameters sequentially. All states were made accepting. Nevertheless, certain EFSM constructs remain challenging to encode. For example, a state may have transitions such as $\mathtt { I \_ g e t } ( \mathtt { p } )$ and $\mathtt { I \_ p u t } ( \mathtt { p } )$ , where I\_get and I\_put are input symbols and $p$ is a parameter. In our translation, input action names are abstracted away, which prevents distinguishing such cases based solely on parameters. As a result, this may introduce nondeterminism in the translated models. After excluding nondeterministic instances, we observed that, with few exceptions such as the alternative bit protocol and password models, the models learned by our tool often accept universal languages and thus collapse to a single state. As these cases provide limited insight for evaluation, we omit them from our experimental results.

## F Additional Experiments

The Family of Languages $L _ { n } .$ . In addition to the figures in the main paper, we report in the following table the number of equivalence queries (EQ), membership queries (MQ) and memorability queries. We also report the total time, which

includes the time for the teacher to solve the three types of queries and the total membership query resolving time to learn each automaton. We can see that the time consumed by resolving membership queries in the teacher dominates runtime in the whole procedure for learning the target automata.
<table><tr><td>n</td><td>States</td><td>Transitions</td><td>EQ</td><td>MQ</td><td>Mem</td><td>MQ time (s)</td><td>Total Time (s)</td></tr><tr><td>1</td><td>3</td><td>3</td><td>2</td><td>24</td><td>3</td><td>&lt;1</td><td>&lt;1</td></tr><tr><td>2</td><td>4</td><td>6</td><td>2</td><td>39</td><td>6</td><td>&lt;1</td><td>&lt;1</td></tr><tr><td>3</td><td>6</td><td>12</td><td>4</td><td>156</td><td>12</td><td>&lt;1</td><td>&lt;1</td></tr><tr><td>4</td><td>8</td><td>18</td><td>8</td><td>662</td><td>18</td><td>&lt;1</td><td>&lt;1</td></tr><tr><td>5</td><td>10</td><td>24</td><td>14</td><td>2173</td><td>24</td><td>&lt;1</td><td>&lt;1</td></tr><tr><td>6</td><td>12</td><td>30</td><td>22</td><td>5885</td><td>30</td><td>&lt;1</td><td>&lt;1</td></tr><tr><td>7</td><td>14</td><td>36</td><td>32</td><td>13653</td><td>36</td><td>&lt;1</td><td>&lt;1</td></tr><tr><td>8</td><td>16</td><td>42</td><td>44</td><td>28192</td><td>42</td><td>1.25</td><td>1.67</td></tr><tr><td>9</td><td>18</td><td>48</td><td>58</td><td>53312</td><td>48</td><td>2.58</td><td>3.23</td></tr><tr><td>10</td><td>20</td><td>54</td><td>74</td><td>94412</td><td>54</td><td>5.11</td><td>6.08</td></tr><tr><td>11</td><td>22</td><td>60</td><td>92</td><td>157820</td><td>60</td><td>9.10</td><td>10.45</td></tr><tr><td>12</td><td>24</td><td>66</td><td>112</td><td>250559</td><td>66</td><td>15.39</td><td>17.32</td></tr><tr><td>13</td><td>26</td><td>72</td><td>134</td><td>384197</td><td>72</td><td>25.21</td><td>28.00</td></tr><tr><td>14</td><td>28</td><td>78</td><td>158</td><td>570135</td><td>78</td><td>38.99</td><td>42.70</td></tr><tr><td>15</td><td>30</td><td>84</td><td>184</td><td>822 663</td><td>84</td><td>59.74</td><td>64.97</td></tr><tr><td>16</td><td>32</td><td>90</td><td>212</td><td>1155 200</td><td>90</td><td>88.16</td><td>95.11</td></tr><tr><td>17</td><td>34</td><td>96</td><td>242</td><td>1 590 956</td><td>96</td><td>129.28</td><td>138.45</td></tr><tr><td>18</td><td>36</td><td>102</td><td>274</td><td>2148 216</td><td>102</td><td>181.08</td><td>193.26</td></tr><tr><td>19</td><td>38</td><td>108</td><td>308</td><td>2852 259</td><td>108</td><td>256.15</td><td>271.12</td></tr><tr><td>20</td><td>40</td><td>114</td><td>344</td><td>3730 170</td><td>114</td><td>347.37</td><td>368.70</td></tr><tr><td>21</td><td>42</td><td>120</td><td>382</td><td>4814609</td><td>120</td><td>473.87</td><td>499.51</td></tr><tr><td>22</td><td>44</td><td>126</td><td>422</td><td>6133 333</td><td>126</td><td>615.41</td><td>647.93</td></tr><tr><td>23</td><td>46</td><td>132</td><td>464</td><td>7 730 804</td><td>132</td><td>823.10</td><td>863.34</td></tr><tr><td>24</td><td>48</td><td>138</td><td>508</td><td>9644 939</td><td>138</td><td>1061.34</td><td>1111.84</td></tr><tr><td>25</td><td>50</td><td>144</td><td>554</td><td>11 922 628</td><td>144</td><td>1369.29</td><td>1429.55</td></tr></table>

The Language $L _ { m i d } .$ . Recall that $L _ { m i d } = \{ a _ { 1 } a _ { 2 } a _ { 3 } \mid a _ { 1 } < a _ { 3 } < a _ { 2 } \}$ over $( \mathbb { Z } , < )$ In our experiments, we learn $L _ { m i d }$ over $( \mathbb { R } , < )$ instead. The minimal well-typed DRA for $L _ { m i d }$ has 5 states and 11 transitions. Learning this automaton requires 2 EQs, 72 MQs, and 11 memorability queries. The final observation table is shown below.

The DRA learned is shown in Fig. 6. Notice that this figure is generated by our tool, and the position index in E starts from 0 rather than 1.

The Language $L _ { r e p }$ of Section 4.1. Recall that $L _ { r e p } = \{ x y x y \in \mathbb { Q } ^ { 4 } \mid x < y \}$ The minimal well-typed DRA for $L _ { r e p }$ has 6 states and 14 transitions. Learning this automaton requires 3 EQs, 156 MQs, and 14 memorability queries. The final observation table and the DRA are the same as the ones in Fig. 2.

![](images/6e900a31f38d07ed74af73ecde8ee8e11f6762ec8253c0774aa70ef294bac5ca.jpg)

![](images/53408c9111688eabcb6f92421ff86a062c0cfcb90d45548ebbcad5156c8425ec.jpg)  
Fig. 6: Learned DRA recognizing $L _ { m i d }$

Randomly Generated DRAs. We generated random minimal well-typed DRAs of sizes 5, 10, . . . , 50, with 50 automata for each size. We report in the following table the average (Avg) number of equivalence queries (EQ), membership queries (MQ), and memorability queries and the average time required. As can be seen in Fig. 7, similar to the trend for the family of languages $L _ { n } ,$ the number of MQs grows the fastest, compared to EQs and then Mem queries, in accordance with Theorem 16. The average time per automaton here includes the time for the teacher to solve the queries.

<table><tr><td>Size Avg #Transitions Avg #EQ</td><td></td><td>Avg #MQ Avg #Mem Avg Time (s)</td><td></td></tr><tr><td>5</td><td>12.2 2.3</td><td>75.0</td><td>12.2 0.12</td></tr><tr><td>10</td><td>34.6 4.4</td><td>593.1</td><td>34.6 0.28</td></tr><tr><td>15</td><td>57.4 6.8</td><td>2086.1</td><td>57.4 1.08</td></tr><tr><td>20 82.9</td><td>9.2</td><td>5131.1</td><td>82.9 2.02</td></tr><tr><td>25 105.8</td><td>11.9</td><td>10542.2</td><td>105.8 2.84</td></tr><tr><td>30 130.5</td><td>13.8</td><td>17883.5</td><td>130.5 5.33</td></tr><tr><td>35 157.4</td><td>15.3</td><td>25 949.8</td><td>157.4 8.03</td></tr><tr><td>40 181.4</td><td>18.9</td><td>43782.9</td><td>181.4 12.54</td></tr><tr><td>45 207.4</td><td>20.6</td><td>60 489.1</td><td>207.4 17.46</td></tr><tr><td>50 233.7</td><td>22.5</td><td>81 354.0</td><td>233.7 23.87</td></tr></table>

![](images/2de1ed07a5ff85863cfa409dea4d5a9c819f29944fcf29ab1f71a3dfbf722a87.jpg)

![](images/aaa405b559590be10a7eb89b900ae56c0c9785ca054b71077109616748943ad8.jpg)

![](images/2cfec590bff83cf645b8e4f747ddc7a6b37cf5de9b022c72609804657354542c.jpg)

![](images/a6d1d862fab53bb58c7a0d056e9d295db4cf9d1d6ff1682ec84c77721e0a990c.jpg)  
Fig. 7: Average statistics for learning random minimal well-typed DRAs.

## G Missing Proofs of Section 4.2

Language Intersection Non-emptiness and Language Inclusion. Given a k -DRA $\mathcal { A } _ { 1 } \ = \ ( Q ^ { 1 } , q _ { 0 } ^ { 1 } , F ^ { 1 } , \varDelta ^ { 1 } )$ and a $k _ { 2 } { \mathrm { - D R A } }$ $\mathcal { A } _ { 2 } ~ = ~ ( Q ^ { 2 } , q _ { 0 } ^ { 2 } , F ^ { 2 } , \varDelta ^ { 2 } )$ , the language intersection non-emptiness asks whether there is a word $w \in L ( \mathcal { A } _ { 1 } ) \cap$ $L ( A _ { 2 } )$ . The language inclusion problem asks whether $L ( \mathcal { A } _ { 1 } ) \subseteq L ( \mathcal { A } _ { 2 } )$ . This is equivalent to the language intersection emptiness problem, that is, whether $L ( \mathcal { A } _ { 1 } ) \cap \overline { { L ( \mathcal { A } _ { 2 } ) } } = \emptyset$ , where $\overline { { L ( \mathcal { A } _ { 2 } ) } }$ denotes the complement of $L ( A _ { 2 } )$ . Since $\boldsymbol { A } _ { 2 }$ is a DRA, it is straightforward to construct a DRA $B _ { 2 }$ accepting the complement language, that is, $L ( B _ { 2 } ) = \overline { { L ( \mathcal { A } _ { 2 } ) } }$ , by simply swapping the accepting and nonaccepting states of $\boldsymbol { A } _ { 2 }$

To solve the language intersection emptiness problem, we can use the product register automaton of Definition 22, which simulates the operations of both automata simultaneously. Recall that this product register automaton difers from the RAs in Definition 1: its registers may contain duplicate values and it is not well-typed. Nevertheless, language non-emptiness can be checked by simply testing whether there exists a path from the initial state $( q _ { 0 } ^ { 1 } , q _ { 0 } ^ { 2 } )$ to some accepting state in $F ^ { \times }$ . Although such a path may be exponentially long—bounded by $| Q _ { 1 } | \cdot | Q _ { 2 } | \cdot ( k _ { 1 } + k _ { 2 } ) !$ —we do not need to store the entire path. It sufices to maintain a counter together with the current configuration of the product automaton, both of which require only polynomial space. This yields a PSPACE decision procedure.

For the lower-bound, we reduce from the PSPACE-complete problem of deciding whether a linear-bounded deterministic Turing machine M accepts an input w. The reduction closely follows [29, Theorem 32], which considers DRAs that do not permit register permutations. The key is to encode the tape contents using DRA registers, which cannot store duplicate values, and to permute the register symbols while simulating the Turing machine.

Theorem 17. For DRAs over $( \Sigma , = )$ and over $( \Sigma , < )$ , the language intersection non-emptiness problem is PSPACE-complete. Consequently, the language inclusion problem is also PSPACE-complete.

Proof. Since DRAs can be easily complemented, it sufices to show that the intersection non-emptiness problem for DRAs is PSPACE-complete. The complement problem—the intersection emptiness problem—is also PSPACE-complete. Consequently, the DRA language inclusion problem, which is equivalent to intersection emptiness, is PSPACE-complete as well.

Inclusion in PSPACE. Given a $k _ { 1 } \mathrm { - D R A } \mathcal { A } _ { 1 } = ( Q ^ { 1 } , q _ { 0 } ^ { 1 } , F ^ { 1 } , \varDelta ^ { 1 } )$ and a $k _ { \mathrm { { 2 } ^ { - } } } \mathrm { { D R A } }$ $\mathcal { A } _ { 2 } = ( Q ^ { 2 } , q _ { 0 } ^ { 2 } , F ^ { 2 } , \varDelta ^ { 2 } )$ , we ask whether whether there is a word $w \in L ( \mathcal { A } _ { 1 } ) \cap$ $L ( A _ { 2 } )$ . Equivalently, it asks whether there is a run from the initial state $( q _ { 0 } ^ { 1 } , q _ { 0 } ^ { 2 } )$ to a final state in $F ^ { \times }$ in the product automaton $\mathcal { A } _ { 1 } \times \mathcal { A } _ { 2 }$

We now consider the product automaton $A _ { 1 } \times A _ { 2 }$ . We show that if there is a run from the initial state $( q _ { 0 } ^ { 1 } , q _ { 0 } ^ { 2 } )$ to a final state in $F ^ { \times }$ , then there exists a run of length at most $| Q _ { 1 } | \cdot | Q _ { 2 } | \cdot ( { \dot { k } } _ { 1 } + k _ { 2 } ) !$ reaching a state in $F ^ { \times }$

Based on this observation, we obtain a nondeterministic polynomial-space algorithm that guesses the input word symbol by symbol. The algorithm keeps only a counter and the current configuration of the product automaton, and therefore uses polynomial space. By Savitch’s theorem, PSPACE = NPSPACE, hence the intersection non-emptiness problem for DRAs is in PSPACE.

At a configuration $( q ^ { 1 } , q ^ { 2 } , x ^ { 1 } , x ^ { 2 } )$ , the algorithm guesses a symbol a. The number of possible choices for a is bounded by

$$
2 ( | x ^ { 1 } | + | x ^ { 2 } | ) + 1 \ \leq \ 2 ( k _ { 1 } + k _ { 2 } ) + 1 .
$$

To justify this, we reorder the concatenation $x ^ { 1 } x ^ { 2 }$ into non-decreasing order and retain only the distinct symbols, denoting the resulting string by $x = x _ { 1 } \cdot \cdot \cdot x _ { \ell }$ The guessed symbol a may then be chosen as:

$- \textit { x } _ { i }$ for any $1 \leq i \leq \ell ,$

a symbol greater than x<sub>ℓ</sub> or smaller than $x _ { 1 }$ , or

a symbol strictly between $x _ { i }$ and $x _ { i + 1 }$ for any $1 \leq i \leq \ell - 1$

After guessing $^ { a , }$ the algorithm transitions to the next configuration and increments the counter. It starts with the initial configuration $( q _ { 0 } ^ { 1 } , q _ { 0 } ^ { 2 } , \varepsilon , \varepsilon )$ with the counter set to 0, and proceeds as follows:

– If the counter has not yet reached $| Q _ { 1 } | \cdot | Q _ { 2 } | \cdot ( k _ { 1 } + k _ { 2 } ) !$ and a final state is reached, it accepts.

– If the counter has not reached this bound and no final state has been reached, it continues by guessing another symbol.

– If the counter reaches $| Q _ { 1 } | \cdot | Q _ { 2 } | \cdot ( k _ { 1 } + k _ { 2 } ) !$ ! without reaching a final state, it rejects.

It remains to show that $| Q _ { 1 } | \cdot | Q _ { 2 } | \cdot ( k _ { 1 } + k _ { 2 } ) !$ is an upper bound on the length of a shortest run reaching a final state. Assume that there is a run in the product automaton from the initial state $( q _ { 0 } ^ { 1 } , q _ { 0 } ^ { 2 } )$ to a final state in $F ^ { \times }$ over a word u of length ℓ. Let the run be

$$
\rho _ { 0 } = ( q _ { 0 } ^ { 1 } , q _ { 0 } ^ { 2 } , \varepsilon , \varepsilon ) \stackrel { u _ { 1 } } { \longrightarrow } \rho _ { 1 } = ( q _ { 1 } ^ { 1 } , q _ { 1 } ^ { 2 } , x _ { 1 } ^ { 1 } , x _ { 1 } ^ { 2 } ) \stackrel { u _ { 2 } } { \longrightarrow } \cdot \cdot \cdot \stackrel { u _ { \ell } } { \longrightarrow } \rho _ { \ell } = ( q _ { \ell } ^ { 1 } , q _ { \ell } ^ { 2 } , x _ { \ell } ^ { 1 } , x _ { \ell } ^ { 2 } ) ,
$$

where $( q _ { \ell } ^ { 1 } , q _ { \ell } ^ { 2 } ) \in F ^ { \times }$

We say that two configurations

$$
\rho _ { i } = ( q _ { i } ^ { 1 } , q _ { i } ^ { 2 } , x _ { i } ^ { 1 } , x _ { i } ^ { 2 } ) \qquad \mathrm { a n d } \qquad \rho _ { j } = ( q _ { j } ^ { 1 } , q _ { j } ^ { 2 } , x _ { j } ^ { 1 } , x _ { j } ^ { 2 } )
$$

satisfy $\rho _ { i } \approx \rho _ { j }$ if and only if $( q _ { i } ^ { 1 } , q _ { i } ^ { 2 } ) = ( q _ { j } ^ { 1 } , q _ { j } ^ { 2 } )$ and $x _ { i } ^ { 1 } x _ { i } ^ { 2 } \sim _ { R } x _ { j } ^ { 1 } x _ { j } ^ { 2 } .$

Let $\ell \geq | Q _ { 1 } | \cdot | Q _ { 2 } | \cdot ( k _ { 1 } + k _ { 2 } ) !$ . By the pigeonhole principle, there exist two distinct configurations along the run such that $\rho _ { i } \approx \rho _ { j }$ . Fix such a pair with $i < j$ so that no earlier–later pair $\rho _ { k } , \rho _ { k ^ { \prime } }$ with $\rho _ { k } \approx \rho _ { k ^ { \prime } }$ satisfies either $k \leq i < j < k ^ { \prime }$ or $k < i \leq j \leq k ^ { \prime }$

We now construct a new run on a new word that still reaches a final state in $F ^ { \times }$ , but in which no two configurations are -equivalent. This new run will therefore have length at most $| Q _ { 1 } | \cdot | Q _ { 2 } | \cdot ( k _ { 1 } + k _ { 2 } ) !$

The idea is to modify the run (and word) from configuration $\rho _ { i }$ onward so that the k-th configuration after $\rho _ { i }$ is $\rho _ { j + k } ^ { \prime }$ with $\rho _ { j + k } ^ { \prime } \approx \rho _ { j + k }$ for all $0 \leq k \leq ( \ell - j )$ . Thus the new run is of the form

$$
\rho _ { 0 } \cdots \cdot \cdot \rho _ { i } = \rho _ { j } ^ { \prime } { \xrightarrow { u _ { j + 1 } ^ { \prime } } } \rho _ { j + 1 } ^ { \prime } { \xrightarrow { u _ { j + 2 } ^ { \prime } } } \cdot \cdot \cdot { \xrightarrow { u _ { \ell } ^ { \prime } } } \rho _ { \ell } ^ { \prime } .
$$

It remains to show the following key property: for any transition $\rho _ { i } \stackrel { a } {  } \rho _ { i + 1 }$ 1 and any configuration $\rho _ { j } \approx \rho _ { i }$ , there exists a symbol $a ^ { \prime }$ such that $\rho _ { j } \stackrel { a ^ { \prime } } { \longrightarrow } \rho _ { j + 1 }$ and $\rho _ { j + 1 } \approx \rho _ { i + 1 }$ . Let $\rho _ { i } = \bar { ( } q _ { i } ^ { 1 } , q _ { i } ^ { 2 } , x _ { i } ^ { 1 } , x _ { i } ^ { 2 } { ) }$ and $\rho _ { j } = ( q _ { j } ^ { 1 } , q _ { j } ^ { 2 } , x _ { j } ^ { 1 } , x _ { j } ^ { 2 } )$ . We only need to pick a symbol $a ^ { \prime }$ such that $x _ { i } ^ { 1 } x _ { i } ^ { 2 } a \ \sim _ { R } \ x _ { j } ^ { 1 } x _ { j } ^ { 2 } a ^ { \prime }$ , which is always possible because $x _ { i } ^ { 1 } x _ { i } ^ { 2 } \ \sim _ { R } \ x _ { j } ^ { 1 } x _ { j } ^ { 2 }$ and the data domain is dense.

PSPACE-Hardness. For PSPACE-hardness, we reduce from the PSPACEcomplete problem of determining whether a linear bounded deterministic Turing machine M accepts an input w (see, for example, the linear bounded prediction problem in [21]).

More precisely, we consider a Turing machine with binary alphabet and linearly bounded tape $M = \langle Q , q _ { 0 } , \delta , q _ { a c c } \rangle$ such that Q is a finite set of states, $q _ { 0 } \in Q$ is the initial state, $q _ { a c c } \in Q$ is the accepting state, and $\delta : Q \times \{ 0 , 1 \} $

$Q \times \{ 0 , 1 \} \times \{ - 1 , 1 \}$ is a transition function. A configuration of M is a triple $\langle q , i , w \rangle$ where $q \in Q$ is the machine state, $0 \leq i < | M |$ is the head position, and $\dot { w } \in \{ 0 , 1 \} ^ { | M | }$ is the tape content. Let $\delta ( q , w ( i ) ) = \langle q ^ { \prime } , b , j \rangle$ $\mathrm { ~ I f ~ } 1 \le i + j \le | M |$ 2 then the state $\langle q ^ { \prime } , i + j , w [ i  b ] \rangle$ is the unique successor of $\langle q , i , w \rangle$ . The PSPACE-hard problem asks whether M accepts a word $w _ { 0 } \in \{ 0 , 1 \} ^ { | M | }$ , that is, whether $\langle q _ { 0 } , 1 , w _ { 0 } \rangle$ reaches a configuration with the accepting state $q _ { f }$

The main idea of the reduction is similar to the PSPACE-hardness proof of the inclusion problem of a diferent type of DRAs, the ones where permutations of registers in transitions are not allowed [29, Theorem 32]. Since the values in the register need to be distinct, we construct two $( | M | + 1 ) – \mathrm { D R A s }$ $A _ { 1 } , A _ { 2 }$ such that whenever they synchronize on a data word, their register assignments $\rho _ { 1 } , \rho _ { 2 }$ represent the content of M tape cells as follows: 0 in the i-th tape cell is represented by $\rho _ { 1 } ( i ) = \rho _ { 2 } ( i )$ , and 1 by $\rho _ { 1 } ( i ) \notin \rho _ { 2 }$ . One extra register plays a technical role that helps maintain the representation. The position of the head and state of the machine are maintained in the state of the automata. We show that M accepts w if and only if $\mathcal { A } _ { 1 }$ and $\boldsymbol { A } _ { 2 }$ both accept some data word $w ^ { \prime }$ (intersection non-emptiness).

There are two phases for the DRAs: the initialization phase and the simulation phase. In the initialization phase, the DRAs are initialized so that they jointly encode the initial configuration of the Turing machine M, with their combined register assignments representing the initial tape contents $w _ { 0 }$

Initialization Phase. In the initialization phase, both $\mathcal { A } _ { 1 }$ and $\boldsymbol { A } _ { 2 }$ read $2 | M |$ symbols, where the (2i 1)-th and 2i-th symbols jointly represent $w _ { 0 } ( i )$ for all $1 \leq i \leq | M | .$ . If $w _ { 0 } ( i ) = 0$ , both automata verify that these two symbols are identical but distinct from all previously seen symbols, and then store the symbol in their respective registers; otherwise, they transition to rejecting sink states. If $w _ { 0 } ( i ) = 1$ , the automata ensure that the two symbols are distinct from each other and from all previously seen symbols: $\mathcal { A } _ { 1 }$ stores the $( 2 i - 1 )$ -th symbol and $\boldsymbol { A } _ { 2 }$ stores the 2i-th symbol. Any violation leads to the rejecting sink.

Thus, whenever neither DRA reaches a rejecting sink, both automata end up in states with register assignments $\rho _ { 1 }$ and $\rho _ { 2 }$ that together encode the tape contents $w _ { 0 }$

Each DRA requires $2 | M |$ states for this phase, in addition to the initial and rejecting sink states. The extra register may be used to check correct initialization: $\mathcal { A } _ { 2 }$ first stores the (2i 1)-th symbol, and upon reading the 2i-th symbol, if it difers from all previously stored symbols (including the $( 2 i { - } 1 ) { \ - } \mathrm { { t h } } )$ , it updates its register accordingly—moving to a new non-sink state as $w _ { 0 } ( i ) = 1$ or rejecting otherwise. If the two symbols are equal and new, it interprets this as $w _ { 0 } ( i ) = 0 ;$ otherwise, it transitions to a rejecting sink. $\mathcal { A } _ { 1 }$ can use its extra register in a similar manner.

For example, when $w _ { 0 } = 0 1 1$ , both $\mathcal { A } _ { 1 }$ and $\boldsymbol { A } _ { 2 }$ can read the data word 001234 and end up in (non-sink) states with register assignments $\rho _ { 1 } = 0 1 3$ and $\rho _ { 2 } = 0 2 4$ respectively; the data words 001034 and 001134 are rejected by both DRAs and the data word 001232 is rejected by $\mathcal { A } _ { 2 }$

Simulation Phase. Once the DRAs are properly initialized, both are in the states representing the initial machine state $q _ { 0 }$ and head position 1 $( { \mathrm { i . e . } }$ , the head on the first tape cell), with register assignments $\rho _ { 1 }$ and $\rho _ { 2 }$ that together encode the initial tape contents $w _ { 0 }$ . At this point, they are ready to simulate the transitions of $M .$

Each transition $\delta ( q , w ( i ) ) = \langle q ^ { \prime } , b , j \rangle$ , that is, $\langle q , i , w \rangle \to \langle q ^ { \prime } , i + j , w [ i \mapsto b ] \rangle$ ， is simulated by at most $2 | M | + 2$ transitions. Starting at the states representing the machine state $q$ and head position i with register assignments $\rho _ { 1 }$ and $\rho _ { 2 }$ that jointly encode $w ,$ after a sequence of transitions the automata reach the states representing the machine state $q ^ { \prime }$ and head position $i + j$ , with register assignments $\rho _ { 1 } ^ { \prime }$ and $\rho _ { 2 } ^ { \prime }$ that jointly encode the updated tape contents $w [ i \mapsto b ]$

The idea is as follows. The automata read two symbols—the i-th symbol in $\mathcal { A } _ { 1 }$ and the i-th symbol in $A _ { 2 } - \mathrm { t o }$ determine whether $w ( i ) = 0$ or 1. Based on this information, each automaton simulates the corresponding transition of $M .$ If M does not rewrite $w ( i )$ , i.e. $b = w ( i )$ , both automata move to new states representing $q ^ { \prime }$ and head position $i + j$ without changing their register assignments, since $w [ i \mapsto b ] = w$

Otherwise, when $b \neq w ( i )$ , the symbol at position i must be updated. Since the register automata can only forget symbols or append new ones at the end of their registers, they first rearrange the registers so that the i-th position can be modified. They do this by reading $2 ( i - 1 )$ symbols— $\rho _ { 1 } ( 1 ) \cdot \cdot \cdot \rho _ { 1 } ( i - 1 )$ and $\rho _ { 2 } ( 1 ) \cdots \rho _ { 2 } ( i - 1 )$ —so that the registers are in a form where the i-th entry is ready to be replaced, reaching temporary assignments

$$
\begin{array} { r l } & { \rho _ { 1 } ( i ) \cdot \cdot \cdot \rho _ { 1 } ( | M | ) \rho _ { 1 } ( 1 ) \cdot \cdot \cdot \rho _ { 1 } ( i - 1 ) , \ \mathrm { a n d } } \\ & { \rho _ { 2 } ( i ) \cdot \cdot \cdot \rho _ { 2 } ( | M | ) \rho _ { 2 } ( 1 ) \cdot \cdot \cdot \rho _ { 2 } ( i - 1 ) . } \end{array}
$$

In the case $w ( i ) = 0$ and it needs to be updated to $b = 1$ , that is, $\rho _ { 1 } ( i ) =$ $\rho _ { 2 } ( i )$ and the registers must be updated to two diferent symbols a and $b ,$ the automata read and store two fresh symbols, reaching states with updated register assignments

$$
\begin{array} { r l } & { \rho _ { 1 } ( i + 1 ) \cdot \cdot \cdot \rho _ { 1 } ( | M | ) \rho _ { 1 } ( 1 ) \cdot \cdot \cdot \rho _ { 1 } ( i - 1 ) a , \mathrm { ~ a n d } } \\ & { \rho _ { 2 } ( i + 1 ) \cdot \cdot \cdot \rho _ { 2 } ( | M | ) \rho _ { 2 } ( 1 ) \cdot \cdot \cdot \rho _ { 2 } ( i - 1 ) b . } \end{array}
$$

Conversely, when $w ( i ) = 1$ and it needs to be updated to $b = 0$ , that is, $\rho _ { 1 } ( i ) \neq \rho _ { 2 } ( i )$ and both registers must be updated to the same symbol, the automata read a new symbol $a ,$ reaching states with updated register assignments

$$
\begin{array} { r l } & { \rho _ { 1 } ( i + 1 ) \cdot \cdot \cdot \rho _ { 1 } ( | M | ) \rho _ { 1 } ( 1 ) \cdot \cdot \cdot \rho _ { 1 } ( i - 1 ) a , \mathrm { ~ a n d ~ } } \\ & { \rho _ { 2 } ( i + 1 ) \cdot \cdot \cdot \rho _ { 2 } ( | M | ) \rho _ { 2 } ( 1 ) \cdot \cdot \cdot \rho _ { 2 } ( i - 1 ) a . } \end{array}
$$

Finally, another $2 ( | M | - i )$ transitions restore the correct register order, yielding

$$
\rho _ { 1 } ^ { \prime } = \rho _ { 1 } ( i \mapsto a ) \quad \mathrm { a n d ~ e i t h e r } \quad \rho _ { 2 } ^ { \prime } = \rho _ { 2 } ( i \mapsto a ) \mathrm { ~ o r ~ } \rho _ { 2 } ^ { \prime } = \rho _ { 2 } ( i \mapsto b ) .
$$

Both automata are in accepting states when they correspond to the machine accepting state $q _ { a c c } .$ It is easy to see that an accepting computation of $M$ on $w _ { 0 }$ corresponds to a data word accepted by both $\mathcal { A } _ { 1 }$ and $A _ { 2 } ,$ , and vice versa. Since both DRAs are of polynomial size in $M ,$ the reduction is in polynomial time. Moreover, the DRAs constructed can use identity tests only. □

Configuration Equivalence. Given a DRA over $( \Sigma , R )$ and a configuration $( q , v )$ of , we define the language of $\boldsymbol { \mathcal { A } } ^ { ( q , v ) }$ as $L ( \mathcal { A } ^ { ( q , v ) } ) = \{ w \in \Sigma ^ { * } \ \bar { | } \ \exists q _ { f } \in$ $F , u \in \Sigma ^ { * } . \ ( q , v ) \ { \stackrel { w } { \to } } _ { \cal A } \left( q _ { f } , u \right) \}$ . Given two DRAs $\mathcal { A } _ { 1 }$ and $\boldsymbol { A } _ { 2 }$ over $( \Sigma , R )$ and two configurations $( q _ { 1 } , v _ { 1 } )$ and $( q _ { 2 } , v _ { 2 } )$ of $\mathcal { A } _ { 1 }$ and $\boldsymbol { A } _ { 2 }$ respectively, the configuration equivalence problem asks whether $L ( \mathcal { A } _ { 1 } ^ { ( q _ { 1 } , v _ { 1 } ) } ) = L ( \mathcal { A } _ { 2 } ^ { ( q _ { 2 } , v _ { 2 } ) } )$

Language Equivalence. Given two DRAs $\mathcal { A } _ { 1 }$ and $\boldsymbol { A } _ { 2 }$ over $( \Sigma , R )$ , the language equivalence problem asks whether $L ( \mathcal { A } _ { 1 } ) = L ( \mathcal { A } _ { 2 } )$

The language equivalence problem is a special case of the configuration equivalence problem, that is, it is equivalent to the configuration equivalence problem whether $L ( \mathcal { A } _ { 1 } ^ { ( q _ { 0 } ^ { 1 } , \varepsilon ) } ) = L ( \mathcal { A } _ { 2 } ^ { ( q _ { 0 } ^ { 2 } , \varepsilon ) } )$ where $( q _ { 0 } ^ { 1 } , \varepsilon )$ and $( q _ { 0 } ^ { 2 } , \varepsilon )$ are the initial configurations of $\mathcal { A } _ { 1 }$ and $\boldsymbol { A } _ { 2 }$ respectively. Both problems are in PSPACE, by arguments analogous to the PSPACE upper-bound proof for language inclusion. Indeed, language equivalence reduces to checking the two inclusions $L ( \mathcal { A } _ { 1 } ) \subseteq L ( \mathcal { A } _ { 2 } )$ and $L ( \mathcal { A } _ { 2 } ) \subseteq L ( \mathcal { A } _ { 1 } )$ , which is then in PSPACE.

Theorem 18. Both the configuration equivalence problem and the language equivalence problem are in PSPACE for DRAs over dense domain $( \Sigma , < )$

Proof. It sufices to prove that the configuration equivalence problem is in PSPACE since the language equivalence problem is a special case of the configuration equivalence problem. Given two $\mathrm { D R A s \ } \mathcal { A } _ { 1 }$ and $\boldsymbol { A } _ { 2 }$ and configurations $( q _ { 1 } , v _ { 1 } )$ and $( q _ { 2 } , v _ { 2 } )$ of $\mathcal { A } _ { 1 }$ and ${ \cal A } _ { 2 } ,$ respectively, the configuration equivalence problem $L ( \mathcal { A } _ { 1 } ^ { ( q _ { 1 } , v _ { 1 } ) } ) \ : = \ : L ( \mathcal { A } _ { 2 } ^ { ( q _ { 2 } , v _ { 2 } ) } )$ reduces to checking two inclusions: $L ( \mathcal { A } _ { 1 } ^ { ( q _ { 1 } , v _ { 1 } ) } ) ~ \subseteq ~ L ( \mathcal { A } _ { 2 } ^ { ( q _ { 2 } , v _ { 2 } ) } )$ and $L ( \mathcal { A } _ { 1 } ^ { ( q _ { 1 } , v _ { 1 } ) } ) ~ \supseteq ~ L ( \mathcal { A } _ { 2 } ^ { ( q _ { 2 } , v _ { 2 } ) } )$ . Each inclusion further reduces to a language intersection emptiness check, assuming the DRAs start from the given initial configurations. For example, checking $L ( \mathcal { A } _ { 1 } ^ { ( q _ { 1 } , v _ { 1 } ) } ) \subseteq$ $L ( \mathcal { A } _ { 2 } ^ { ( q _ { 2 } , v _ { 2 } ) } )$ reduces to checking $L ( \mathcal { A } _ { 1 } ^ { ( q _ { 1 } , v _ { 1 } ) } ) \cap L ( \mathcal { B } _ { 2 } ^ { ( q _ { 2 } , v _ { 2 } ) } ) = \emptyset$ , where $B _ { 2 }$ is the complement of $\boldsymbol { A } _ { 2 }$ , obtained by swapping its accepting and non-accepting states. This intersection emptiness check is in PSPACE, similar to the case of DRAs without initial assignments. □

Memorability. Given a k-DRA over $( \Sigma , < )$ , a word $u \in \Sigma ^ { * }$ and $a \ \in \ \boldsymbol { u } .$ the memorability problem asks whether a is L-memorable in $u ,$ that is, whether $a \in m e m ( u )$

We first present a simpler version of Definition 4 for the case in which the alphabet Σ is dense. In Definition 4, the alphabet $\varSigma$ may be dense or non-dense. When Σ is non-dense, we may not find a symbol b and an R-preserving mapping $\sigma$ for u with $\sigma ( a ) = b$ and $u \sim _ { R } \sigma ( u )$ , hence the definition refers to another word of the same type. In contrast, for dense Σ, such a symbol b and mapping σ always exist. We simplify the definition of memorable symbol for dense alphabet as follows.

Lemma 28. Let L be a language over $( \Sigma , R )$ , where Σ is dense. A symbol $a \in \Sigma$ is L-memorable in a word $u \in \Sigma ^ { * } \mathrm { ~ } i f$ and only if there exist a symbol $b \in \Sigma$ , a word $v \in \Sigma ^ { * }$ and an R-preserving mapping σ with $\sigma ( a ) = b$ and $\sigma ( d ) = d$ for all other $d \in u \cdot v ,$ such that v is an L-distinguishing extension for u and $\sigma ( u )$ , that $i s ,$ either $u \cdot v \in L$ or $\sigma ( u ) \cdot v \in L$

Proof. First, we consider the only-if direction. Let $u ^ { \prime } = u , a ^ { \prime } = a , b ^ { \prime } = b$ . We have $( 1 ) \ u \cdot a \ = \ u ^ { \prime } \cdot a ^ { \prime } \sim _ { R } \ \sigma ( u ^ { \prime } \cdot a ^ { \prime } )$ ; and (2) either $u \cdot v \ = \ u ^ { \prime } \cdot v \in L$ or $u \cdot \sigma ( v ) = u ^ { \prime } \cdot \sigma ( v ) \in L$

Now, we consider the if direction. Suppose we have a word $u ^ { \prime }$ of the same ${ \sim } _ { R ^ { - } } \mathrm { t y p e }$ as u, symbols $a ^ { \prime } , b ^ { \prime } \in \Sigma$ , a word $w \in \Sigma ^ { * }$ and an R-preserving mapping $\sigma$ with $\sigma ( a ^ { \prime } ) = b ^ { \prime }$ and $\sigma ( d ) =$ d for all other $d \in \boldsymbol { u } ^ { \prime } \cdot \boldsymbol { w }$ such that the two conditions in Definition 4 hold. Since Σ is dense, we can use $\sigma _ { u \to u ^ { \prime } } .$ , the bijective R-preserving mapping for u and $u ^ { \prime } .$ . Let $b = \sigma _ { u  u ^ { \prime } } ^ { - 1 } ( b ^ { \prime } )$ and $v = \sigma _ { u  u ^ { \prime } } ^ { - 1 } ( w )$ . Let $\sigma ^ { \prime }$ be a mapping such that $\sigma ^ { \prime } ( a ) = b$ and $\sigma ^ { \prime } ( d ) = d$ for all other symbols d $\mathbf { \Psi } \cdot \in { \boldsymbol { u } } \cdot \mathbf { \boldsymbol { v } }$ We have

$$
u \cdot v \sim _ { R } u ^ { \prime } \cdot w \sim _ { R } \sigma ( u ^ { \prime } \cdot w ) \sim _ { R } \sigma _ { u  u ^ { \prime } } ^ { - 1 } \big ( \sigma ( u ^ { \prime } \cdot w ) \big ) = \sigma ^ { \prime } ( u \cdot w )\tag{4}
$$

$$
u ^ { \prime } \cdot w \sim _ { R } \sigma _ { u  u ^ { \prime } } ^ { - 1 } ( u ^ { \prime } \cdot w ) = u \cdot v\tag{5}
$$

$$
\sigma ( u ^ { \prime } ) \cdot w \sim _ { R } \sigma _ { u  u ^ { \prime } } ^ { - 1 } \big ( \sigma ( u ^ { \prime } ) \cdot w \big ) = \sigma ^ { \prime } ( u ) \cdot v\tag{6}
$$

The equality $\sigma _ { u \to u ^ { \prime } } ^ { - 1 } \bigl ( \sigma ( u ^ { \prime } \cdot w ) \bigr ) = \sigma ^ { \prime } ( u \cdot w )$ in 4 holds since σ only changes the symbol $a ^ { \prime }$ in $u ^ { \prime } \cdot w \ \mathrm { t o } \ b ^ { \prime }$ which is then mapped to b by $\sigma _ { u  u ^ { \prime } } ^ { - 1 }$ , while all other symbols will be mapped to the corresponding symbols in u v by $\sigma _ { u  u ^ { \prime } } ^ { - 1 }$ . It follows from $\sigma _ { u  u ^ { \prime } } ^ { - 1 } ( w )$ that the equality $\sigma _ { u \to u ^ { \prime } } ^ { - 1 } ( \sigma ( u ^ { \prime } ) \cdot w ) = \sigma ^ { \prime } ( u ) \cdot v$ in 6 holds.

$$
6 ,
$$

$$
u ^ { \prime } \cdot w \in L
$$

$$
\sigma ( u ^ { \prime } ) \cdot w \in L ,
$$

$$
u \cdot v \in L
$$

$$
\sigma ^ { \prime } ( u ) \cdot v \notin L
$$

We use the technical lemma below to establish the PSPACE upper bound. Informally, the lemma states that it sufices to replace the symbol a with an arbitrary symbol $b ,$ provided that the resulting word remains order-equivalent to $u ,$ and then compare the corresponding residual languages.

Lemma 29. Let be a $k { - } D R A$ over $( \Sigma , < )$ , and let $u \in \Sigma ^ { * }$ and $a \in u$ . Then $a \in m e m _ { L } ( u )$ if and only if, for any fresh symbol b and any simple mapping σ such that $\sigma ( a ) = b$ and $\sigma ( d ) = d$ for all other symbol $d \in \Sigma$ , we have

$$
u \cdot v \in L ( { \cal A } ) \iff \sigma ( u ) \cdot v \notin L ( { \cal A } ) \quad f o r \ s o m e \ v \in { \Sigma } ^ { * } .
$$

Proof. Given a word $u \in \Sigma ^ { * }$ and $a \in u$ , assume a mem $\mathbf { \Psi } _ { L } ( u )$

By Lemma 28, we know there is a fresh symbol $b \notin u .$ , a word $v \in \Sigma ^ { * }$ and an R-preserving mapping σ with $\sigma ( a ) = b$ and $\sigma ( d ) = d$ for all other symbols $d \in u \cdot v$ such that v is an L-distinguishing word for u and $\sigma ( u )$

Let $b ^ { \prime } \notin$ u and let $\sigma ^ { \prime }$ be a mapping with $\sigma ^ { \prime } ( a ) = b ^ { \prime }$ and $\sigma ^ { \prime } ( d ) = d$ for all other symbols $d \in \Sigma$ such that $u \sim _ { R } \sigma ^ { \prime } ( u )$ . We want to prove that there is an L-distinguishing word $v ^ { \prime }$ for u and $\sigma ^ { \prime } ( u )$ . If $b ^ { \prime } = b .$ , this is immediate by taking $v ^ { \prime } = v$ . For the case $b ^ { \prime } \neq b ,$ we show below that it is always possible to find such a $v ^ { \prime }$

Without loss of generality assume $b ^ { \prime } > a$ . We define a mapping $\sigma ^ { \prime \prime }$ that helps to construct $v ^ { \prime }$ from v. The key idea is to map all symbols in v that lie between a and $b ^ { \prime }$ to symbols greater than $b ^ { \prime } ,$ while preserving their relative order with respect to all symbols except $b ^ { \prime } .$ . Formally, let w be the increasing sequence of symbols in v that satisfy $a < w [ i ] \le b ^ { \prime }$ for all i, and $w [ i ] < w [ j ]$ whenever $i < j$

Let $c \in u \cdot v$ be the smallest symbol strictly larger than $b ^ { \prime }$ such that there is no symbol in $u \cdot v$ between $b ^ { \prime }$ and c. We now define $\sigma ^ { \prime \prime }$ as follows:

$$
\sigma ^ { \prime \prime } ( w [ i ] ) ~ = ~ b ^ { \prime } + { \frac { c - b ^ { \prime } } { | w | + 1 } } \cdot ( i + 1 ) \quad { \mathrm { f o r ~ a l l ~ } } w [ i ] ,
$$

and $\sigma ^ { \prime \prime } ( d ) = d$ for all other symbols $d \in v$

We then let $v ^ { \prime } = \sigma ^ { \prime \prime } ( v )$ . It is not hard to verify that

$$
u \cdot v \sim _ { R } u \cdot v ^ { \prime } \qquad \mathrm { a n d } \qquad \sigma ( u \cdot v ) \sim _ { R } \sigma ^ { \prime } ( u \cdot v ^ { \prime } ) .
$$

This completes the proof.

Theorem 19. The memorability problem is in PSPACE for DRAs over dense domain $( \Sigma , < )$

Proof. Given a DRA $\mathcal { A } = ( Q , q _ { 0 } , F , \varDelta )$ , a word $u ,$ and a symbol $a \in u .$ we give a PSPACE algorithm to decide whether $a \in m e m _ { L } ( u )$ . Let $b \notin$ u be a symbol such that $u \sim _ { R } u ^ { \prime }$ , where $u ^ { \prime }$ is obtained from u by replacing the occurrence of a with b. By Lemma 29, it sufices to check whether there exists an L-distinguishing word v for u and $u ^ { \prime }$

This reduces to checking whether two configurations $( q , r )$ and $( \boldsymbol { q } ^ { \prime } , \boldsymbol { r } ^ { \prime } )$ are equivalent, where

$$
( q _ { 0 } , \varepsilon ) \stackrel { u } {  } _ { \cal A } ( q , r ) \mathrm { ~ a n d ~ } ( q _ { 0 } , \varepsilon ) \stackrel { u ^ { \prime } } { \longrightarrow } _ { \cal A } ( q ^ { \prime } , r ^ { \prime } ) .
$$

By Theorem 18, configuration equivalence for DRAs can be decided in PSPACE. Thus, whether a is $L .$ -memorable in u can also be checked in PSPACE. □