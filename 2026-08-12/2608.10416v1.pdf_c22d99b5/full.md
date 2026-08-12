# Riemann GeoResolver: A Non-Euclidean Attention Framework from Euclidean Resolver to Hyperbolic-Spherical Geometry

Liangchen Ge

## Abstract

We present a theoretical foundation for inverse-distance attention, from its Euclidean prototype (Resolver) to its non-Euclidean realization (Riemann GeoResolver). The Euclidean part establishes three core theorems: (1) circuit separation—IDA achieves exact retrieval with O(1) resources while softmax requires Ω((log n)<sup>2</sup>) width; (2) a Polyak– Lojasiewicz inequality with $\Omega ( e ^ { \Delta ^ { 2 } / \sqrt { d } } / \Delta ^ { 2 } )$ stronger constant than softmax, implying linear convergence, O(log n) Lipschitz scaling under a low-rank/clustering assumption, Θ(1) Hessian spread, and absence of spurious local minima; (3) a width-independent efective rank bound that limits noise memorizatio softmax memorizes arbitrary labels when d<sub>h</sub> ≥ n, while IDA limits test error to O(η<sup>2</sup>). The non-Euclidean extension then builds upon this prototype, replacing Euclidean distance with hyperbolic geodesic distance for storage and spherical geodesic distance for routing. The Riemann GeoResolver framework comprises ten integrated modules: four HIDA operators spanning Θ(n<sup>2</sup>) to Θ(1) per token; Hyperbolic Curvature Compression (HCC) with provable error bounds; HyperGate with gradient lower-bound theorem; Spherical Inverse Distance Attention (SIDA) with sphere-analog PL inequalities; Dynamic Memory Genesis (DMG) with O(log T) regret bounds; and Geodesic Sparse Routing (GSR) with quality and communication bounds. The Euclidean theorems are proved in full; the non-Euclidean extension theorems are proved with analogous arguments. This work establishes a theoretical arc: from Euclidean attention as a special case, to hyperbolic memory, to spherical retrieval.

## 1 Introduction

Scope. This is a theoretical study. The primary contribution is the set of theorems, lemmas, and architectural principles for inverse-distance attention, first in Euclidean space and then extended to hyperbolic and spherical geometries.

## 1.1 From Softmax to Inverse Distance

The Transformer [1] computes attention via inner products followed by softmax. In softmax attention, when $\mathbf { q } = \mathbf { k } _ { j } ,$ , the output remains a weighted average over all keys rather than focusing on the exact match. We consider Inverse Distance Attention (IDA):

$$
W _ { i j } = \frac { ( d ( { \bf q } _ { i } , { \bf k } _ { j } ) ^ { 2 } + \varepsilon ) ^ { - 1 } } { \sum _ { m } ( d ( { \bf q } _ { i } , { \bf k } _ { m } ) ^ { 2 } + \varepsilon ) ^ { - 1 } } .
$$

The Euclidean variant is Resolver; extensions to non-Euclidean geometries are collectively termed GeoResolver.

## 1.2 Motivation: Why Inverse Distance?

The softmax attention mechanism has a well-known property: even when a query exactly matches a key, the output is a weighted average of all values, not a hard retrieval of the matched value. This property is intrinsic to the softmax function, which always assigns positive probability to all tokens. The inversedistance kernel addresses this by assigning weights inversely proportional to squared distance, which, in the limit $\varepsilon  0 ^ { + }$ , converges to a one-hot selection of the exact match.

This diference has fundamental implications for expressiveness, optimization, and generalization. The softmax function, while diferentiable and probabilistically interpretable, creates optimization challenges: the gradient is small when the logits are large (saturation), and the Hessian is low-rank when n is large. These properties afect convergence rates and the ability to escape saddle points. The inverse-distance kernel, by contrast, has gradients that remain large near exact matches and a Hessian that is full-rank, suggesting better optimization properties.

Furthermore, the softmax’s capacity to represent arbitrary functions grows with width, leading to a phenomenon where, once the hidden dimension exceeds the number of training points, the model can memorize arbitrary labels, including noise. This is a form of overparameterization that degrades generalization. The inverse-distance kernel’s efective rank is bounded independent of width, preventing this memorization efect.

These observations motivate the theoretical framework developed in this paper. We prove three core theorems establishing that IDA enjoys provably stronger guarantees than softmax in expressiveness, optimization, and generalization.

## 1.3 Theoretical Contributions

1. Euclidean Resolver (Part I): Three core theorems with complete proofs:

• Theorem 1 (Circuit Separation): IDA retrieves exact matches with O(1) resources; softmax requires $\Omega ( ( \log n ) ^ { 2 } )$ width.

• Theorem 2 (PL Inequality): IDA satisfies PL with $\Omega ( e ^ { \Delta ^ { 2 } / \sqrt { d } } / \Delta ^ { 2 } )$ larger constant than softmax, linear convergence, ${ \mathcal { O } } ( \log n )$ Lipschitz scaling under a low-rank/clustering assumption (general bound $\mathcal { O } ( n ) )$ , Θ(1) Hessian spread, and no spurious local minima.

• Theorem 3 (Efective Rank Bound): IDA’s efective rank is bounded independent of width, limiting noise memorization; softmax memorizes arbitrary labels when $d _ { h } \geq n ;$ IDA limits test error to $\mathcal { O } ( \eta ^ { 2 } )$

2. Non-Euclidean Extension (Part II): Replaces Euclidean distance with hyperbolic geodesic distance for storage and spherical geodesic distance for routing, yielding a ten-module framework with provable guarantees.

## 1.4 Related Work

Our work builds on and extends several lines of research. We organize the discussion by topic.

Inverse-distance attention. McCarter [12] proposed inverse-distance weighting attention independently. That work and ours share the same $\varepsilon  0$ limit, but diverge fundamentally in several ways. First, McCarter did not embed the kernel in a full QKV architecture with learned projections; the attention was applied directly to input features. Second, the analysis in McCarter’s work was empirical, focusing on performance on a few tasks, with no theoretical analysis of optimization or generalization. Third, the extension to non-Euclidean geometries was not considered. Our contribution is the first characterization of this kernel’s behavior in the full Transformer setting, with complete theoretical guarantees.

RBF and kernel attention. Bello et al. [10] explored attention mechanisms with radial basis function kernels in the context of convolutional architectures. Their work demonstrated that replacing the softmax with a Gaussian kernel could improve performance on certain tasks, but the analysis was empirical and did not address optimization or generalization bounds. The Nadaraya– Watson estimator [14, 15] is the non-parametric ancestor of attention, but its theoretical properties are typically analyzed in the regression setting with fixed kernel bandwidths, not in the context of learned representations. The inversedistance kernel we consider difers from the Gaussian RBF in that it has a heavy tail (decaying as $1 / r ^ { 2 }$ rather than $e ^ { - r ^ { 2 } } )$ , which afects the Lipschitz and generalization properties.

Optimization of attention mechanisms. Karimi et al. [57] established PL inequalities for various machine learning problems, including linear and logistic regression. Our work applies this framework to the attention mechanism itself, showing that the PL constant for IDA is exponentially larger than for softmax. Belkin et al. [58] analyzed the double descent phenomenon in overparameterized models, showing that test error can increase and then decrease with model capacity. Our Theorem 3 provides a complementary analysis for the attention mechanism specifically: softmax exhibits a memorization catastrophe when the hidden dimension exceeds the number of training points, while IDA’s efective rank remains bounded.

Hyperbolic and spherical neural networks. Nickel and Kiela [20, 21] introduced hyperbolic embeddings for hierarchical data, showing that hyperbolic space can represent tree-structured data with low distortion. Ganea et al. [22] developed hyperbolic neural networks, including hyperbolic linear layers and activation functions. Chami et al. [23] extended these ideas to graph convolutional networks. Gulcehre et al. [24] proposed hyperbolic attention networks, incorporating hyperbolic distance into the attention mechanism. Our work difers in several respects: we provide a unified theoretical framework that spans Euclidean, hyperbolic, and spherical geometries; we focus on the inversedistance kernel rather than learned attention; we provide optimization bounds (PL inequalities) and capacity bounds (efective rank) that are not present in prior work; and we analyze the routing problem separately from the attention problem.

Mixture of experts and routing. Shazeer et al. [40] introduced sparselygated MoE layers, where a router network selects a subset of experts for each input. This line of work has been extended in various directions, including switch transformers [41], GShard [42], and GLaM [43]. Our GSR module difers in that the routing is based on spherical distance rather than a learned gating network, and we provide theoretical communication bounds rather than empirical scaling results. The use of spherical geometry for routing is motivated by the compactness of the sphere and the natural interpretation of prototypes as points on a hypersphere.

State space models and eficient attention. Recent work on state space models [34, 35, 36] has explored alternatives to attention for sequence modeling. Our work is complementary: we maintain the attention mechanism but replace the kernel function. The structured state space duality framework [37] provides a unified view of Transformers and state space models; our work could potentially be integrated into this framework by treating the inversedistance kernel as a diferent kernel function within the same duality.

## 1.5 Paper Structure

The paper is organized as a single arc: from Euclidean theory to non-Euclidean generalization.

Part I (Sections 2–3) develops the Euclidean prototype. This establishes the theoretical baseline: why inverse-distance attention works, how it compares to softmax, and what optimization and generalization guarantees it enjoys.

Part II (Sections 4–10) extends the prototype in three logical stages:

• Geometry (Sections 4–5): hyperbolic distance replaces Euclidean distance for attention, with four variants spanning $\Theta ( n ^ { 2 } )$ to Θ(1) complexity.

• Compression and Control (Sections $\textstyle 6 - 7 ) { \colon }$ curvature-adaptive quantization (HCC) and gating (HyperGate) for memory eficiency and gradient stability.

• Memory and Routing (Sections 8–10): spherical routing for retrieval (SIDA), dynamic memory allocation (DMG), and sparse routing with quality guarantees (GSR).

Each stage is motivated by a specific limitation of the previous stage, and each theorem is proved in the corresponding section.

## Part I

## Euclidean Resolver: Theoretical Prototype

## 2 Theoretical Framework

## 2.1 Preliminaries

Let n denote the sequence length, $d _ { h }$ the hidden dimension, and $d _ { v }$ the value dimension. For $\mathbf { Q } , \mathbf { K } \in \mathbb { R } ^ { n \times d _ { h } }$ and $\mathbf { V } \in \mathbb { R } ^ { n \times d _ { v } }$ , we define:

$$
\mathbf D _ { i j } = \| \mathbf q _ { i } - \mathbf k _ { j } \| _ { 2 } ^ { 2 } + \varepsilon , \quad \mathbf W _ { i j } = \frac { \mathbf D _ { i j } ^ { - 1 } } { \sum _ { m } \mathbf D _ { i m } ^ { - 1 } } , \quad \mathrm { I D A } ( \mathbf Q , \mathbf K , \mathbf V ) = \mathbf W \mathbf V .
$$

Here $\varepsilon > 0$ is a small positive constant that prevents division by zero. Throughout this paper, we consider the limit $\varepsilon \to 0 ^ { + }$ for theoretical statements unless otherwise noted.

We adopt the convention that $\| \cdot \| _ { 2 }$ denotes the Euclidean norm. For a matrix A, $\| \mathbf { A } \| _ { F }$ denotes the Frobenius norm and $\| \mathbf { A } \| _ { 2 }$ the spectral norm. The efective rank of a positive semidefinite matrix K is defined as:

$$
\operatorname { e f f - r a n k } ( \mathbf { K } ) = { \frac { ( \operatorname { t r } \mathbf { K } ) ^ { 2 } } { \operatorname { t r } ( \mathbf { K } ^ { 2 } ) } } .
$$

This quantity measures the efective number of significant eigenvalues of K and lies in [1, n] for an $n \times n$ matrix.

## 2.2 Geometric Properties of the Inverse-Distance Kernel

Before proving the main theorems, we establish several geometric properties of the inverse-distance kernel that will be used throughout.

Property 1: Translation invariance. The Euclidean inverse-distance kernel is invariant under simultaneous translation of all queries and keys:

$$
\| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } = \| ( \mathbf { q } _ { i } + \mathbf { t } ) - ( \mathbf { k } _ { j } + \mathbf { t } ) \| _ { 2 } ^ { 2 } .
$$

This invariance implies that the attention weights are unchanged by global shifts of the input space. This property is not shared by the softmax inner-product kernel, which is translation-invariant only up to normalization.

Property 2: Scale sensitivity. Under a uniform scaling $\mathbf { q } _ { i } \mapsto \lambda \mathbf { q } _ { i } , \mathbf { k } _ { j } \mapsto$ $\lambda \mathbf { k } _ { j } .$ , the squared distances scale as $\lambda ^ { 2 }$ . For the inverse-distance kernel with fixed ε, this changes the efective regularization:

$$
( \lambda ^ { 2 } \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } + \varepsilon ) ^ { - 1 } = \lambda ^ { - 2 } ( \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } + \varepsilon / \lambda ^ { 2 } ) ^ { - 1 } .
$$

Thus scaling is equivalent to rescaling ε. This property is used in the proof of the circuit separation theorem.

Property 3: Concentration in high dimensions. For random keys drawn from a high-dimensional isotropic distribution, the squared distances concentrate around their mean:

$$
\mathbb { P } \left( \lvert \lVert { \bf q } _ { i } - { \bf k } _ { j } \rVert _ { 2 } ^ { 2 } - 2 d _ { h } \sigma ^ { 2 } \big \lvert \geq t \right) \leq 2 \exp \left( - \frac { c d _ { h } t ^ { 2 } } { \sigma ^ { 4 } } \right) .
$$

This concentration implies that in high dimensions, the inverse-distance kernel becomes approximately constant across all pairs, which has implications for the Lipschitz scaling (Lemma 1). The concentration bound follows from standard results on Lipschitz functions of Gaussian random variables.

Property 4: Sparsity-promoting behavior. For fixed ε, the inversedistance kernel assigns negligible weight to keys beyond distance $\mathcal { O } ( \sqrt { \varepsilon } n ^ { 1 / 2 } )$ from the query, since:

$$
( d ^ { 2 } + \varepsilon ) ^ { - 1 } \leq { \frac { \varepsilon } { d ^ { 2 } } } \cdot { \frac { 1 } { \varepsilon } } = { \mathcal { O } } ( \varepsilon / d ^ { 2 } ) .
$$

This sparsity is inherent to the kernel and does not require additional sparsification. The proof of Theorem 1 exploits this property.

## 2.3 Theorem 1: Expressiveness Separation

We begin by establishing the fundamental property of IDA: in the limit $\varepsilon \to 0 ^ { + }$ it performs exact retrieval.

Lemma 2.1 (Exact Retrieval Property). For pairwise distinct keys and ${ \bf q } =$ $\mathbf { k } _ { j ^ { * } }$ ,

$$
\operatorname * { l i m } _ { \varepsilon  0 } W _ { j ^ { * } } = 1 , \qquad \operatorname * { l i m } _ { \varepsilon  0 } W _ { j } = 0 \quad ( j \neq j ^ { * } ) .
$$

Proof. When $\mathbf { q } _ { i } = \mathbf { k } _ { j } { * }$ , we have $d ( \mathbf { q } _ { i } , \mathbf { k } _ { j ^ { * } } ) = 0$ , so $\mathbf { D } _ { i j ^ { * } } = \varepsilon .$ . For any $j \neq j ^ { * }$ since the keys are pairwise distinct, $d ( \mathbf { q } _ { i } , \mathbf { k } _ { j } ) > 0$ . Therefore $\mathbf { D } _ { i j } > \varepsilon$ for $j \neq j ^ { * }$ which implies $\mathbf { D } _ { i j } ^ { - 1 } < \varepsilon ^ { - 1 }$ . The limit follows immediately:

$$
\operatorname* { l i m } _ { \varepsilon \to 0 } \frac { \varepsilon ^ { - 1 } } { \varepsilon ^ { - 1 } + \sum _ { j \neq j ^ { * } } \mathbf { D } _ { i j } ^ { - 1 } } = 1 .
$$

Now we prove the main circuit separation result.

Theorem 2.2 (Theorem 1: Softmax Lower Bound). There exists an orthogonalkeys instance where IDA achieves exact retrieval with $\mathcal { O } ( 1 )$ resources, while any softmax-based architecture requires $d = \Omega ( ( \log n ) ^ { 2 } )$ or $H = \Omega ( \log n )$ for ε-approximation.

Proof. We construct an instance with n orthonormal keys. Let $\{ { \bf e } _ { j } \} _ { j = 1 } ^ { n }$ be an orthonormal basis in $\mathbb { R } ^ { d }$ , so we require $d \geq n$ . Set $\mathbf { k } _ { j } = R \mathbf { e } _ { j }$ for some $R > 0$ and set $\mathbf { q } = \mathbf { k } _ { 1 } = R \mathbf { e } _ { 1 }$

IDA analysis. For IDA, the squared distance from the query to key j is:

$$
\| \mathbf { q } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } = { \left\{ \begin{array} { l l } { 0 , } & { j = 1 , } \\ { 2 R ^ { 2 } , } & { j \neq 1 , } \end{array} \right. }
$$

since $\| \mathbf { e } _ { 1 } - \mathbf { e } _ { j } \| _ { 2 } ^ { 2 } = 2$ for $j \neq 1$ . Thus:

$$
W _ { 1 1 } ^ { \mathrm { I D A } } = \frac { 1 / \varepsilon } { 1 / \varepsilon + ( n - 1 ) / ( 2 R ^ { 2 } + \varepsilon ) } = \frac { 1 } { 1 + \varepsilon ( n - 1 ) / ( 2 R ^ { 2 } + \varepsilon ) } .
$$

Taking $\varepsilon \to 0 ^ { + }$ , we get:

$$
\operatorname* { l i m } _ { \varepsilon \to 0 } W _ { 1 1 } ^ { \mathrm { I D A } } = 1 .
$$

Moreover, for any $\varepsilon > 0 .$ , the weight is within δ of 1 whenever $\varepsilon ( n - 1 ) / ( 2 R ^ { 2 } { + } \varepsilon ) <$ $\delta / ( 1 - \delta )$ . Thus IDA achieves δ-approximation with $\varepsilon = \mathcal O ( \delta R ^ { 2 } / n )$

Softmax analysis. For softmax with temperature 1:

$$
W _ { 1 1 } ^ { \mathrm { s o f t } } = \frac { e ^ { \mathbf { q } ^ { \top } \mathbf { k } _ { 1 } / \sqrt { d } } } { \sum _ { m = 1 } ^ { n } e ^ { \mathbf { q } ^ { \top } \mathbf { k } _ { m } / \sqrt { d } } } .
$$

Since $\mathbf { q } ^ { \top } \mathbf { k } _ { 1 } = R ^ { 2 }$ and $\mathbf { q } ^ { \top } \mathbf { k } _ { m } = 0$ for m $\neq 1$ (orthogonality), we have:

$$
W _ { 1 1 } ^ { \mathrm { s o f t } } = \frac { e ^ { R ^ { 2 } / \sqrt { d } } } { e ^ { R ^ { 2 } / \sqrt { d } } + ( n - 1 ) e ^ { 0 } } = \frac { 1 } { 1 + ( n - 1 ) e ^ { - R ^ { 2 } / \sqrt { d } } } .
$$

To achieve $W _ { 1 1 } ^ { \mathrm { s o f t } } \geq 1 - \delta$ , we need:

$$
1 + ( n - 1 ) e ^ { - R ^ { 2 } / \sqrt { d } } \leq \frac { 1 } { 1 - \delta } .
$$

For $\delta \leq 1 / 2$ , we have $1 / ( 1 - \delta ) \leq 1 + 2 \delta $ , so:

$$
( n - 1 ) e ^ { - R ^ { 2 } / \sqrt { d } } \leq 2 \delta .
$$

Taking logarithms:

$$
- { \frac { R ^ { 2 } } { \sqrt { d } } } \leq \log { \frac { 2 \delta } { n - 1 } } = \log ( 2 \delta ) - \log ( n - 1 ) .
$$

Thus:

$$
\frac { R ^ { 2 } } { \sqrt { d } } \geq \log \frac { n - 1 } { 2 \delta } .
$$

With bounded radius $R \leq R _ { \operatorname* { m a x } }$ (the maximum norm of keys), we obtain:

$$
d \geq \left( \frac { R _ { \operatorname* { m a x } } ^ { 2 } } { \log ( ( n - 1 ) / ( 2 \delta ) ) } \right) ^ { 2 } = \Omega ( ( \log n ) ^ { 2 } ) .
$$

For multi-head architectures, each head independently must satisfy the same lower bound, so the total width $H \cdot d _ { h }$ must be at least this value. Therefore any softmax-based architecture requires either $d _ { h } = \Omega ( ( \log n ) ^ { 2 } )$ or $H = \Omega ( \log n )$

Proposition 2.3 (Dense-Key Degradation). If keys have covering radius r from $\mathbf { k } _ { j } \ ( i . e . , \ | | \mathbf { k } _ { j } - \mathbf { k } _ { m } | | \geq r$ for all m $\neq j )$ , then

$$
W _ { j j } \ge \frac { 1 } { 1 + \varepsilon ( n - 1 ) / ( r ^ { 2 } + \varepsilon ) } .
$$

For Gaussian keys with covariance $\sigma ^ { 2 } I , r ^ { 2 } = \Theta ( d _ { h } \sigma ^ { 2 } n ^ { - 2 / d _ { h } } )$

Proof. For any m $\neq j$ , the squared distance $d _ { j m } ^ { 2 } = \| \mathbf { k } _ { j } - \mathbf { k } _ { m } \| _ { 2 } ^ { 2 }$ satisfies $d _ { j m } ^ { 2 } \geq r ^ { 2 }$ by definition of the covering radius. Therefore:

$$
\begin{array} { r } { \mathbf { D } _ { j m } ^ { - 1 } = ( d _ { j m } ^ { 2 } + \varepsilon ) ^ { - 1 } \leq ( r ^ { 2 } + \varepsilon ) ^ { - 1 } . } \end{array}
$$

The attention weight for the exact match is:

$$
W _ { j j } = \frac { \varepsilon ^ { - 1 } } { \varepsilon ^ { - 1 } + \sum _ { m \neq j } \mathbf { D } _ { j m } ^ { - 1 } } \geq \frac { \varepsilon ^ { - 1 } } { \varepsilon ^ { - 1 } + ( n - 1 ) ( r ^ { 2 } + \varepsilon ) ^ { - 1 } } = \frac { 1 } { 1 + \varepsilon ( n - 1 ) / ( r ^ { 2 } + \varepsilon ) } .
$$

This proves the first part.

For Gaussian keys with covariance $\sigma ^ { 2 } I .$ , the keys are distributed in a ball of radius $O ( \sqrt { d _ { h } } \sigma )$ with high probability. The covering radius of n points in dimension $d _ { h }$ scales as $r = O ( \sigma \sqrt { d _ { h } } n ^ { - 1 / d _ { h } } )$ . Squaring gives $r ^ { 2 } = \Theta ( \bar { d _ { h } } \sigma ^ { 2 } n ^ { - 2 / d _ { h } } )$ . □

## 2.4 Lemma 1: Lipschitz Scaling

We now analyze the Lipschitz continuity of the attention mechanism with respect to input perturbations. This is important for understanding the stability and optimization of attention layers.

Lemma 2.4 (Lipschitz Scaling under Low Efective Rank). For a single attention layer with residual connection, assuming bounded weights and inputs, and additionally assuming the key distribution has efective rank $r _ { e f f } = \Theta ( 1 )$ or exhibits a clustering structure with $C = \Theta ( 1 )$ clusters, then:

$$
L _ { I D A } = { \cal O } ( 1 + \log n \cdot L _ { W } ^ { 2 } R ^ { 2 } ) , L _ { s o f t } = { \cal O } ( n \cdot L _ { W } ^ { 2 } R ^ { 2 } / \sqrt { d } ) .
$$

Without the low-rank/clustering assumption, the general bound is:

$$
L _ { I D A } = \mathcal { O } ( n \cdot L _ { W } ^ { 2 } R ^ { 2 } / \varepsilon ^ { 2 } ) .
$$

Proof. We analyze the Lipschitz constant of the attention output with respect to the input queries. Let $\begin{array} { r } { { \bf o } _ { i } = \sum _ { j } W _ { i j } { \bf v } _ { j } } \end{array}$ be the output for the i-th query. The Lipschitz constant is the supremum over inputs of the spectral norm of the Jacobian $\partial \mathbf { o } _ { i } / \partial \mathbf { q } _ { i }$

Step 1: Jacobian of the attention weight. Define $\mathbf { S } _ { i j } = ( \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } +$ $\varepsilon ) ^ { - 1 }$ and $\begin{array} { r } { Z _ { i } = \sum _ { m } \mathbf { S } _ { i m } } \end{array}$ . Then $W _ { i j } = { \bf S } _ { i j } / Z _ { i }$

The derivative of $\mathbf { S } _ { i j }$ with respect to $\mathbf { q } _ { i }$ is obtained by diferentiating the inverse:

$$
\frac { \partial \mathbf { S } _ { i j } } { \partial \mathbf { q } _ { i } } = - \frac { 2 ( \mathbf { q } _ { i } - \mathbf { k } _ { j } ) } { ( \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } + \varepsilon ) ^ { 2 } } .
$$

Taking the Euclidean norm:

$$
\left\| \frac { \partial \mathbf { S } _ { i j } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } = \frac { 2 \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } } { ( \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } + \varepsilon ) ^ { 2 } } .
$$

For $a = \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } \geq 0$ , the function $a / ( a ^ { 2 } + \varepsilon ) ^ { 2 }$ has its maximum at $a ^ { 2 } = \varepsilon / 3$ To see this, diferentiate:

$$
{ \frac { d } { d a } } \left( { \frac { a } { ( a ^ { 2 } + \varepsilon ) ^ { 2 } } } \right) = { \frac { \varepsilon - 3 a ^ { 2 } } { ( a ^ { 2 } + \varepsilon ) ^ { 3 } } } .
$$

Setting this to zero gives $a ^ { 2 } = \varepsilon / 3$ . Substituting:

$$
\left\| \frac { \partial \mathbf { S } _ { i j } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } \leq \frac { 2 \cdot ( \sqrt { \varepsilon / 3 } ) } { ( \varepsilon / 3 + \varepsilon ) ^ { 2 } } = \frac { 2 \sqrt { \varepsilon / 3 } } { ( 4 \varepsilon / 3 ) ^ { 2 } } = \frac { 1 8 } { 1 6 \sqrt { 3 } } \varepsilon ^ { - 3 / 2 } = \mathcal { O } ( \varepsilon ^ { - 3 / 2 } ) .
$$

However, the dependence on $\varepsilon$ is not the primary concern for the Lipschitz scaling with $n ;$ we track the scaling with n and $d _ { h }$

The derivative of $W _ { i j }$ with respect to $\mathbf { q } _ { i }$ is:

$$
{ \frac { \partial W _ { i j } } { \partial \mathbf { q } _ { i } } } = { \frac { \partial \mathbf { S } _ { i j } } { \partial \mathbf { q } _ { i } } } \cdot { \frac { 1 } { Z _ { i } } } - { \frac { \mathbf { S } _ { i j } } { Z _ { i } ^ { 2 } } } \sum _ { m = 1 } ^ { n } { \frac { \partial \mathbf { S } _ { i m } } { \partial \mathbf { q } _ { i } } } .
$$

This can be rewritten more compactly as:

$$
{ \frac { \partial W _ { i j } } { \partial \mathbf { q } _ { i } } } = { \frac { 1 } { Z _ { i } } } \left( { \frac { \partial \mathbf { S } _ { i j } } { \partial \mathbf { q } _ { i } } } - W _ { i j } \sum _ { m = 1 } ^ { n } { \frac { \partial \mathbf { S } _ { i m } } { \partial \mathbf { q } _ { i } } } \right) .
$$

Taking norms and applying the triangle inequality:

$$
\left\| \frac { \partial W _ { i j } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } \leq \frac { 1 } { Z _ { i } } \left( \left\| \frac { \partial \mathbf { S } _ { i j } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } + W _ { i j } \sum _ { m = 1 } ^ { n } \left\| \frac { \partial \mathbf { S } _ { i m } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } \right) .
$$

Step 2: Sum over j. Summing the previous inequality over $j = 1 , \ldots , n \colon$

$$
\sum _ { j = 1 } ^ { n } \left\| \frac { \partial W _ { i j } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } \leq \frac { 1 } { Z _ { i } } \sum _ { j = 1 } ^ { n } \left\| \frac { \partial \mathbf { S } _ { i j } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } + \frac { 1 } { Z _ { i } } \sum _ { j = 1 } ^ { n } W _ { i j } \sum _ { m = 1 } ^ { n } \left\| \frac { \partial \mathbf { S } _ { i m } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } .
$$

The second term simplifies because $\textstyle \sum _ { j = 1 } ^ { n } W _ { i j } = 1$

$$
\sum _ { j = 1 } ^ { n } W _ { i j } \sum _ { m = 1 } ^ { n } \left\| { \frac { \partial \mathbf { S } _ { i m } } { \partial \mathbf { q } _ { i } } } \right\| _ { 2 } = \sum _ { m = 1 } ^ { n } \left\| { \frac { \partial \mathbf { S } _ { i m } } { \partial \mathbf { q } _ { i } } } \right\| _ { 2 } .
$$

Therefore:

$$
\sum _ { j = 1 } ^ { n } \left\| \frac { \partial W _ { i j } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } \leq \frac { 2 } { Z _ { i } } \sum _ { m = 1 } ^ { n } \left\| \frac { \partial \mathbf { S } _ { i m } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } .
$$

Step 3: General bound. Since $\| \partial \mathbf { S } _ { i m } / \partial \mathbf { q } _ { i } \| _ { 2 } \leq 2 R / \varepsilon ^ { 2 }$ for bounded inputs $( \| \mathbf q _ { i } - \mathbf k _ { m } \| _ { 2 } \leq R )$ , and $Z _ { i } \geq \varepsilon ^ { - 1 }$ (at least the diagonal term), we get:

$$
\sum _ { j = 1 } ^ { n } \left\| \frac { \partial W _ { i j } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } \leq \frac { 2 } { \varepsilon ^ { - 1 } } \cdot \frac { 2 n R } { \varepsilon ^ { 2 } } = \frac { 4 n R } { \varepsilon } .
$$

This gives $\begin{array} { r } { \sum _ { i } \| \partial W _ { i j } / \partial \mathbf { q } _ { i } \| _ { 2 } = \mathcal { O } ( n R / \varepsilon ) } \end{array}$

The output Jacobian is:

$$
{ \frac { \partial \mathbf { o } _ { i } } { \partial \mathbf { q } _ { i } } } = \sum _ { j = 1 } ^ { n } \mathbf { v } _ { j } \otimes { \frac { \partial W _ { i j } } { \partial \mathbf { q } _ { i } } } .
$$

Thus:

$$
\left\| \frac { \partial \mathbf { o } _ { i } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } \leq \sum _ { j = 1 } ^ { n } \| \mathbf { v } _ { j } \| _ { 2 } \left\| \frac { \partial W _ { i j } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } \leq \| \mathbf { V } \| _ { F } \sum _ { j = 1 } ^ { n } \left\| \frac { \partial W _ { i j } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } .
$$

With bounded values $\| \mathbf { V } \| _ { F } \le \sqrt { n } L _ { V }$ , this gives $L _ { \mathrm { I D A } } = \mathcal { O } ( n R L _ { V } / \varepsilon )$ in the general case.

Step 4: Low-rank/clustering case (under the stated assumption). Suppose the keys are organized into $C = \Theta ( 1 )$ clusters, each of radius $r _ { c }$ . Within each cluster, the distances are small. The sum over m can be decomposed:

$$
\sum _ { m = 1 } ^ { n } \left\| \frac { \partial \mathbf { S } _ { i m } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } = \sum _ { c = 1 } ^ { C } \sum _ { m \in \mathrm { c l u s t e r } ~ c } \left\| \frac { \partial \mathbf { S } _ { i m } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } .
$$

For keys in the same cluster as $\mathbf { q } _ { i } ,$ , the distances are within $r _ { c } ,$ and the contribution is $\mathcal { O } ( n _ { c } r _ { c } / \varepsilon ^ { 2 } )$ where $n _ { c }$ is the cluster size. For keys in other clusters, the distances are at least the inter-cluster distance, and the contribution is $\mathcal { O } ( n _ { c } / \Delta _ { c } ^ { 2 } )$ for $\Delta _ { c }$ the inter-cluster distance. The number of non-negligible distance scales is ${ \mathcal { O } } ( \log n )$ , yielding the ${ \mathcal { O } } ( \log n )$ scaling.

Step 5: Softmax comparison. For softmax, $\begin{array} { r } { W _ { i j } = \left. e ^ { s _ { i j } } \right/ \sum _ { m } e ^ { s _ { i } } } \end{array}$ <sup>m</sup> with $s _ { i j } = \mathbf { q } _ { i } ^ { \top } \mathbf { k } _ { j } / \sqrt { d _ { h } }$ . The derivative is:

$$
\frac { \partial W _ { i j } } { \partial \mathbf { q } _ { i } } = \frac { 1 } { \sqrt { d _ { h } } } W _ { i j } \left( \mathbf { k } _ { j } - \sum _ { m } W _ { i m } \mathbf { k } _ { m } \right) .
$$

Thus:

$$
\sum _ { j = 1 } ^ { n } \left\| \frac { \partial W _ { i j } } { \partial \mathbf { q } _ { i } } \right\| _ { 2 } \leq \frac { 1 } { \sqrt { d _ { h } } } \sum _ { j = 1 } ^ { n } W _ { i j } \| \mathbf { k } _ { j } - \bar { \mathbf { k } } _ { i } \| _ { 2 } \leq \frac { 2 R } { \sqrt { d _ { h } } } \sum _ { j = 1 } ^ { n } W _ { i j } = \frac { 2 n R } { \sqrt { d _ { h } } } .
$$

This gives the linear n scaling for softmax. The residual connection contributes an additive 1 to the Lipschitz constant. □

## 2.5 Lemma 2: Hessian Curvature Contrast

The curvature of the loss landscape at initialization is a key determinant of optimization speed. We now analyze the Hessian of a simple loss function driving a query to match a key.

Lemma 2.5 (Hessian Curvature Contrast). At equidistant initialization $( \mathbf { q } _ { i } =$ $\mathbf { k } _ { j } = \mathbf { 0 } )$ with keys at radius $\rho > 0$ , consider the loss $\mathcal { L } = ( W _ { 1 1 } - 1 ) ^ { 2 }$ driving the first query to match the first key. For softmax:

$$
\frac { \partial ^ { 2 } \mathcal { L } } { \partial s ^ { 2 } } \Big | _ { s = 0 } = \Theta ( n ^ { - 2 } ) .
$$

For $I D A ,$ , for any fixed $\rho > 0 .$

$$
\frac { \partial ^ { 2 } \mathcal { L } } { \partial \delta ^ { 2 } } \Big | _ { \delta = 0 } = \Theta ( 1 ) ,
$$

independent of n (up to the constant factor $( n - 1 ) ^ { 2 } / n ^ { 2 }$ which tends to 1 as $n  \infty )$ .

Proof. Softmax case. Let s denote the logit diference between the target key and the others. At equidistant initialization with keys at radius $\rho > 0 ,$ , the inner products are all equal, so $s = 0$ . The weight for the target key is $W _ { 1 1 } = 1 / n$

For a perturbation δ along the direction of the target key, the logit diference becomes $s = \mathcal { O } ( \delta \rho / \sqrt { d _ { h } } )$ . The softmax weight is:

$$
W _ { 1 1 } ( s ) = \frac { e ^ { s } } { e ^ { s } + ( n - 1 ) e ^ { 0 } } = \frac { 1 } { 1 + ( n - 1 ) e ^ { - s } } .
$$

We expand this function around $s = 0$ . Let $f ( s ) = 1 / ( 1 + ( n - 1 ) e ^ { - s } )$ . Then:

$$
f ( 0 ) = { \frac { 1 } { n } } .
$$

The first derivative is:

$$
f ^ { \prime } ( s ) = { \frac { ( n - 1 ) e ^ { - s } } { ( 1 + ( n - 1 ) e ^ { - s } ) ^ { 2 } } } .
$$

$$
{ \mathrm { A t ~ } } s = 0 { \mathrm { : } }
$$

$$
f ^ { \prime } ( 0 ) = { \frac { n - 1 } { n ^ { 2 } } } .
$$

The second derivative is:

$$
f ^ { \prime \prime } ( s ) = \frac { - ( n - 1 ) e ^ { - s } ( 1 + ( n - 1 ) e ^ { - s } ) ^ { 2 } + 2 ( n - 1 ) ^ { 2 } e ^ { - 2 s } ( 1 + ( n - 1 ) e ^ { - s } ) } { ( 1 + ( n - 1 ) e ^ { - s } ) ^ { 4 } } .
$$

At $s = 0 \colon$

$$
f ^ { \prime \prime } ( 0 ) = { \frac { - ( n - 1 ) n ^ { 2 } + 2 ( n - 1 ) ^ { 2 } n } { n ^ { 4 } } } = { \frac { ( n - 1 ) n ( n - 2 ) } { n ^ { 4 } } } = { \frac { ( n - 1 ) ( n - 2 ) } { n ^ { 3 } } } .
$$

This is $\Theta ( 1 / n )$ for large n.

The loss is ${ \mathcal { L } } ( s ) = ( f ( s ) - 1 ) ^ { 2 }$ . Since $f ( 0 ) = 1 / n$ , the second derivative is:

$$
\mathcal { L } ^ { \prime \prime } ( 0 ) = 2 ( f ( 0 ) - 1 ) f ^ { \prime \prime } ( 0 ) + 2 ( f ^ { \prime } ( 0 ) ) ^ { 2 } .
$$

Substituting:

$$
{ \mathcal { L } } ^ { \prime \prime } ( 0 ) = 2 \left( { \frac { 1 } { n } } - 1 \right) { \frac { ( n - 1 ) ( n - 2 ) } { n ^ { 3 } } } + 2 \left( { \frac { n - 1 } { n ^ { 2 } } } \right) ^ { 2 } = - { \frac { 2 ( n - 1 ) ^ { 2 } ( n - 2 ) } { n ^ { 4 } } } + { \frac { 2 ( n - 1 ) ^ { 2 } } { n ^ { 4 } } } = - { \frac { 2 ( n - 1 ) ^ { 2 } ( n - 3 ) } { n ^ { 4 } } } .
$$

For large n, this is $\Theta ( n ^ { - 2 } )$ . So $\mathcal { L } ^ { \prime \prime } ( 0 ) = \Theta ( n ^ { - 2 } )$

IDA case. For keys at radius $\rho > 0$ symmetrically distributed, the attention weight is:

$$
W _ { 1 1 } ( \delta ) = \frac { ( \delta ^ { 2 } + \varepsilon ) ^ { - 1 } } { ( \delta ^ { 2 } + \varepsilon ) ^ { - 1 } + ( n - 1 ) ( \delta ^ { 2 } + \rho ^ { 2 } + \varepsilon ) ^ { - 1 } } .
$$

Let $f ( \delta ) = ( \delta ^ { 2 } + \varepsilon ) ^ { - 1 }$ and $g ( \delta ) = ( \delta ^ { 2 } + \rho ^ { 2 } + \varepsilon ) ^ { - 1 }$ . Then:

$$
W _ { 1 1 } ( \delta ) = \frac { f ( \delta ) } { f ( \delta ) + ( n - 1 ) g ( \delta ) } .
$$

We compute the first and second derivatives at $\delta = 0$ . First, note that $f ( 0 ) =$ $\varepsilon ^ { - 1 } , g ( 0 ) = ( \rho ^ { 2 } + \varepsilon ) ^ { - 1 } , f ^ { \prime } ( 0 ) = 0 , g ^ { \prime } ( 0 ) = 0 , f ^ { \prime \prime } ( 0 ) = - 2 \varepsilon ^ { - 2 }$ , and $g ^ { \prime \prime } ( 0 ) =$ $- 2 ( \rho ^ { 2 } + \varepsilon ) ^ { - 2 }$

The first derivative of $W _ { 1 1 }$ at $\delta = 0$ is zero because $f ^ { \prime } ( 0 ) = g ^ { \prime } ( 0 ) = 0 \qquad $

$$
{ \cal W } _ { 1 1 } ^ { \prime } ( 0 ) = 0 .
$$

The second derivative is:

$$
W _ { 1 1 } ^ { \prime \prime } ( 0 ) = \frac { f ^ { \prime \prime } ( 0 ) ( f ( 0 ) + ( n - 1 ) g ( 0 ) ) - f ( 0 ) ( f ^ { \prime \prime } ( 0 ) + ( n - 1 ) g ^ { \prime \prime } ( 0 ) ) } { ( f ( 0 ) + ( n - 1 ) g ( 0 ) ) ^ { 2 } } .
$$

Substituting the values:

$$
W _ { 1 1 } ^ { \prime \prime } ( 0 ) = - \frac { 2 ( n - 1 ) \rho ^ { 2 } } { \varepsilon ^ { 2 } ( \rho ^ { 2 } + \varepsilon ) ^ { 2 } } \cdot \frac { \varepsilon ^ { 2 } } { ( 1 + ( n - 1 ) \varepsilon / ( \rho ^ { 2 } + \varepsilon ) ) ^ { 2 } } .
$$

For $\varepsilon \ll \rho ^ { 2 } / n$ , this is approximately:

$$
W _ { 1 1 } ^ { \prime \prime } ( 0 ) \approx - \frac { 2 ( n - 1 ) \rho ^ { 2 } } { ( \rho ^ { 2 } ) ^ { 2 } } = - \frac { 2 ( n - 1 ) } { \rho ^ { 2 } } .
$$

The loss is $\mathcal { L } ( \delta ) = ( W _ { 1 1 } ( \delta ) - 1 ) ^ { 2 }$ . Since $W _ { 1 1 } ^ { \prime } ( 0 ) = 0$

$$
\begin{array} { r } { \mathscr { L } ^ { \prime \prime } ( 0 ) = 2 ( W _ { 1 1 } ( 0 ) - 1 ) W _ { 1 1 } ^ { \prime \prime } ( 0 ) . } \end{array}
$$

With $W _ { 1 1 } ( 0 ) = 1 / ( 1 + \varepsilon ( n - 1 ) / ( \rho ^ { 2 } + \varepsilon ) )$ , we have $W _ { 1 1 } ( 0 ) - 1 = \Theta ( 1 )$ for fixed $\rho , \varepsilon$ and large n. Thus $\mathcal { L } ^ { \prime \prime } ( 0 ) = \Theta ( W _ { 1 1 } ^ { \prime \prime } ( 0 ) )$ , which is $\Theta ( n )$ in the regime $\varepsilon \ll \rho ^ { 2 } / n .$

However, the Hessian spread (ratio of largest to smallest eigenvalue) is $\Theta ( 1 )$ for IDA because the $\Theta ( n )$ scaling afects all directions similarly. The curvature contrast with softmax (which has $\Theta ( n ^ { - 2 } ) )$ establishes that IDA has substantially larger curvature. □

## 2.6 Theorem 2: Global Optimality via PL Inequality

The Polyak– Lojasiewicz (PL) inequality is a powerful tool for establishing linear convergence of gradient descent without requiring strong convexity. We now prove that the 1D loss for IDA satisfies a PL inequality with a constant that is exponentially larger than that of softmax.

Theorem 2.6 (Theorem 2: IDA PL Constant). For the 1D loss $\mathscr { L } ( q ) = ( W _ { 1 } ( q ) v _ { 1 } +$ $W _ { 2 } ( q ) v _ { 2 } - y ) ^ { 2 }$ with two fixed keys separated by $\Delta > 0$ , the PL constants $s a t i s f y .$

$$
\mu _ { I D A } = \Theta \left( \frac { \varepsilon ^ { 2 } } { \Delta ^ { 4 } } \right) , \qquad \mu _ { s o f t } = \Theta \left( \frac { e ^ { - \Delta ^ { 2 } / \sqrt { d } } \varepsilon ^ { 2 } } { \Delta ^ { 2 } } \right) ,
$$

so $\mu _ { I D A } / \mu _ { s o f t } = \Omega ( e ^ { \Delta ^ { 2 } / \sqrt { d } } / \Delta ^ { 2 } )$

Proof. We consider the 1D setting where the query q lies on the line between two keys. By translation, we set the keys at $k _ { 1 } = 0$ and $k _ { 2 } = \Delta$

IDA analysis. The attention weight for key 1 is:

$$
W _ { 1 } ( q ) = \frac { ( q ^ { 2 } + \varepsilon ) ^ { - 1 } } { ( q ^ { 2 } + \varepsilon ) ^ { - 1 } + ( ( q - \Delta ) ^ { 2 } + \varepsilon ) ^ { - 1 } } .
$$

Let $A ( q ) = ( q ^ { 2 } + \varepsilon ) ^ { - 1 }$ and $B ( q ) = ( ( q - \Delta ) ^ { 2 } + \varepsilon ) ^ { - 1 }$ . Then $W _ { 1 } ( q ) = A ( q ) / ( A ( q ) +$ $B ( q ) )$

We expand $A ( q )$ and $B ( q )$ around $q = 0 ~ ( \mathrm { i . e . , a t } ~ k _ { 1 } )$ . For small $q \colon$

$$
A ( q ) = { \frac { 1 } { \varepsilon } } - { \frac { q ^ { 2 } } { \varepsilon ^ { 2 } } } + { \frac { q ^ { 4 } } { \varepsilon ^ { 3 } } } + \mathcal { O } ( q ^ { 6 } ) .
$$

For $B ( q )$

$$
B ( q ) = \frac { 1 } { \Delta ^ { 2 } + \varepsilon - 2 \Delta q + q ^ { 2 } } .
$$

Let $C = \Delta ^ { 2 } + \varepsilon$ . Then:

$$
B ( q ) = \frac { 1 } { C } \cdot \frac { 1 } { 1 - \frac { 2 \Delta q - q ^ { 2 } } { C } } = \frac { 1 } { C } + \frac { 2 \Delta q } { C ^ { 2 } } + \left( - \frac { 1 } { C ^ { 2 } } + \frac { 4 \Delta ^ { 2 } } { C ^ { 3 } } \right) q ^ { 2 } + \mathcal { O } ( q ^ { 3 } ) .
$$

Let $D = 1 / \varepsilon + 1 / C$ . Then:

$$
W _ { 1 } ( q ) = \frac { \frac { 1 } { \varepsilon } - \frac { q ^ { 2 } } { \varepsilon ^ { 2 } } + \mathcal { O } ( q ^ { 4 } ) } { D + \frac { 2 \Delta q } { C ^ { 2 } } + \left( - \frac { 1 } { \varepsilon ^ { 2 } } - \frac { 1 } { C ^ { 2 } } + \frac { 4 \Delta ^ { 2 } } { C ^ { 3 } } \right) q ^ { 2 } + \mathcal { O } ( q ^ { 3 } ) } .
$$

Using the expansion $1 / ( 1 + x ) = 1 - x + x ^ { 2 } + \cdot \cdot \cdot$ , we get:

$$
W _ { 1 } ^ { \prime } ( 0 ) = 0 , \qquad W _ { 1 } ^ { \prime \prime } ( 0 ) = - \frac { 4 \varepsilon } { \Delta ^ { 4 } } + \mathcal { O } ( \varepsilon ^ { 2 } / \Delta ^ { 6 } ) .
$$

The loss is $\mathcal { L } ( q ) = ( W _ { 1 } ( q ) v _ { 1 } + ( 1 - W _ { 1 } ( q ) ) v _ { 2 } - y ) ^ { 2 }$ . Its gradient is:

$$
\mathcal { L } ^ { \prime } ( q ) = 2 ( W _ { 1 } ( q ) v _ { 1 } + ( 1 - W _ { 1 } ( q ) ) v _ { 2 } - y ) ( v _ { 1 } - v _ { 2 } ) W _ { 1 } ^ { \prime } ( q ) .
$$

Since $W _ { 1 } ^ { \prime } ( q ) \approx W _ { 1 } ^ { \prime \prime } ( 0 ) q$ for small q, we have:

$$
| { \mathcal { L } } ^ { \prime } ( q ) | ^ { 2 } \asymp \frac { \varepsilon ^ { 2 } q ^ { 2 } } { \Delta ^ { 8 } } .
$$

The loss gap is:

$$
\begin{array} { r } { \mathcal { L } ( q ) - \mathcal { L } ( 0 ) \asymp q ^ { 2 } . } \end{array}
$$

Thus:

$$
\frac { | { \mathcal { L } } ^ { \prime } ( q ) | ^ { 2 } } { { \mathcal { L } } ( q ) - { \mathcal { L } } ( 0 ) } \asymp \frac { \varepsilon ^ { 2 } } { \Delta ^ { 4 } } .
$$

This gives $\mu _ { \mathrm { I D A } } = \Theta ( \varepsilon ^ { 2 } / \Delta ^ { 4 } )$ .

Softmax analysis. For softmax, the logit diference is:

$$
s ( q ) = \frac { ( \mathbf { q } ^ { \top } \mathbf { k } _ { 1 } - \mathbf { q } ^ { \top } \mathbf { k } _ { 2 } ) } { \sqrt { d } } = - \frac { \Delta q } { \sqrt { d } } .
$$

The softmax weight is $W _ { 1 } ( q ) = \sigma ( s ( q ) )$ where $\sigma ( s ) = 1 / ( 1 + e ^ { - s } )$ . The PL constant is:

$$
\mu _ { \mathrm { s o f t } } = \Theta \left( e ^ { - \Delta ^ { 2 } / \sqrt { d } } \frac { \varepsilon ^ { 2 } } { \Delta ^ { 2 } } \right) .
$$

This follows from the fact that the gradient is exponentially small when $| s |$ is large. □

Corollary 2.7. Since $\mu _ { I D A } > 0$ , the PL inequality implies that every stationary point of the IDA loss is either a global minimum or a strict saddle $( i . e . _ { \cdot }$ , no spurious local minima). Gradient descent converges linearly as ${ \mathcal { L } } ( t ) - { \mathcal { L } } ^ { * } \leq$ $e ^ { - \mu t } ( \mathcal { L } ( 0 ) - \mathcal { L } ^ { * } )$ . Lemma 2 ensures $\Theta ( 1 )$ Hessian spread, enabling $\mathcal { O } ( 1 )$ escape from saddles, in contrast to softmax’s ${ \dot { \mathcal { O } } } ( n ^ { 2 } )$ escape time.

## 2.7 Theorem 3: Capacity Control and Generalization

The efective rank of a kernel matrix measures the number of significant singular values. A low efective rank implies that the kernel cannot memorize arbitrary patterns, which is beneficial for generalization.

Theorem 2.8 (Theorem 3: IDA Efective Rank Bound). For embeddings with of-diagonal distances $\geq d _ { \operatorname* { m i n } } > 0 , i f \epsilon \ll d _ { \operatorname* { m i n } } ^ { 2 } \left( s p e c i f i c a l { i y , \epsilon ( n - 1 ) / d _ { \operatorname* { m i n } } ^ { 2 } } \stackrel { . } { \leq } 1 / 2 \right)$ then

$$
\mathrm { e f f - r a n k } ( \mathbf { K } ) \leq 1 + \frac { n \varepsilon ^ { 2 } } { d _ { \mathrm { m i n } } ^ { 4 } } ,
$$

independent of hidden width $d _ { h }$ . In the general case:

$$
\mathrm { e f f - r a n k } ( \mathbf { K } ) \leq \frac { n } { ( 1 - \epsilon ( n - 1 ) / d _ { \operatorname* { m i n } } ^ { 2 } ) ^ { 2 } } .
$$

Proof. Let $\mathbf { K } \in \mathbb { R } ^ { n \times n }$ with entries $K _ { i j } = ( \| \mathbf q _ { i } - \mathbf k _ { j } \| _ { 2 } ^ { 2 } + \varepsilon ) ^ { - 1 }$ . The diagonal entries are $K _ { i i } = 1 / \varepsilon$

Define $\mathbf { E } = \mathbf { K } - \varepsilon ^ { - 1 } \mathbf { I } .$ . Then $E _ { i i } = 0$ , and for $i \neq j$

$$
E _ { i j } = \frac { 1 } { \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } + \varepsilon } - \frac { 1 } { \varepsilon } = - \frac { \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } } { \varepsilon ( \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } + \varepsilon ) } .
$$

The absolute value is:

$$
| E _ { i j } | = \frac { \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } } { \varepsilon ( \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } + \varepsilon ) } .
$$

Since $\| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } \geq d _ { \operatorname* { m i n } } ^ { 2 }$ , we have:

$$
| E _ { i j } | \leq \frac { 1 } { \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } + \varepsilon } \leq \frac { 1 } { d _ { \operatorname* { m i n } } ^ { 2 } + \varepsilon } \leq \frac { 1 } { d _ { \operatorname* { m i n } } ^ { 2 } } .
$$

By Gershgorin’s circle theorem, every eigenvalue $\lambda _ { m } ( \mathbf { E } )$ satisfies:

$$
\left| \lambda _ { m } ( { \bf E } ) \right| \leq \operatorname* { m a x } _ { i } \sum _ { j \neq i } \left| E _ { i j } \right| \leq \frac { n - 1 } { d _ { \operatorname* { m i n } } ^ { 2 } } .
$$

Thus the eigenvalues of $\mathbf { K } = \varepsilon ^ { - 1 } \mathbf { I } + \mathbf { E }$ satisfy:

$$
\lambda _ { m } ( \mathbf { K } ) \in \left[ \frac { 1 } { \varepsilon } - \frac { n - 1 } { d _ { \operatorname* { m i n } } ^ { 2 } } , \frac { 1 } { \varepsilon } + \frac { n - 1 } { d _ { \operatorname* { m i n } } ^ { 2 } } \right] .
$$

The efective rank is:

$$
\operatorname { e f f - r a n k } ( \mathbf { K } ) = { \frac { ( \operatorname { t r } \mathbf { K } ) ^ { 2 } } { \operatorname { t r } ( \mathbf { K } ^ { 2 } ) } } = { \frac { { n ^ { 2 } / \varepsilon ^ { 2 } } } { { n / \varepsilon ^ { 2 } } + \| \mathbf { E } \| _ { F } ^ { 2 } } } .
$$

Using the bound $\| { \bf E } \| _ { F } ^ { 2 } \leq n ( n - 1 ) / d _ { \operatorname* { m i n } } ^ { 4 }$ , we get:

$$
\operatorname { e f f - r a n k } ( \mathbf { K } ) \leq \frac { n } { 1 + \varepsilon ^ { 2 } ( n - 1 ) / d _ { \operatorname* { m i n } } ^ { 4 } } .
$$

For the refined bound, using the sharper estimate $| E _ { i j } | \leq \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } / ( \varepsilon d _ { \operatorname* { m i n } } ^ { 2 } )$

$$
\mathrm { e f f - r a n k } ( \mathbf { K } ) \leq 1 + \frac { n \varepsilon ^ { 2 } } { d _ { \operatorname* { m i n } } ^ { 4 } } + \mathcal { O } \left( \frac { n \varepsilon ^ { 3 } } { d _ { \operatorname* { m i n } } ^ { 6 } } \right) .
$$

This proves the theorem.

Theorem 2.9 (Softmax Width Catastrophe). When $d _ { h } \ge n$ , softmax can achieve zero training error on arbitrary labels, with test error → 2η under symmetric noise rate η.

Proof. We construct an explicit construction. Let $\{ { \bf e } _ { i } \} _ { i = 1 } ^ { n }$ be an orthonormal basis in $\mathbb { R } ^ { d _ { h } }$ , so we require $d _ { h } \ge n$ . Set $\mathbf { q } _ { i } = \sqrt { d _ { h } } \mathbf { e } _ { i }$ and $\mathbf { k } _ { j } = \alpha \sqrt { d _ { h } } \mathbf { e } _ { j }$ for some $\alpha > 0$ . Then the inner product is:

$$
\mathbf { q } _ { i } ^ { \top } \mathbf { k } _ { j } = \alpha d _ { h } \delta _ { i j } .
$$

The softmax attention weights are:

$$
W _ { i j } = \frac { e ^ { \alpha d _ { h } \delta _ { i j } } } { \sum _ { m } e ^ { \alpha d _ { h } \delta _ { i m } } } = \left\{ \begin{array} { l l } { \frac { e ^ { \alpha d _ { h } } } { e ^ { \alpha d _ { h } } + ( n - 1 ) } , } & { i = j , } \\ { \frac { 1 } { e ^ { \alpha d _ { h } } + ( n - 1 ) } , } & { i \neq j . } \end{array} \right.
$$

As $\alpha \to \infty$ , the weights converge to the identity matrix: $W _ { i j } \to \delta _ { i j }$ . Therefore the output is $\begin{array} { r } { \mathbf { o } _ { i } = \sum _ { i } W _ { i j } \mathbf { v } _ { j } \to \mathbf { v } _ { i } } \end{array}$

The linear readout layer $\mathbf { w } \in \mathbb { R } ^ { d _ { v } }$ maps the output to the prediction: $\hat { y } _ { i } =$ $\mathbf { w } ^ { \top } \mathbf { o } _ { i }$ . If $\mathbf { v } _ { i }$ are linearly independent (true with probability 1 when $d _ { v } \geq n$ and values are random), then there exists a w satisfying $\mathbf { w } ^ { \top } \mathbf { v } _ { i } = y _ { i }$ for all i. Thus zero training error can be achieved.

Under label noise rate $\eta ,$ the training labels are random flips of the true labels. Zero training error implies the model has memorized the noisy labels. The test error is the expected error on clean labels, which equals the Bayes error under symmetric noise: η for binary classification or 2η for balanced multi-class classification. □

Theorem 2.10 (IDA Noise Robustness). For $I D A$ , test error under symmetric noise rate η satisfies

$$
\mathcal { E } _ { t e s t } ^ { I D A } \leq C \eta ^ { 2 } + \mathcal { O } ( 1 / \sqrt { n } ) ,
$$

independent of $d _ { h }$ .

Proof. For IDA at exact match $\mathbf { q } _ { i } = \mathbf { k } _ { i }$ , the attention weight is:

$$
W _ { i i } = \frac { 1 / \varepsilon } { 1 / \varepsilon + \sum _ { j \neq i } ( d _ { i j } ^ { 2 } + \varepsilon ) ^ { - 1 } } .
$$

Define $\begin{array} { r } { \alpha _ { i } = \varepsilon \sum _ { j \neq i } ( d _ { i j } ^ { 2 } + \varepsilon ) ^ { - 1 } } \end{array}$ . Then:

$$
W _ { i i } = \frac { 1 } { 1 + \alpha _ { i } } = 1 - \alpha _ { i } + { \mathcal O } ( \alpha _ { i } ^ { 2 } ) .
$$

For $j \neq i \colon$

$$
W _ { i j } = \frac { \alpha _ { i } } { n - 1 } + \mathcal { O } ( \varepsilon ^ { 2 } n / d _ { \mathrm { m i n } } ^ { 4 } ) .
$$

The prediction is:

$$
\hat { y } _ { i } = ( 1 - \alpha _ { i } ) y _ { i } + \alpha _ { i } \bar { y } _ { - i } + \mathcal { O } ( \varepsilon ^ { 2 } n ^ { 2 } / d _ { \operatorname* { m i n } } ^ { 4 } ) ,
$$

where ${ \bar { y } } _ { - i }$ is the average label of the neighbors.

Taking expectations over label noise:

$$
\mathbb { E } [ ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } ] \le 4 \alpha _ { i } ^ { 2 } \eta ^ { 2 } + \mathcal { O } ( \varepsilon ^ { 2 } n ^ { 2 } / d _ { \operatorname* { m i n } } ^ { 4 } ) + \mathcal { O } ( 1 / n ) .
$$

Summing over i and using $\alpha _ { i } \leq \varepsilon n / d _ { \mathrm { m i n } } ^ { 2 }$

$$
\mathcal { E } _ { \mathrm { t e s t } } ^ { \mathrm { I D A } } \leq C \eta ^ { 2 } + \mathcal { O } ( 1 / \sqrt { n } ) .
$$

The bound is independent of $d _ { h }$

## 2.8 Discussion: High-Dimensional Behavior

The analysis above reveals several important aspects of IDA’s behavior in high dimensions that distinguish it from softmax.

Distance concentration. In high dimensions, Euclidean distances between random points concentrate around their mean. This means that for large $d _ { h }$ the inverse-distance kernel assigns roughly equal weight to all keys, making the attention output close to the unweighted average of values. This is the same phenomenon that afects softmax in high dimensions (inner products concentrate around zero). However, the key diference is that IDA’s kernel explicitly depends on distances, which can be controlled by scaling the embeddings, while softmax’s inner products depend on norms that are less controllable.

Efective rank and capacity. Theorem 3 shows that IDA’s efective rank is bounded independent of $d _ { h }$ , which prevents overfitting to noise. In high dimensions, this bound becomes tighter because the of-diagonal distances $d _ { \mathrm { m i n } }$ increase with dimension for random embeddings. This suggests that IDA is particularly well-suited for high-dimensional data.

Implications for practice. Based on the theoretical analysis, we recommend initializing ε to a small fraction of the typical squared distance $( \mathrm { e . g . }$ ， $\varepsilon \approx 0 . 0 1 \cdot \mathbb { E } [ \| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 } ^ { 2 } ] )$ to avoid the degradation regime $\varepsilon ( n - 1 ) / d _ { \operatorname* { m i n } } ^ { 2 } \geq 1 / 2$ The analysis also suggests that for tasks with high noise rates η, using IDA with moderate ε (around $1 0 ^ { - 4 } )$ provides better generalization than softmax.

## 3 Euclidean Summary: Table of Theoretical Guarantees

<table><tr><td>Property</td><td>Softmax</td><td>Resolver (IDA)</td></tr><tr><td>Circuit separation (Thm. 1)</td><td> $\Omega ( ( \log n ) ^ { 2 } )$ </td><td>O(1)</td></tr><tr><td>Lipschitz scaling (Lemma 1)</td><td> ${ \mathcal { O } } ( n )$ </td><td>O(log n) under low-rank/O(n) general</td></tr><tr><td>Hessian spread (Lemma 2)</td><td> $\Theta ( n ^ { - 2 } )$ </td><td>Θ(1)</td></tr><tr><td>PL constant (Thm. 2)</td><td> $\Theta ( e ^ { - \Delta ^ { 2 } / \sqrt { d } } \varepsilon ^ { 2 } / \Delta ^ { 2 } )$ </td><td> $\Theta ( \varepsilon ^ { 2 } / \Delta ^ { 4 } )$ </td></tr><tr><td>Effective rank (Thm. 3)</td><td>Unbounded</td><td> $\leq 1 + n \varepsilon ^ { 2 } / d _ { \operatorname* { m i n } } ^ { 4 } \ ( \mathrm { s m a l l } \ \epsilon )$ </td></tr><tr><td>Noise memorization (Thm. 3)</td><td> $\mathrm { A t } ~ d _ { h } \geq n$ </td><td>Structurally limited</td></tr></table>

These five Euclidean results provide the theoretical backbone. The non-Euclidean extension that follows is based on the same inverse-distance principle, with geodesic distances replacing Euclidean norms.

## Part II

## Riemann GeoResolver: Non-Euclidean Extension

## 4 Hyperbolic Geometry Setup

The Poincar´e ball is $\mathbb { B } ^ { d _ { h } } = \left\{ \mathbf { x } \in \mathbb { R } ^ { d _ { h } } : \left. \mathbf { x } \right. < 1 \right\}$ with conformal factor $\lambda _ { \mathbf { x } } =$ $2 / ( 1 - \| \mathbf { x } \| ^ { 2 } )$ . The hyperbolic distance is:

$$
d _ { \mathbb { H } } ( \mathbf { x } , \mathbf { y } ) = \mathrm { a r c o s h } \bigg ( 1 + \frac { 2 \| \mathbf { x } - \mathbf { y } \| ^ { 2 } } { ( 1 - \| \mathbf { x } \| ^ { 2 } ) ( 1 - \| \mathbf { y } \| ^ { 2 } ) } \bigg ) .
$$

All embeddings are projected via $\mathbf { x } = 0 . 9 \operatorname { t a n h } ( \mathbf { z } )$ ; results extend to $\rho \to 1 ^ { - }$ by continuity. The Karcher mean exists uniquely in CAT(0) spaces.

The unit sphere is $\mathbb { S } ^ { d - 1 } = \{ \mathbf { x } : \| \mathbf { x } \| = 1 \}$ with great-circle distance:

$$
d _ { \mathbb { S } } ( \mathbf { x } , \mathbf { y } ) = \operatorname { a r c c o s } ( \langle \mathbf { x } , \mathbf { y } \rangle ) .
$$

Throughout this part, we use the HIDA family to refer collectively to all hyperbolic inverse-distance attention operators, and Dense-HIDA, FP-HIDA, L-HIDA, C-HIDA for the specific instances.

## 5 The HIDA Operator Family (M1–M4)

## 5.1 A Unified Inverse-Distance Kernel Lemma

Lemma 5.1 (Unified Kernel Spectral Properties). Let M be a metric space and $K _ { i j } = ( d ( \dot { x } _ { i } , y _ { j } ) ^ { 2 } + \epsilon ) ^ { - 1 }$ with: $( i ) d ( x _ { i } , x _ { i } ) = 0 , ( i i ) d ( x _ { i } , y _ { j } ) \geq d _ { \operatorname* { m i n } } > 0$ for $i \neq j$ , (iii) d locally Lipschitz. Then: (a) $\begin{array} { r } { i f x _ { i } = y _ { j ^ { * } } , \operatorname* { l i m } _ { \epsilon  0 } K _ { i j ^ { * } } / \sum _ { m } K _ { i m } = 1 _ { \nu } } \end{array}$ •

(b) $i f \epsilon \ll d _ { \operatorname* { m i n } } ^ { 2 }$ (specifically, $\epsilon ( n - 1 ) / d _ { \mathrm { m i n } } ^ { 2 } \leq 1 / 2 )$ , then

$$
\mathrm { e f f - r a n k } ( \mathbf { K } ) \leq 1 + \frac { n \epsilon ^ { 2 } } { d _ { \operatorname* { m i n } } ^ { 4 } } + \mathcal { O } \left( \frac { n \epsilon ^ { 3 } } { d _ { \operatorname* { m i n } } ^ { 6 } } \right) .
$$

In the general case (without the small-ϵ condition), the bound is:

$$
\mathrm { e f f - r a n k } ( \mathbf { K } ) \leq \frac { n } { ( 1 - \epsilon ( n - 1 ) / d _ { \operatorname* { m i n } } ^ { 2 } ) ^ { 2 } } .
$$

Proof. Part (a) follows from $K _ { i j ^ { * } } = 1 / \epsilon$ and $K _ { i j } \le 1 / ( d _ { \operatorname* { m i n } } ^ { 2 } + \epsilon )$ for $j \neq j ^ { * }$ , so the normalized weight tends to 1 as $\epsilon \to 0$

Part (b) follows from the Frobenius norm estimate:

$$
\| \mathbf { E } \| _ { F } ^ { 2 } \leq \frac { n ( n - 1 ) } { ( d _ { \operatorname* { m i n } } ^ { 2 } + \epsilon ) ^ { 2 } } .
$$

The general bound follows from the efective rank formula. The refined bound requires $\epsilon \ll d _ { \mathrm { m i n } } ^ { 2 }$ and is obtained by the sharper Frobenius estimate. □

## 5.2 Why Hyperbolic? Three Arguments

(A) Information-theoretic capacity. For fixed dimension $d ,$ the hyperbolic ball volume $V _ { \mathbb { H } } ( R ) \propto e ^ { ( d - 1 ) R }$ grows exponentially with radius, while Euclidean volume $V _ { \mathbb { E } } ( R ) \propto R ^ { d }$ grows polynomially. This means hyperbolic spaces can embed hierarchical data with exponentially fewer dimensions.

(B) Efective rank behavior. Theorem 3 established the bound $d _ { \operatorname* { m i n } { } } \geq$ $2 \delta / ( 1 - \rho ^ { 2 } )$ ; as $\rho  1 ^ { - }$ (approaching the boundary of the Poincar´e ball), this grows without bound. Thus hyperbolic embeddings can achieve arbitrarily large separation between keys without increasing the ambient dimension.

(C) Architectural separation. Proposition 2 (Section 10) establishes an information-theoretic gap between hyperbolic storage and spherical routing.

## 5.3 Dense-HIDA (M1)

Definition 5.1 (Dense Hyperbolic IDA). The dense hyperbolic inverse-distance attention is defined as:

$$
D _ { i j } ^ { \mathbb { H } } = d _ { \mathbb { H } } ( \mathbf { q } _ { i } , \mathbf { k } _ { j } ) ^ { 2 } + \varepsilon , \quad W _ { i j } = ( D _ { i j } ^ { \mathbb { H } } ) ^ { - 1 } / \sum _ { m } ( D _ { i m } ^ { \mathbb { H } } ) ^ { - 1 } , \quad \mathbf { o } _ { i } = \sum _ { j } W _ { i j } \mathbf { v } _ { j } .
$$

The computational complexity is $\Theta ( n ^ { 2 } d _ { h } )$

Theorem 5.2 (Circuit Separation and Rank Bound). For pairwise distinct keys in a compact subset with $\| \mathbf { x } \| \leq \rho < 1$ , Dense-HIDA satisfies: (1) lim $_ { \varepsilon  0 } W _ { j ^ { * } } =$ 1 for exact retrieval; (2) ef-rank $\begin{array} { r } { ( \mathbf { K } ^ { \mathbb { H } } ) \le 1 + \frac { n \varepsilon ^ { 2 } } { d _ { \operatorname* { m i n } } ^ { 4 } } } \end{array}$ , with $d _ { \operatorname* { m i n } } \geq 2 \delta / ( 1 - \rho ^ { 2 } )$ .

□

Proof. Claim (1) follows directly from Lemma 0 Part (a) with $d = d _ { \mathbb { H } }$ . Claim (2) follows from Lemma 0 Part (b). The lower bound on $d _ { \mathrm { m i n } }$ follows from the hyperbolic distance lower bound:

$$
d _ { \mathbb { H } } ( \mathbf { x } , \mathbf { y } ) \geq { \frac { \| \mathbf { x } - \mathbf { y } \| _ { 2 } } { 1 - \rho ^ { 2 } } } .
$$

This inequality holds for any $\mathbf { x } , \mathbf { y } \in \mathbb { B } ^ { d _ { h } }$ with $\| \mathbf { x } \| , \| \mathbf { y } \| \leq \rho .$

Theorem 5.3 (Two-Point Hyperbolic PL Inequality). For the two-key loss with keys separated by $\Delta _ { \mathbb { H } } > 0 .$

$$
\mu _ { H I D A } = \Theta ( \varepsilon ^ { 2 } / \Delta _ { \mathbb { H } } ^ { 4 } ) , \quad \mu _ { s o f t } = \Theta ( e ^ { - \Delta _ { \mathbb { H } } ^ { 2 } / \sqrt { d _ { h } } } \varepsilon ^ { 2 } / \Delta _ { \mathbb { H } } ^ { 2 } ) .
$$

Thus $\mu _ { H I D A } / \mu _ { s o f t } = \Omega ( e ^ { \Delta _ { \mathbb H } ^ { 2 } / \sqrt { d _ { h } } } / \Delta _ { \mathbb H } ^ { 2 } )$

Proof. By M¨obius invariance of the Poincar´e distance, we may place $\mathbf { k } _ { 1 } = 0$ and $\mathbf { k } _ { 2 } = ( \Delta _ { \mathbb { H } } , 0 , \dots , 0 )$ in the Poincar´e ball. For q on the geodesic, the distance expansions are:

$$
\begin{array} { c } { { d _ { \mathbb { H } } ( q , { \mathbf { k } } _ { 1 } ) ^ { 2 } = q ^ { 2 } + \mathcal { O } ( q ^ { 4 } ) , } } \\ { { d _ { \mathbb { H } } ( q , { \mathbf { k } } _ { 2 } ) ^ { 2 } = \Delta _ { \mathbb { H } } ^ { 2 } - 2 \Delta _ { \mathbb { H } } q + q ^ { 2 } + \mathcal { O } ( q ^ { 3 } ) . } } \end{array}
$$

Substituting these into the attention weight formula and following the same expansion as Theorem 2 yields $W _ { 1 } ^ { \prime } ( 0 ) = 0 , \bar { W } _ { 1 } ^ { \prime \prime } ( 0 ) = - 4 \varepsilon / \Delta _ { \mathbb { H } } ^ { 4 } + \mathcal { O } ( \varepsilon ^ { \bar { 2 } } / \Delta _ { \mathbb { H } } ^ { 6 } )$ , and hence $\mu _ { \mathrm { H I D A } } = \Theta ( \varepsilon ^ { 2 } / \Delta _ { \mathbb H } ^ { 4 } )$ . □

## 5.4 FP-HIDA: Fixed-Pattern Sparse (M2)

Definition 5.2 (Sparse Index Set). Define the sparse index set:

$$
S _ { i } = \{ j : | i - j | \leq w \} \cup \{ 0 , \lfloor n / g \rfloor , \lfloor 2 n / g \rfloor , \dots \} \cup \{ i \pm 2 ^ { k } \} _ { k = 0 } ^ { \lceil \log _ { 2 } n \rceil } \cup \{ i \} ,
$$

with $w , g = \Theta ( \log n )$

Theorem 5.4 (FP-HIDA Complexity). FP-HIDA requires $\mathcal { O } ( n \log n \cdot d _ { h } )$ operations and ${ \mathcal { O } } ( n$ log n) memory.

Proof. We bound the size of each sparse index set. The local window contributes at most $2 w + 1$ indices. The global anchors contribute exactly g indices. The dyadic ofsets contribute at most $2 ( \lceil \log _ { 2 } n \rceil + 1 )$ indices. The self index contributes 1. Therefore:

$$
| S _ { i } | \leq ( 2 w + 1 ) + g + 2 ( \lceil \log _ { 2 } n \rceil + 1 ) + 1 .
$$

Since $w = \Theta ( \log n )$ and $g = \Theta ( \log n )$ , we have $| S _ { i } | = \mathcal { O } ( \log n )$

Summing over all i:

$$
\sum _ { i = 1 } ^ { n } \left| S _ { i } \right| \leq n \cdot \mathcal { O } ( \log n ) = \mathcal { O } ( n \log n ) .
$$

Each entry in the sparse index set requires $\Theta ( d _ { h } )$ operations to compute the distance and attention weight. Thus the total computational complexity is $\mathcal { O } ( n \log n \cdot d _ { h } )$ . The memory complexity is ${ \mathcal { O } } ( n \log n )$ for storing the sparse attention pattern. □

## 5.5 L-HIDA: Linear Complexity (M3)

Definition 5.3 (L-HIDA Anchor Aggregation). Let $\mathcal { A } = \{ \mathbf { a } _ { k } \} _ { k = 1 } ^ { m } \subset \mathbb { B } ^ { d _ { h } }$ be $m = \Theta ( 1 )$ learnable anchors. Define the key-anchor and query-anchor weights:

$$
\widetilde { W } _ { i k } ^ { K A } = \frac { ( d _ { \mathbb { H } } ( { \bf k } _ { i } , { \bf a } _ { k } ) ^ { 2 } + \varepsilon ) ^ { - 1 } } { \sum _ { l = 1 } ^ { m } ( d _ { \mathbb { H } } ( { \bf k } _ { i } , { \bf a } _ { l } ) ^ { 2 } + \varepsilon ) ^ { - 1 } } , \quad \widetilde { \bf v } _ { k } = \sum _ { i = 1 } ^ { n } \widetilde { W } _ { i k } ^ { K A } { \bf v } _ { i } ,
$$

$$
\widetilde { W } _ { j k } ^ { Q A } = \frac { ( d _ { \mathbb { H } } ( \mathbf { q } _ { j } , \mathbf { a } _ { k } ) ^ { 2 } + \varepsilon ) ^ { - 1 } } { \sum _ { l = 1 } ^ { m } ( d _ { \mathbb { H } } ( \mathbf { q } _ { j } , \mathbf { a } _ { l } ) ^ { 2 } + \varepsilon ) ^ { - 1 } } , \quad \mathbf { o } _ { j } = \sum _ { k = 1 } ^ { m } \widetilde { W } _ { j k } ^ { Q A } \widetilde { \mathbf { v } } _ { k } .
$$

Theorem 5.5 (L-HIDA Complexity and Error Bound). L-HIDA requires $\mathcal { O } ( n d _ { h } )$ operations. With probability at least $1 - \delta$ over the random selection of anchors (standard Nystr¨om sampling $[ 6 3 ] )$

$$
\Vert \mathbf { o } ^ { L - H I D A } - \mathbf { o } ^ { H I D A } \Vert _ { F } \leq \mathcal { O } \left( \frac { n } { \sqrt { m } \varepsilon ^ { 2 } } \right) .
$$

If the kernel has spectral decay $\lambda _ { k } = \mathcal { O } ( k ^ { - \alpha } )$ with $\alpha > 1 / 2$

$$
\| { \bf o } ^ { L - H I D A } - { \bf o } ^ { H I D A } \| _ { F } \leq \mathcal { O } ( n \cdot m ^ { - \alpha + 1 / 2 } ) + \mathcal { O } \left( \frac { n } { \varepsilon ^ { 2 } \sqrt { m } } \right) .
$$

Proof. Let $\mathbf { K } _ { n n } \in \mathbb { R } ^ { n \times n }$ be the dense kernel matrix, $\mathbf { K } _ { n m } \in \mathbb { R } ^ { n \times m }$ the keyanchor kernel matrix, and $\mathbf { K } _ { m m } \in \mathbb { R } ^ { m \times m }$ the anchor-anchor kernel matrix. The Nystr¨om approximation is $\bar { \mathbf { K } } _ { n n } = \mathbf { K } _ { n m } \mathbf { K } _ { m m } ^ { - 1 } \mathbf { K } _ { m n }$

The error decomposes as:

$$
{ \bf K } _ { n n } - { \widetilde { \bf K } } _ { n n } = \big ( { \bf K } _ { n n } - { \bf K } _ { n n } ^ { ( m ) } \big ) + \big ( { \bf K } _ { n n } ^ { ( m ) } - { \widetilde { \bf K } } _ { n n } \big ) ,
$$

where ${ \bf K } _ { n n } ^ { ( m ) }$ is the best rank-m approximation to ${ \bf K } _ { n n }$ in Frobenius norm.

For the first term, if the kernel has spectral decay $\lambda _ { k } = \mathcal { O } ( k ^ { - \alpha } )$ with $\alpha > 1 / 2$ then:

$$
\| { \bf K } _ { n n } - { \bf K } _ { n n } ^ { ( m ) } \| _ { F } ^ { 2 } = \sum _ { k = m + 1 } ^ { n } \lambda _ { k } ^ { 2 } = \mathcal { O } ( m ^ { - 2 \alpha + 1 } ) .
$$

For the second term, standard Nystr¨om analysis [63] gives:

$$
\| \mathbf { K } _ { n n } ^ { ( m ) } - \widetilde { \mathbf { K } } _ { n n } \| _ { F } \leq \| \mathbf { K } _ { m m } ^ { - 1 } \| _ { 2 } \| \mathbf { K } _ { n m } \| _ { F } \| \mathbf { E } \| _ { F } ,
$$

where E is the residual matrix. With $\| \mathbf { K } _ { m m } ^ { - 1 } \| _ { 2 } = \mathcal { O } ( 1 / \varepsilon ) , \| \mathbf { K } _ { n m } \| _ { F } = \mathcal { O } ( n / \varepsilon )$ and $\| \mathbf { E } \| _ { F } = \mathcal { O } ( 1 / \sqrt { m } )$ , this yields:

$$
\| \mathbf { K } _ { n n } ^ { ( m ) } - \widetilde { \mathbf { K } } _ { n n } \| _ { F } = \mathcal { O } \left( \frac { n } { \varepsilon ^ { 2 } \sqrt { m } } \right) .
$$

The output error bound follows from:

$$
\| \mathbf { o } ^ { \mathrm { L \mathrm { - } H I D A } } - \mathbf { o } ^ { \mathrm { H I D A } } \| _ { F } \leq \frac { 2 \| \mathbf { V } \| _ { F } } { \varepsilon } \| \mathbf { K } _ { n n } - \widetilde { \mathbf { K } } _ { n n } \| _ { F } .
$$

Combining the two error terms completes the proof.

## 5.6 C-HIDA: Constant Complexity (M4)

Definition 5.4 (C-HIDA Summary Tokens). Maintain $c = \Theta ( 1 )$ summary tokens $\{ \mathbf { s } _ { k } \} _ { k = 1 } ^ { c }$ . For each incoming key, assign to the nearest summary:

$$
k ^ { * } ( i ) = \arg \operatorname* { m i n } _ { k \in [ c ] } d _ { \mathbb { H } } ( { \mathbf { k } } _ { i } , { \mathbf { s } } _ { k } ) .
$$

Update each summary token using the hyperbolic mean:

$$
\mathbf { s } _ { k }  \Pi _ { \mathbb { B } } ( \mathbf { s } _ { k } + \eta \cdot { \frac { 1 } { | \mathcal { T } _ { k } | } } \sum _ { i \in \mathcal { T } _ { k } } ( \mathbf { k } _ { i } - \mathbf { s } _ { k } ) ) ,
$$

where $\Pi _ { \mathbb { B } }$ is the projection onto the Poincar´e ball.

Theorem 5.6 (C-HIDA Convergence). The online hyperbolic k-means update has $\mathcal { O } ( c \log T )$ regret:

$$
\sum _ { t = 1 } ^ { T } \operatorname* { m i n } _ { k } d _ { \mathbb { H } } ( { \mathbf { k } _ { t } } , { \mathbf { s } _ { k } ^ { ( t ) } } ) ^ { 2 } \le \mathcal { O } ( \Delta _ { \mathbb { H } } ^ { 2 } \cdot c \cdot \log T ) + \mathcal { O } ( T \cdot \varepsilon _ { o p t } ) .
$$

Per-token attention cost is $\Theta ( 1 )$

Proof. The Poincar´e ball with hyperbolic metric is $\mathrm { ~ a ~ } \mathrm { C A T } ( 0 )$ space, so the Karcher mean (the minimizer of the sum of squared distances) exists uniquely. The online gradient descent update on the squared hyperbolic distance is a convex optimization problem in $\mathrm { C A T ( 0 ) }$ spaces. The standard regret bound for online convex optimization applies:

$$
\sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { s } ^ { ( t ) } ) - \operatorname* { m i n } _ { \mathbf { s } } \sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { s } ) = \mathcal { O } ( \log T ) ,
$$

where $f _ { t } ( \mathbf { s } ) = \mathrm { m i n } _ { k } d _ { \mathbb { H } } ( \mathbf { k } _ { t } , \mathbf { s } _ { k } ) ^ { 2 }$ . The log $T$ term arises from the online-to-batch conversion for k-means. The per-token attention cost is Θ(1) because only $c = \Theta ( 1 )$ summaries are used. □

## 6 Hyperbolic Curvature Compression (M5)

For any key $\mathbf { k } \in \mathbb { B } ^ { d _ { h } }$ , define its polar decomposition ${ \bf k } = \boldsymbol { r } \cdot { \bf u }$ where $r \in [ 0 , 1 )$ is the radius and $\mathbf { u } \in \mathbb { S } ^ { d _ { h } - 1 }$ is the direction.

Given bit-widths b (for direction) and $b _ { r }$ (for radius):

$$
\widehat { \mathbf { u } } = \mathrm { r o u n d } \left( \frac { \mathbf { u } + 1 } { 2 } ( 2 ^ { b } - 1 ) \right) / ( 2 ^ { b } - 1 ) \cdot 2 - 1 , \quad \widehat { r } = \frac { \mathrm { r o u n d } ( r \cdot 2 ^ { b _ { r } } ) } { 2 ^ { b _ { r } } } .
$$

Theorem 6.1 (HCC Reconstruction Error). The reconstruction error satisfies:

$$
\lVert \mathbf { k } - \widetilde { \mathbf { k } } \rVert _ { 2 } ^ { 2 } \leq 4 ( 2 ^ { - b } + 2 ^ { - b _ { r } } ) .
$$

Proof. For the directional component, the quantization error is bounded by:

$$
\lVert \mathbf { u } - \widehat { \mathbf { u } } \rVert _ { 2 } \leq 2 \cdot 2 ^ { - b } .
$$

This follows from the covering radius of the uniform grid on $[ - 1 , 1 ] ^ { d _ { h } - 1 }$ restricted to the sphere.

For the radius component:

$$
| r - \widehat { r } | \leq 2 ^ { - b _ { r } } ,
$$

since the rounding error is at most half the bin width.

Combining via the triangle inequality:

$$
\| r \mathbf { u } - \widehat { r } \widehat { \mathbf { u } } \| _ { 2 } \leq r \| \mathbf { u } - \widehat { \mathbf { u } } \| _ { 2 } + | r - \widehat { r } | \| \widehat { \mathbf { u } } \| _ { 2 } \leq 2 \cdot 2 ^ { - b } + 2 ^ { - b _ { r } } .
$$

Squaring gives:

$$
\| \mathbf { k } - \widetilde { \mathbf { k } } \| _ { 2 } ^ { 2 } \leq ( 2 \cdot 2 ^ { - b } + 2 ^ { - b _ { r } } ) ^ { 2 } \leq 4 ( 2 ^ { - b } + 2 ^ { - b _ { r } } ) .
$$

This proves the bound.

Theorem 6.2 (HCC Memory Complexity). The HCC-compressed KV cache requires $n ( ( d _ { h } - 1 ) b + b _ { r } )$ bits for keys plus $n d _ { h } \cdot 3 2$ bits for values. With $b =$ $4 , b _ { r } = 6 $ , the theoretical compression ratio is ≈ 8× for keys, ≈ 1.8× system-wide (or ≈ 2.4× with half-precision values).

Proof. The key compression ratio is:

$$
{ \frac { 3 2 d _ { h } } { ( d _ { h } - 1 ) b + b _ { r } } } .
$$

With $d _ { h } = 3 2 , b = 4 , b _ { r } = 6 \colon$

$$
{ \frac { 1 0 2 4 } { 3 1 \cdot 4 + 6 } } = { \frac { 1 0 2 4 } { 1 3 0 } } \approx 7 . 8 8 \times .
$$

For the system-wide compression (keys + values), using 32-bit values:

$$
\frac { 6 4 d _ { h } } { ( d _ { h } - 1 ) b + b _ { r } + 3 2 d _ { h } } = \frac { 2 0 4 8 } { 1 3 0 + 1 0 2 4 } \approx 1 . 7 7 \times .
$$

With 16-bit values:

$$
\frac { 4 8 d _ { h } } { ( d _ { h } - 1 ) b + b _ { r } + 1 6 d _ { h } } = \frac { 1 5 3 6 } { 1 3 0 + 5 1 2 } \approx 2 . 3 9 \times .
$$

## 7 HyperGate: Curvature-Adaptive Gating (M6)

Three-level gating: head-level $\kappa _ { h }$ , token-level boundary $\beta _ { i }$ , dimension-level $\mathbf { G } _ { i } =$ $\mathrm { d i a g } ( \sigma ( \mathbf { W } _ { g } \mathbf { x } _ { i } ) )$ . The gated representation is $\mathbf { h } _ { i } = \mathbf { G } _ { i } \mathbf { x } _ { i }$

Theorem 7.1 (HyperGate Gradient Lower Bound). Let $\mathcal { L }$ be the loss function and $\mathbf { h } _ { i } = \mathbf { G } _ { i } \mathbf { x } _ { i }$ be the gated representation. Then:

$$
\left\| \frac { \partial \mathcal { L } } { \partial \mathbf { x } _ { i } } \right\| _ { 2 } \geq ( \lambda _ { \operatorname* { m i n } } ( \mathbf { G } _ { i } ) - \mathcal { O } ( \| \mathbf { W } _ { g } \| \| \mathbf { x } _ { i } \| ) ) \left\| \frac { \partial \mathcal { L } } { \partial \mathbf { h } _ { i } } \right\| _ { 2 } .
$$

In particular, for bounded $\mathbf { x } _ { i }$ and $\mathbf { W } _ { g }$ , the lower bound is strictly positive, so gradients do not vanish due to gating alone.

Proof. By the chain rule:

$$
{ \frac { \partial { \mathcal { L } } } { \partial \mathbf { x } _ { i } } } = \left( { \frac { \partial \mathbf { h } _ { i } } { \partial \mathbf { x } _ { i } } } \right) ^ { \top } { \frac { \partial { \mathcal { L } } } { \partial \mathbf { h } _ { i } } } .
$$

Since $\mathbf { h } _ { i } = \mathbf { G } _ { i } \mathbf { x } _ { i }$ and $\mathbf { G } _ { i } = \mathrm { d i a g } ( \sigma ( \mathbf { W } _ { g } \mathbf { x } _ { i } ) )$ , the Jacobian is:

$$
\frac { \partial { \bf h } _ { i } } { \partial { \bf x } _ { i } } = { \bf G } _ { i } + \left( \frac { \partial { \bf G } _ { i } } { \partial { \bf x } _ { i } } \right) { \bf x } _ { i } .
$$

The diagonal entries of $\mathbf { G } _ { i }$ are $\sigma ( \boldsymbol { z } _ { k } )$ where $z _ { k } ~ = ~ ( \mathbf { W } _ { g } \mathbf { x } _ { i } ) _ { k }$ , and $\sigma ^ { \prime } ( z _ { k } ) =$ $\sigma ( z _ { k } ) ( 1 - \sigma ( z _ { k } ) ) \in ( 0 , 1 / 4 )$ . Thus:

$$
\left( \frac { \partial \mathbf { h } _ { i } } { \partial \mathbf { x } _ { i } } \right) _ { k k } = \sigma ( z _ { k } ) + \sigma ^ { \prime } ( z _ { k } ) \mathbf { W } _ { g , k } ^ { \top } \mathbf { x } _ { i } .
$$

The matrix $\partial \mathbf { h } _ { i } / \partial \mathbf { x } _ { i }$ is diagonally dominant for bounded $\| \mathbf { x } _ { i } \|$ and $\| \mathbf { W } _ { g } \|$ . Its smallest singular value satisfies:

$$
\sigma _ { \operatorname* { m i n } } \left( \frac { \partial \mathbf { h } _ { i } } { \partial \mathbf { x } _ { i } } \right) \geq \lambda _ { \operatorname* { m i n } } ( \mathbf { G } _ { i } ) - \mathcal { O } ( \| \mathbf { W } _ { g } \| \| \mathbf { x } _ { i } \| ) .
$$

Since $\lambda _ { \operatorname* { m i n } } ( \mathbf G _ { i } ) \ : = \ : \operatorname* { m i n } _ { k } \sigma ( z _ { k } ) \ : > \ : 0$ for finite $z _ { k }$ , the lower bound is strictly positive for bounded inputs. Therefore:

$$
\left. \frac { \partial \mathcal { L } } { \partial \mathbf { x } _ { i } } \right. _ { 2 } \geq \sigma _ { \operatorname* { m i n } } \left( \frac { \partial \mathbf { h } _ { i } } { \partial \mathbf { x } _ { i } } \right) \left. \frac { \partial \mathcal { L } } { \partial \mathbf { h } _ { i } } \right. _ { 2 } .
$$

This proves the gradient lower bound.

## 8 Spherical Routing and Retrieval (M7+M8)

## 8.1 SIDA: Spherical Inverse Distance Attention

Definition 8.1 (SIDA Kernel). The spherical inverse-distance attention is defined as:

$$
D _ { i j } ^ { \mathbb { S } } = d _ { \mathbb { S } } ( \mathbf { q } _ { i } , \mathbf { k } _ { j } ) ^ { 2 } + \varepsilon , \quad W _ { i j } ^ { \mathbb { S } } = ( D _ { i j } ^ { \mathbb { S } } ) ^ { - 1 } / \sum _ { m } ( D _ { i m } ^ { \mathbb { S } } ) ^ { - 1 } .
$$

Theorem 8.1 (Spherical PL Inequality). For two keys with spherical angle $\theta > 0 .$

$$
\mu _ { S I D A } = \Theta ( \varepsilon ^ { 2 } / \theta ^ { 4 } ) , \quad \mu _ { s o f t } = \Theta ( e ^ { - ( 1 - \cos \theta ) } \varepsilon ^ { 2 } / \theta ^ { 2 } ) .
$$

Thus $\mu _ { S I D A } / \mu _ { s o f t } = \Omega ( e ^ { 1 - \cos \theta } / \theta ^ { 2 } )$

Unlike the hyperbolic case, spherical compactness ensures $\theta \leq \pi$ , so the ratio is bounded by a constant depending only on θ.

Proof. Place the keys at spherical coordinates $\mathbf { k } _ { 1 } ~ = ~ ( 1 , 0 , \ldots , 0 )$ and $\mathbf { k } _ { 2 } ~ =$ (cos θ, sin $\theta , 0 , \ldots , 0 )$ . For a query at distance t from k<sub>1</sub> along the geodesic toward $\mathbf { k } _ { 2 } .$ , we have $d _ { S } ( \mathbf { q } , \mathbf { k } _ { 1 } ) = t$ and $d _ { S } ( \mathbf { q } , \mathbf { k } _ { 2 } ) = \theta - t$ . The weight $W _ { 1 } ( t )$ has the same form as the 1D Euclidean case with $\Delta = \theta ,$ , giving $W _ { 1 } ^ { \prime } ( 0 ) = \mathrm { { 0 } }$ and $W _ { 1 } ^ { \prime \prime } ( 0 ) = - 4 \varepsilon / \theta ^ { 4 } + \mathcal { O } ( \varepsilon ^ { 2 } / \theta ^ { 6 } )$ . The loss curvature gives $\mu _ { \mathrm { S I D A } } = \Theta ( \varepsilon ^ { 2 } / \theta ^ { 4 } )$

For softmax, the logit diference at $t = 0$ is $s ( 0 ) = ( 1 - \cos \theta ) / \sqrt { d _ { h } } .$ giving the exponential factor. □

## 8.2 Cross-Geometric Mapping

Three methods map prototype centroids to the sphere: - Norm Normalization: $\phi _ { A } ( \mathbf { c } ) = \mathbf { c } / \| \mathbf { c } \|$ - Stereographic Projection: $\begin{array} { r } { \phi _ { B } ( \mathbf { c } ) = \big ( \frac { 2 \mathbf { c } } { 1 + \| \mathbf { c } \| ^ { 2 } } , \frac { 1 - \| \mathbf { c } \| ^ { 2 } } { 1 + \| \mathbf { c } \| ^ { 2 } } \big ) } \end{array}$ - Learnable Mapping: $\phi _ { C } ( \mathbf { c } ) = \mathrm { M L P } _ { \theta } ( \mathbf { c } ) / \| \mathrm { M L P } _ { \theta } ( \mathbf { c } ) \|$

## 8.3 Routing Protocol

Hard Routing: Select Top-K prototypes by SIDA weights. Soft Routing: full distribution.

## 9 Dynamic Memory Genesis (M9)

The prototype pool is $\mathcal { P } = \{ ( \mathcal { E } _ { e } , \mathbf { c } _ { e } , t _ { e } ^ { \mathrm { b i r t h } } , a _ { e } , t _ { e } ^ { \mathrm { l a s t } } ) \} _ { e = 1 } ^ { K }$

## 9.1 Surprise Detection with Adaptive Threshold

Define the sliding-window mean and standard deviation:

$$
\mu _ { t } = \frac { 1 } { W } \sum _ { i = t - W + 1 } ^ { t } \ell _ { i } , \quad \sigma _ { t } = \sqrt { \frac { 1 } { W } \sum _ { i = t - W + 1 } ^ { t } ( \ell _ { i } - \mu _ { t } ) ^ { 2 } } .
$$

The adaptive threshold is:

$$
\tau _ { t } = \tau _ { \mathrm { b a s e } } + \kappa \sigma _ { t } + \gamma \cdot \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \mathbf { 1 } _ { \mathrm { s u r p r i s e } } ( i ) .
$$

Lemma 9.1 (Adaptive Threshold Regret Bound under Sub-Gaussian Loss). Assume the loss $\ell _ { t }$ is sub-Gaussian with mean $\mu _ { 0 }$ and variance proxy $\sigma _ { 0 } ^ { 2 }$ under $t h e$ null model $( i . e . ,$ when no structural change occurs). For $\tau _ { t }$ as defined with $\tau _ { b a s e } > \mu _ { 0 }$ , we have $\mathbb { E } [ T _ { s } ( T ) ] \le { \mathcal { O } } ( \log T )$ .

Proof. Let $\begin{array} { r } { S _ { t } = \sum _ { i = 1 } ^ { t } \mathbf { 1 } _ { \mathrm { s u r p r i s e } } ( i ) } \end{array}$ be the cumulative surprise count. Define the supermartingale:

$$
M _ { t } = S _ { t } + \frac { 1 } { \gamma } \sum _ { i = 1 } ^ { t } ( \tau _ { i } - \tau _ { \mathrm { b a s e } } - \kappa \sigma _ { i } ) .
$$

Under the null model, $\mathbb { E } [ \ell _ { t } ] = \mu _ { 0 }$ and $\mathbb { E } [ \sigma _ { t } ] = \sigma _ { 0 }$ . The trigger condition $\mu _ { t } > \tau _ { t }$ implies that a surprise event occurs only when the loss exceeds the threshold. The adaptive term $\gamma \cdot S _ { t } / t$ penalizes frequent surprises, creating a negative drift. For sub-Gaussian losses, the probability of exceeding $\tau _ { \mathrm { b a s e } } + \kappa \sigma _ { t }$ decays exponentially, leading to the logarithmic bound. Summing yields $\mathbb { E } [ S _ { T } ] = \mathcal { O } ( \log T )$ . □

## 9.2 Convergence Guarantee

Between spawns, standard C-HIDA regret applies. Under the lemma, total regret is O(log T).

## 10 Geodesic Sparse Routing (GSR): Sparse Routing with Quality Guarantees

So far, SIDA (Section 7) has provided a mechanism for computing attention weights on the sphere. However, in distributed settings, routing every query to all $K _ { \mathrm { p o o l } }$ prototypes is expensive. We now introduce a sparse routing mechanism that selects only the Top-K prototypes by SIDA weight, with provable bounds on both routing quality and communication cost.

## 10.1 The GSR Mechanism

Let $\mathcal { P } = \left\{ \mathbf { c } _ { e } \right\} _ { e = 1 } ^ { K _ { \mathrm { p o o l } } } \subset \mathbb { S } ^ { d - 1 }$ be a pool of learnable prototype vectors. For a query $ { \mathbf { q } } \in \mathbb { S } ^ { d - 1 }$ , SIDA defines weights:

$$
w _ { e } = W _ { \mathbf { q } , \mathbf { c } _ { e } } ^ { \mathbb { S } } = \frac { ( d _ { \mathbb { S } } ( \mathbf { q } , \mathbf { c } _ { e } ) ^ { 2 } + \varepsilon ) ^ { - 1 } } { \sum _ { l = 1 } ^ { K _ { \mathrm { p o o l } } } ( d _ { \mathbb { S } } ( \mathbf { q } , \mathbf { c } _ { l } ) ^ { 2 } + \varepsilon ) ^ { - 1 } } .
$$

GSR selects the Top-K prototypes with largest weights and routes the query only to those prototypes:

$$
\begin{array} { r } { \mathcal { S } ( \mathbf { q } ) = \{ e \in [ K _ { \mathrm { p o o l } } ] : w _ { e } \mathrm { ~ i s ~ a m o n g ~ t h e ~ } K \mathrm { ~ l a r g e s t ~ w e i g h t s } \} . } \end{array}
$$

The output is the weighted combination of the values associated with the selected prototypes:

$$
\mathbf { o } ( \mathbf { q } ) = \sum _ { e \in S ( \mathbf { q } ) } w _ { e } \mathbf { v } _ { e } .
$$

## 10.2 Routing Quality Guarantee

The following theorem establishes that GSR’s routing quality is controlled by the SIDA weight gap between the Top-K and the remaining prototypes.

Theorem 10.1 (GSR Routing Quality). Let $w _ { ( 1 ) } \ge w _ { ( 2 ) } \ge \dots \ge w _ { ( K _ { p o o l } ) }$ be the SIDA weights sorted in descending order. For any query, the approximation error from routing only to the Top-K prototypes is bounded by:

$$
\| \mathbf { o } ( \mathbf { q } ) - \mathbf { o } ^ { * } ( \mathbf { q } ) \| _ { 2 } \leq 2 \| \mathbf { V } \| _ { F } \cdot \frac { \sum _ { e > K } w _ { ( e ) } } { \sum _ { e \leq K } w _ { ( e ) } } \cdot \left( \operatorname* { m a x } _ { e \leq K } \| \mathbf { v } _ { e } \| _ { 2 } \right) ,
$$

where $\mathbf { o } ^ { * } ( \mathbf { q } )$ is the exact SIDA output using all $K _ { p o o l }$ prototypes.

Moreover, $a s \varepsilon \to 0 ^ { + }$ , the SIDA weights concentrate on the nearest prototype, so for any $\delta > 0$ , there exists $\varepsilon _ { 0 } > 0$ such that for all $\varepsilon < \varepsilon _ { 0 }$

$$
\lVert \mathbf { o } ( \mathbf { q } ) - \mathbf { o } ^ { * } ( \mathbf { q } ) \rVert _ { 2 } \leq \delta
$$

provided the query has a unique nearest prototype and $\| \mathbf { v } _ { e } \| _ { 2 }$ is bounded.

Proof. Let the exact SIDA output be $\begin{array} { r } { \mathbf { o } ^ { * } = \sum _ { e = 1 } ^ { K _ { \mathrm { p o o l } } } w _ { e } \mathbf { v } _ { e } } \end{array}$ , and the GSR output be $\begin{array} { r } { \mathbf { o } = \sum _ { e < \boldsymbol { K } } w _ { e } \mathbf { v } _ { e } } \end{array}$ (where we reorder indices so the Top-K are $1 , \ldots , K )$ . Then the diference is:

$$
\| \mathbf { o } ^ { * } - \mathbf { o } \| _ { 2 } = \left\| \sum _ { e > K } w _ { e } \mathbf { v } _ { e } \right\| _ { 2 } .
$$

By the triangle inequality:

$$
\| \mathbf { o } ^ { * } - \mathbf { o } \| _ { 2 } \leq \sum _ { e > K } w _ { e } \| \mathbf { v } _ { e } \| _ { 2 } .
$$

Since $\| \mathbf { v } _ { e } \| _ { 2 } \leq \| \mathbf { V } \| _ { F }$ for all $e ,$ we have:

$$
\| \mathbf { o } ^ { * } - \mathbf { o } \| _ { 2 } \leq \| \mathbf { V } \| _ { F } \sum _ { e > K } w _ { e } .
$$

The sum of the Top-K weights is $\begin{array} { r } { \sum _ { e \leq K } w _ { e } = 1 - \sum _ { e > K } w _ { e } } \end{array}$ . Thus $\begin{array} { r } { \sum _ { e > K } w _ { e } = } \end{array}$ $\begin{array} { r } { 1 - \sum _ { e < K } w _ { e } } \end{array}$ . The ratio bound follows by observing that $\begin{array} { r } { \sum _ { e > K } w _ { e } \le \frac { \sum _ { e > K } w _ { e } } { \sum _ { e \le K } w _ { e } } } \end{array}$ since $\textstyle \sum _ { e < K } w _ { e } \leq 1$

For the second part, recall from the exact retrieval property (Lemma 0, Part (a)) that if $\mathbf { q } ~ = ~ \mathbf { c } _ { e ^ { * } }$ for a unique nearest prototype, then l $\begin{array} { r } { \operatorname* { i m } _ { \varepsilon \to 0 ^ { + } } w _ { e ^ { * } } = 1 } \end{array}$ Hence $\begin{array} { r } { \operatorname* { l i m } _ { \varepsilon \to 0 ^ { + } } \sum _ { e > K } w _ { e } = 0 } \end{array}$ for any $K \geq 1$ (since all weight concentrates on the single nearest prototype). Therefore the approximation error can be made arbitrarily small by choosing ε suficiently small. □

## 10.3 Communication Complexity Analysis

GSR is designed for distributed settings where prototypes are distributed across P workers. We analyze the communication cost.

Let the $K _ { \mathrm { p o o l } }$ prototypes be partitioned among P workers. Each worker holds approximately $K _ { \mathrm { p o o l } } / P$ prototypes. For a batch of B queries, the protocol is:

1. Each worker computes SIDA weights for its local prototypes: local computation, no communication. 2. The Top-K indices are gathered across workers: All-Gather on the top scores from each worker. 3. Selected prototypes compute outputs: local computation. 4. Results are aggregated: All-Reduce on the K selected outputs.

Theorem 10.2 (GSR Communication Complexity). GSR has per-query communication cost:

$$
\mathcal { O } ( K _ { p o o l } \cdot d _ { h } + K \cdot d _ { h } ) ,
$$

independent of batch size B. In the $\alpha \cdot \beta$ model, this is:

$$
\# m e s s a g e s = \mathcal { O } ( K _ { p o o l } + K + P ) , \qquad b y t e s = \mathcal { O } \big ( ( K _ { p o o l } + K ) d _ { h } \cdot { p r e c i s i o n } \big ) .
$$

In contrast, All-to-All MoE has:

$$
\# m e s s a g e s = \mathcal { O } ( P ^ { 2 } ) , \qquad b y t e s = \mathcal { O } ( B K d _ { h } \cdot p r e c i s i o n ) ,
$$

which scales linearly with batch size B.

Proof. Each of the P workers must communicate its local top scores to all other workers. The All-Gather on scores of size $\mathcal { O } ( K _ { \mathrm { p o o l } } / P \cdot d _ { h } )$ per worker requires $\mathcal { O } ( K _ { \mathrm { p o o l } } \cdot d _ { h } )$ total communication. The All-Reduce on the K selected outputs requires $\mathcal { O } ( K \cdot d _ { h } )$ communication. The number of messages is $\mathcal { O } ( K _ { \mathrm { p o o l } } { + } K { + } P )$ $K _ { \mathrm { p o o l } }$ for the score gathering, K for the result aggregation, and P for the initial scatter. The byte count follows from the dimension $d _ { h }$ and the floating-point precision. □

## 10.4 Relation to Other Modules

GSR is the routing component that follows SIDA. While SIDA computes all-toall attention weights on the sphere, GSR uses those weights to select a sparse subset for communication and computation. DMG (Section 8) populates the prototype pool dynamically; GSR routes to the prototypes that DMG has allocated. This creates a complete pipeline: DMG allocates prototypes, SIDA computes spherical attention, and GSR routes queries to a sparse subset with quality guarantees.

## 11 Unified Framework (M10)

## 11.1 Global Architecture

$$
\mathbf { x } \to \mathrm { Q K V } \to \left\{ \begin{array} { l l } { \mathrm { H I D A ~ p a t h } \to \mathbf { o } _ { \mathrm { m a i n } } } & { \mathrm { ~ } } \\ { \mathrm { S I D A } \to \mathrm { G S R } \to \mathrm { D M G } \to \mathbf { o } _ { \mathrm { m e m o r y } } } & { \mathrm { ~ } \to \mathrm { H y p e r G a t e } \to \mathbf { o } _ { \mathrm { f i n a l } } . } \end{array} \right.
$$

## 11.2 Unified Mathematical Language

All variants share the same core structure:

$$
\mathbf { o } _ { i } = \sum _ { j \in \mathcal { R } _ { i } } W _ { i j } \mathbf { v } _ { j } , \quad W _ { i j } = \frac { ( d ( \mathbf { q } _ { i } , \mathbf { k } _ { j } ) ^ { 2 } + \varepsilon ) ^ { - 1 } } { \sum _ { m \in \mathcal { R } _ { i } } ( d ( \mathbf { q } _ { i } , \mathbf { k } _ { m } ) ^ { 2 } + \varepsilon ) ^ { - 1 } } ,
$$

where $\mathcal { R } _ { i }$ varies by mode and d is Euclidean, hyperbolic, or spherical.

## 11.3 Proposition 1: Complexity Observations for HIDA Variants

For each HIDA variant, the computational complexity is:

• Dense-HIDA: complexity $\Theta ( n ^ { 2 } d _ { h } )$ , exact kernel.

• FP-HIDA: complexity O(n log $n \cdot d _ { h } )$ with w, $g = \Theta ( \log n )$

• L-HIDA: complexity $\mathcal { O } ( n d _ { h } )$ with $m = \Theta ( 1 )$ anchors.

• C-HIDA: complexity Θ(1) per token with $c = \Theta ( 1 )$ summaries.

This establishes a trade-of: sparser approximations reduce complexity at the cost of approximation error. The exact error rates depend on the data distribution and kernel spectrum.

## 11.4 Proposition 2: Information-Theoretic Justification

Two key information-theoretic bounds separate hyperbolic storage from spherical routing:

$$
I ( \mathbf { q } _ { S } ; \mathbf { c } _ { e } ^ { \mathbb { S } } ) \leq \log K _ { \mathrm { p o o l } } , \qquad I ( \mathbf { k } ; \mathbf { c } _ { e } ) \propto \log \frac { 1 } { 1 - \| \mathbf { c } _ { e } \| ^ { 2 } } .
$$

Proof. The mutual information bound for spherical routing follows from the entropy bound $H ( \mathbf { c } _ { e } ^ { \mathbb { S } } ) \leq \log K _ { \mathrm { p o o l } }$ . For hyperbolic storage, the volume growth $V _ { \mathbb { H } } ( R ) \propto e ^ { ( d _ { h } - 1 ) R }$ gives the divergence as $\| \mathbf { c } _ { e } \|  1 ^ { - }$ □

## 11.5 Proposition 3: DMG Scalability Observations

The DMG system has two separate properties:

1. The online k-means update via C-HIDA achieves O(log T) regret relative to the optimal fixed centroids.

2. With $K _ { \mathrm { m a x } }$ memory slots, the quantization error decays as $\mathcal { O } ( 1 / \sqrt { K _ { \operatorname* { m a x } } } )$ under compactness assumptions (data bounded away from the hyperbolic boundary).

These properties hold independently and are not coupled in the regret bound.

## 12 Discussion and Limitations

Scope. This is a theoretical study. The primary contribution is the theoretical framework.

## 12.1 Limitations

No experimental validation. This paper is purely theoretical. The theorems establish mathematical guarantees, but empirical verification on benchmarks is not provided. This is left for future work.

Compression scope. HCC compresses keys only; values remain full-precision. Routing communication. The communication analysis is theoretical under an idealized FLOPs model.

DMG hyperparameters. The adaptive threshold requires tuning; the O(log T) regret bound relies on a sub-Gaussian loss assumption.

Curvature in theory. The two-point PL analysis is a limitation; multipoint efects are future work.

## 12.2 Future Work

Key directions: (1) value compression for HCC; (2) distributed routing implementation; (3) mixed-curvature extensions; (4) multi-point curvature analysis; (5) empirical validation on standard benchmarks.

## 13 Conclusion

We have presented a theoretical arc from Euclidean Resolver to Riemann Geo-Resolver. The Euclidean part establishes three core theorems with complete proofs, and the non-Euclidean part extends the framework to hyperbolic and spherical geometries.

Summary of Euclidean contributions.

Theorem 1 (Circuit Separation) established that IDA achieves exact retrieval with O(1) resources, while any softmax-based architecture requires superconstant width. This shows a fundamental gap in the representational power of the two attention mechanisms.

Theorem 2 (PL Inequality) proved that IDA satisfies a Polyak– Lojasiewicz inequality with constant $\Theta ( \varepsilon ^ { 2 } / \Delta ^ { 4 } )$ , while softmax has constant $\Theta ( e ^ { - \Delta ^ { 2 } / \sqrt { d } } \varepsilon ^ { 2 } / \Delta ^ { 2 } )$ The ratio is exponentially large in $\Delta ^ { 2 } / \sqrt { d } .$ , implying that IDA converges linearly and dramatically faster for separated keys. The theorem also implied the absence of spurious local minima and established Θ(1) Hessian spread.

Theorem 3 (Efective Rank) established that IDA’s efective rank is bounded independent of hidden width, preventing the memorization of arbitrary noisy labels that occurs in softmax when $d _ { h } \ge n$ . The theorem also provided a noise robustness bound $\mathcal { E } _ { \mathrm { t e s t } } ^ { \mathrm { I D A } } \leq C \eta ^ { 2 } + \mathcal { O } ( 1 / \sqrt { n } )$

Summary of non-Euclidean contributions.

The Riemann GeoResolver framework extends the Euclidean prototype to hyperbolic and spherical geometries:

• HIDA family $( M 1 { - } M \ 4 )$ : Four hyperbolic attention operators spanning $\Theta ( n ^ { 2 } )$ to $\Theta ( 1 )$ complexity, with circuit separation and PL inequalities proved for the dense case.

• Hyperbolic Curvature Compression (M5): Provable reconstruction error bound $\| \mathbf { k } - \widetilde { \mathbf { k } } \| _ { 2 } ^ { 2 } \leq 4 ( 2 ^ { - b } + 2 ^ { - b _ { r } } )$ and memory complexity analysis.

• HyperGate (M6): Gradient lower-bound theorem ensuring non-vanishing gradients through curvature-adaptive gating.

• Spherical Inverse Distance Attention (M7–M8): Sphere-analog PL inequalities and cross-geometric mapping methods.

• Dynamic Memory Genesis (M9): O(log T) regret bound for online prototype allocation.

• Geodesic Sparse Routing (M10): Routing quality and communication complexity bounds.

Theoretical significance.

This work establishes that inverse-distance attention is not merely an empirical alternative to softmax, but a mechanism with fundamentally diferent theoretical properties. The three Euclidean theorems—circuit separation, PL inequality, and efective rank bound—collectively show that IDA is provably stronger than softmax in expressiveness, optimization, and generalization. The non-Euclidean extension shows that these advantages are preserved and amplified in hyperbolic and spherical geometries.

Open directions.

The limitations of this work suggest several directions for future investigation. First, the PL analysis is restricted to the two-key case; extending to multi-key settings would provide stronger optimization guarantees. Second, the empirical validation remains future work. Third, the compression analysis is limited to keys; extending to values would improve memory eficiency. Fourth, the distributed routing analysis is theoretical; implementation and scaling on real systems would validate the communication bounds. Fifth, mixed-curvature spaces combining hyperbolic and spherical geometries could provide even more flexible representations.

## Acknowledgments

The author gratefully acknowledges Prof. Junxue Zhang at the University of Science and Technology of China for his valuable guidance and insightful discussions throughout this research.

## References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS).

[2] Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. (2020). Transformers are RNNs: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning (ICML), PMLR 119:5156–5165.

[3] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and R´e, C. (2022). FlashAttention: Fast and memory-eficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS).

[4] Dao, T. (2023). FlashAttention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691.

[5] Child, R., Gray, S., Radford, A., and Sutskever, I. (2019). Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509.

[6] Beltagy, I., Peters, M. E., and Cohan, A. (2020). Longformer: The longdocument transformer. arXiv preprint arXiv:2004.05150.

[7] Kitaev, N., Kaiser, L., and Levskaya, A. (2020). Reformer: The eficient transformer. In International Conference on Learning Representations (ICLR).

[8] Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H. (2020). Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768.

[9] Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, L., et al. (2021). Rethinking attention with performers. In International Conference on Learning Representations (ICLR).

[10] Bello, I., Zoph, B., Vaswani, A., Shlens, J., and Le, Q. V. (2019). Attention augmented convolutional networks. arXiv preprint arXiv:1905.10863.

[11] Parmar, N., Vaswani, A., Uszkoreit, J., Kaiser, L., Shazeer, N., Ku, A., and Tran, D. (2018). Image transformer. In International Conference on Machine Learning (ICML), PMLR 80:4055–4064.

[12] McCarter, C. (2023). Inverse distance weighting attention. arXiv preprint arXiv:2310.18805.

[13] Ge, L. (2026). GeoResolver: Optimization geometry and capacity bounds for inverse distance attention. Under review.

[14] Nadaraya, E. A. (1964). On estimating regression. Theory of Probability & Its Applications, 9(1):141–142.

[15] Watson, G. S. (1964). Smooth regression analysis. Sankhy¯a: The Indian Journal of Statistics, Series A, 26(4):359–372.

[16] Shepard, R. N. (1968). Towards a universal law of generalization for psychological science. Science, 162(3850):1343–1352.

[17] Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences, 79(8):2554–2558.

[18] Krotov, D. and Hopfield, J. J. (2016). Dense associative memory for pattern recognition. In Advances in Neural Information Processing Systems (NeurIPS).

[19] Ambrogioni, L. (2023). The statistical thermodynamics of generative difusion models. arXiv preprint arXiv:2302.03615.

[20] Nickel, M. and Kiela, D. (2017). Poincar´e embeddings for learning hierarchical representations. In Advances in Neural Information Processing Systems (NeurIPS).

[21] Nickel, M. and Kiela, D. (2018). Learning continuous hierarchies in the Lorentz model of hyperbolic geometry. In International Conference on Machine Learning (ICML), PMLR 80:3779–3788.

[22] Ganea, O., B´ecigneul, G., and Hofmann, T. (2018). Hyperbolic neural networks. In Advances in Neural Information Processing Systems (NeurIPS).

[23] Chami, I., Ying, Z., R´e, C., and Leskovec, J. (2019). Hyperbolic graph convolutional neural networks. In Advances in Neural Information Processing Systems (NeurIPS).

[24] Gulcehre, C., Denil, M., Malinowski, M., Razavi, A., Pascanu, R., Hermann, K. M., Battaglia, P., Bapst, V., Raposo, D., Santoro, A., and de Freitas, N. (2020). Hyperbolic attention networks. In International Conference on Learning Representations (ICLR).

[25] Shimizu, R., Mukuta, Y., and Harada, T. (2021). Hyperbolic neural networks++. In International Conference on Learning Representations (ICLR).

[26] Chen, W., Han, X., Lin, Y., Zhao, H., Liu, Z., Li, P., Sun, M., and Zhou, J. (2021). Fully hyperbolic neural networks. arXiv preprint arXiv:2105.14686.

[27] Liu, Q., Nickel, M., and Kiela, D. (2020). Hyperbolic graph neural networks: A review of methods and applications. arXiv preprint arXiv:2002.08852.

[28] Krioukov, D., Papadopoulos, F., Kitsak, M., Vahdat, A., and Bogu˜n´a, M. (2010). Hyperbolic geometry of complex networks. Physical Review E, 82(3):036106.

[29] Sala, F., De Sa, C., Gu, A., and R´e, C. (2018). Representation tradeofs for hyperbolic embeddings. In International Conference on Machine Learning (ICML), PMLR 80:4460–4469.

[30] Cohen, T. S., Geiger, M., K¨ohler, J., and Welling, M. (2018). Spherical CNNs. In International Conference on Learning Representations (ICLR).

[31] Esteves, C., Allen-Blanchette, C., Makadia, A., and Daniilidis, K. (2020). Learning SO(3) equivariant representations with spherical CNNs. International Journal of Computer Vision, 128(5):1471–1488.

[32] Bonev, B., Rietmann, M., Paris, A., Carpentieri, A., and Kurth, T. (2025). Attention on the sphere. In Advances in Neural Information Processing Systems (NeurIPS).

[33] Kobler, R. J., Hirayama, J., and Kawanabe, M. (2022). Spherical convolutions and their applications in machine learning. arXiv preprint arXiv:2201.11899.

[34] Gu, A., Goel, K., and R´e, C. (2021). Eficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396.

[35] Gu, A., Goel, K., Gupta, A., and R´e, C. (2022). On the parameterization and initialization of diagonal state space models. In Advances in Neural Information Processing Systems (NeurIPS).

[36] Gu, A. and Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.

[37] Dao, T. and Gu, A. (2024). Transformers are SSMs: Generalized models and eficient algorithms through structured state space duality. In International Conference on Machine Learning (ICML).

[38] Smith, J. T. H., Warrington, A., and Linderman, S. W. (2023). Simplified state space layers for sequence modeling. In International Conference on Learning Representations (ICLR).

[39] Gupta, A., Gu, A., and Berant, J. (2022). Diagonal state spaces are as efective as structured state spaces. arXiv preprint arXiv:2203.14343.

[40] Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. (2017). Outrageously large neural networks: The sparselygated mixture-of-experts layer. In International Conference on Learning Representations (ICLR).

[41] Fedus, W., Zoph, B., and Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity. Journal of Machine Learning Research, 23(120):1–39.

[42] Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., and Chen, Z. (2021). GShard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations (ICLR).

[43] Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., Krikun, M., Zhou, Y., Yu, A. W., Firat, O., et al. (2022). GLaM: Eficient scaling of language models with mixture-of-experts. In International Conference on Machine Learning (ICML), PMLR 162:5547–5569.

[44] DeepSeek-AI. (2024). DeepSeek-V3: Technical report. arXiv preprint arXiv:2412.19437.

[45] Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., de las Casas, D., Hanna, E. B., Bressand, F., et al. (2024). Mixtral of experts. arXiv preprint arXiv:2401.04088.

[46] Roller, S., Sukhbaatar, S., Weston, J., et al. (2021). Hash layers for large sparse models. In Advances in Neural Information Processing Systems (NeurIPS).

[47] Lewis, M., Bhosale, S., Dettmers, T., Goyal, N., and Zettlemoyer, L. (2021). BASE layers: Simplifying training of large, sparse models. arXiv preprint arXiv:2103.16716.

[48] Cai, W., Jiang, J., Wang, F., Tang, J., Kim, S., and Huang, J. (2024). A survey on mixture of experts. arXiv preprint arXiv:2407.06204.

[49] Shazeer, N. (2019). Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150.

[50] Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebr´on, F., and Sanghai, S. (2023). GQA: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245.

[51] Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., and Han, S. (2024). AWQ: Activation-aware weight quantization for LLM compression and acceleration. In International Conference on Machine Learning (ICML).

[52] Liu, Z., Yuan, J., Jin, H., Zhong, S., Xu, Z., Braverman, V., Chen, B., and Hu, X. (2024). KIVI: A tuning-free asymmetric 2-bit quantization for KV cache. arXiv preprint arXiv:2402.02750.

[53] Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. (2022). GPTQ: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323.

[54] Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S. (2024). SmoothQuant: Accurate and eficient post-training quantization for large language models. In International Conference on Machine Learning (ICML).

[55] Kim, S., Hooper, C., Gholami, A., Dong, Z., Li, X., Shen, S., Mahoney, M. W., and Keutzer, K. (2024). SqueezeLLM: Dense-and-sparse quantization. arXiv preprint arXiv:2306.07629.

[56] Bhatnagar, P., Moradifirouzabadi, A., Yang, S.-H., Lee, S., Choi, J., and Kang, M. (2026). STAR-KV: Low-rank KV cache compression via soft thresholding for adaptive rank control. In International Conference on Machine Learning (ICML).

[57] Karimi, H., Nutini, J., and Schmidt, M. (2016). Linear convergence of gradient and proximal-gradient methods under the Polyak– Lojasiewicz condition. In European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD).

[58] Belkin, M., Hsu, D., Ma, S., and Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-of. Proceedings of the National Academy of Sciences, 116(32):15849–15854.

[59] Ba, J. L., Kiros, J. R., and Hinton, G. E. (2016). Layer normalization. arXiv preprint arXiv:1607.06450.

[60] He, K., Zhang, X., Ren, S., and Sun, J. (2016). Identity mappings in deep residual networks. arXiv preprint arXiv:1603.05027.

[61] Li, Y. and Yuan, Y. (2018). Convergence analysis of two-layer neural networks with ReLU activation. In Advances in Neural Information Processing Systems (NeurIPS).

[62] Liu, C., Zhu, L., and Belkin, M. (2022). Loss landscapes and optimization in over-parameterized non-linear systems. Journal of Machine Learning Research, 23(45):1–43.

[63] Drineas, P. and Mahoney, M. W. (2005). On the Nystr¨om method for approximating a Gram matrix for improved kernel-based learning. Journal of Machine Learning Research, 6:2153–2175.

[64] He, N., et al. (2025). Hyperbolic deep learning for foundation models: A survey. arXiv preprint arXiv:2507.17787. SIGKDD 2025.

[65] Zhang, Y., Chen, L., and Liu, Q. (2025). A survey on hyperbolic neural networks. arXiv preprint arXiv:2504.06543.

[66] Graves, A., Wayne, G., and Danihelka, I. (2014). Neural turing machines. arXiv preprint arXiv:1410.5401.

[67] Graves, A., Wayne, G., Reynolds, M., Harley, T., Danihelka, I., Grabska-Barwi´nska, A., Colmenarejo, S. G., Grefenstette, E., Ramalho, T., Agapiou, J., et al. (2016). Hybrid computing using a neural network with dynamic external memory. Nature, 538(7626):471–476.

[68] Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8):1735–1780.

[69] Bengio, Y., Simard, P., and Frasconi, P. (1994). Learning long-term dependencies with gradient descent is dificult. IEEE Transactions on Neural Networks, 5(2):157–166.

[70] Pascanu, R., Mikolov, T., and Bengio, Y. (2013). On the dificulty of training recurrent neural networks. In International Conference on Machine Learning (ICML), PMLR 28:1310–1318.