# Simplex Relaxation for Discrete Diffusion

Jinya Sakurai<sup>1</sup> <sup>2</sup> <sup>4∗</sup> Patrick Pynadath<sup>3</sup> Satoshi Hayakawa<sup>2</sup> Jaehong Yoon<sup>1</sup> Xulei Yang<sup>4</sup> Nancy F. Chen<sup>4,5</sup> Xun Xu<sup>4,5</sup>

<sup>1</sup>NTU Singapore <sup>2</sup>The University of Tokyo <sup>3</sup>Purdue University <sup>4</sup>Institute for Advanced Intelligence and Computing (IAIC), A\*STAR <sup>5</sup>Centre for Frontier AI Research (CFAR), A\*STAR

## Abstract

Discrete diffusion models for categorical generation are defined by a corruption kernel, which determines the intermediate state space and the associated reverse prediction problem. We study uniform discrete diffusion and ask whether its training objective and reverse transitions can be enriched without changing the underlying categorical corruption process. We introduce Simplax, an exact Dirichlet– categorical augmentation that couples each corrupted categorical state with an auxiliary simplex-valued variable while preserving the original uniform diffusion process as its categorical marginal. This augmentation yields a tractable Rao–Blackwellized reverse-bridge objective and a corresponding stochastic reverse sampler, while retaining the corrupted categorical state as the denoiser input. Empirically, Simplax improves the generative perplexity–entropy tradeoff on unconditional OpenWebText generation. On Sudoku, a model trained exclusively on 30-clue puzzles achieves the highest accuracy among the compared methods across all evaluated clue densities, including the minimum uniquely solvable 17-clue regime, and also achieves the highest validity in unconditional generation.

## 1 Introduction

Discrete diffusion models have emerged as a promising framework for generative modeling over categorical data, including text, biological sequences, and other symbolic domains [Hoogeboom et al., 2021, Austin et al., 2021, Campbell et al., 2022, Lou et al., 2024, Zhang et al., 2025]. Compared with autoregressive generation, they offer a conceptually different route to parallel prediction by learning to invert a noising process on full categorical states. A central design choice in this framework is the corruption kernel. This choice determines the intermediate state space and the semantics of reverse updates, and shapes the form of the training objective used to approximate the reverse process [Austin et al., 2021, Campbell et al., 2022, Lou et al., 2024].

Existing discrete diffusion models instantiate this design choice in different ways. Masked diffusion corrupts tokens toward a distinguished mask state, yielding intermediate sequences that can be interpreted as partially observed data [Austin et al., 2021, Sahoo et al., 2024, Shi et al., 2024, Ou et al., 2025, Zheng et al., 2025]. Uniform diffusion instead replaces tokens toward the uniform distribution over the original categorical alphabet, treating all categories symmetrically without introducing a distinguished absorbing state [Austin et al., 2021, Schiff et al., 2025, Sahoo et al., 2025, Deschenaux et al., 2026]. Recent work has studied uniform diffusion in connection with guidance, repeated token revision, few-step generation, self-correction, and scaling [Schiff et al., 2025, Sahoo et al., 2025, von Rütte et al., 2025, 2026, Sahoo et al., 2026].

![](images/bf853af892d29d38071e475afaffb1f56c2148897e421a2e9f657ccfa1e34dab.jpg)

(b) Simplax (ours): all K bridges are matched at once  
![](images/0f150cfa4c941ca1be6dea502f88c2dccd28edfde172fc67cb229e65d489ff94.jpg)  
Figure 1: Reverse-bridge matching, illustrated for $K = 3 \colon$ one-hot states are vertices of $\Delta ^ { K - 1 }$ and $\mathbf { w } _ { t }$ is an interior point. (a) The corrupted state $\mathbf { z } _ { t }$ is both the denoiser input and the sole bridge anchor, so a single bridge is matched per sample. (b) Simplax samples $\left( \mathbf { w } _ { t } , \mathbf { z } _ { t } \right)$ jointly; $\mathbf { z } _ { t }$ remains the denoiser input, while extra draws $\widetilde { \mathbf { z } } _ { t } \sim \mathrm { C a t } ( \mathbf { w } _ { t } )$ anchor all K bridges, which are matched at once and marginalized exactly to give (15).

These developments leave open a methodological question for uniform discrete diffusion: can one enrich its training objectives and samplers while keeping the categorical corruption process unchanged? In standard uniform diffusion, the reverse update between two noise levels is expressed directly through sampled categorical states. This preserves the discrete generative process, but it also means that both training and sampling are formulated through categorical intermediate states. We ask whether an auxiliary probabilistic structure can be introduced around these transitions so that tractable objectives and samplers can be derived without changing the forward process itself.

In this work, we consider a probabilistic augmentation of uniform discrete diffusion that leaves its categorical corruption process unchanged while introducing an auxiliary simplex-valued state. We introduce Simplax, an exact Dirichlet–categorical augmentation in which each corrupted categorical state $\mathbf { z } _ { t }$ is coupled to an auxiliary simplex variable $\mathbf { w } _ { t }$ through a shifted Dirichlet conditional. The resulting augmented hierarchy preserves the original uniform diffusion process as its categorical marginal and admits the exact decoder

$$
q ( \mathbf { z } _ { t } \mid \mathbf { w } _ { t } ) = \operatorname { C a t } ( \mathbf { z } _ { t } ; \mathbf { w } _ { t } ) .
$$

Thus, the simplex variable is not introduced as a replacement for the categorical state, but as an auxiliary random variable that is probabilistically coupled to it and can be used to construct the training objective and reverse transition.

A direct construction of the reverse objective in the augmented space is complicated by the fact that the induced simplex reverse bridges are mixtures of shifted Dirichlet components, whose KL divergence is generally intractable. We therefore derive a categorical reverse-bridge surrogate by averaging the standard discrete reverse KL divergence over an auxiliary categorical decode from $\mathbf { w } _ { t }$ This expectation admits an analytic Rao–Blackwellized form. We further derive a stochastic ancestral sampler from the same augmented hierarchy, using the denoiser prediction together with the auxiliary simplex state to parameterize the reverse update.

We evaluate Simplax on unconditional text generation with OpenWebText and constrained categorical generation with Sudoku. On OpenWebText, Simplax achieves favorable generative perplexity–entropy tradeoffs across a wide range of inference budgets, outperforming the compared methods at most reported operating points. On Sudoku, all models are trained exclusively on puzzles with 30 clues and evaluated in-distribution, under transfer to both easier and harder clue densities, and in unconditional generation. Simplax achieves the highest performance among the compared methods across all evaluated Sudoku settings.

Our main contributions are as follows. First, we introduce an exact Dirichlet–categorical augmentation of uniform discrete diffusion that preserves the original categorical process as a marginal. Second, we derive a tractable categorical reverse-bridge surrogate whose auxiliary categorical expectation admits a Rao–Blackwellized closed form, together with a stochastic ancestral sampler derived from the same augmented hierarchy. Third, we empirically demonstrate that the resulting framework improves generation across both open-ended text modeling and constrained categorical generation, including broad inference-budget regimes and distribution shifts in Sudoku.

## 2 Preliminaries

Notation. Let $\begin{array} { r } { \mathcal { V } = \{ \mathbf { x } \in \{ 0 , 1 \} ^ { K } : \sum _ { i = 1 } ^ { K } x _ { i } = 1 \} } \end{array}$ denote the set of one-hot vectors over K categories, and let $\Delta ^ { K - 1 }$ denote the probability simplex over K categories. We represent scalar discrete random variables taking K values as one-hot column vectors in V. We write $\operatorname { C a t } ( \cdot ; \pi )$ for the categorical distribution with class probabilities $\pi \in \Delta ^ { K - 1 }$ . Results involving Dirichlet densities assume $\pi _ { k } > 0$ for every category k; the uniform base distribution used in our experiments satisfies this condition. We write $\operatorname { D i r } ( \cdot ; \pmb { \alpha } )$ for the Dirichlet distribution with concentration vector $\pmb { \alpha } \in \mathbb { R } _ { > 0 } ^ { K } .$ We use $\mathbf { 1 } \in \mathbb { R } ^ { K }$ for the all-ones vector, $\langle \mathbf { a } , \mathbf { b } \rangle$ for the inner product, $\mathbf { a } \odot$ b for the Hadamard product, and a ⊘ b for elementwise division. For sequences of length L, we write $\mathbf { x } ^ { 1 : L } \in \mathcal { V } ^ { L }$

## 2.1 Dirichlet distribution

The Dirichlet distribution is a distribution over the probability simplex. Its mean is given by the normalized concentration vector, while the sum of the concentration parameters controls how concentrated the distribution is around its mean. In particular, for $\mathbf { p } \in \Delta ^ { K - 1 }$ and $\eta > 0 , \operatorname { D i r } ( \cdot ; \eta \mathbf { p } )$ denotes the Dirichlet distribution centered at p, with η controlling its concentration.

## 2.2 Discrete diffusion

We consider a discrete diffusion process with prior $\pi \in \Delta ^ { K - 1 }$ and noise schedule $\alpha _ { t } \in [ 0 , 1 ]$ Following the standard parameterization, the noisy categorical state at time t is distributed as

$$
q ( \mathbf { z } _ { t } \mid \mathbf { x } ) = \operatorname { C a t } ( \mathbf { z } _ { t } ; \mathbf { p } _ { t } ( \mathbf { x } ) ) , \quad \mathbf { p } _ { t } ( \mathbf { x } ) : = \alpha _ { t } \mathbf { x } + ( 1 - \alpha _ { t } ) \pi .\tag{1}
$$

We use a schedule with $\alpha _ { 0 } = 1 , \alpha _ { 1 } = 0$ , and $\alpha _ { t } < 1$ for every $t > 0 .$ . Hence $\mathbf { p } _ { t } ( \mathbf { x } )$ is strictly positive for $t > 0$

For two times $s < t ,$ the forward transition can be written as

$$
q ( \mathbf { z } _ { t } \mid \mathbf { z } _ { s } ) = \operatorname { C a t } \left( \mathbf { z } _ { t } ; \alpha _ { t \mid s } \mathbf { z } _ { s } + ( 1 - \alpha _ { t \mid s } ) \boldsymbol { \pi } \right) , \quad \alpha _ { t \mid s } : = \frac { \alpha _ { t } } { \alpha _ { s } } .\tag{2}
$$

The corresponding reverse posterior has the usual closed form

$$
q ( \mathbf { z } _ { s } \mid \mathbf { z } _ { t } , \mathbf { x } ) = \operatorname { C a t } \left( \mathbf { z } _ { s } ; \mathbf { r } _ { s \mid t } ( \mathbf { x } , \mathbf { z } _ { t } ) \right) ,\tag{3}
$$

where

$$
\mathbf { r } _ { s \mid t } ( \mathbf { x } , \mathbf { z } _ { t } ) : = \frac { \left[ \alpha _ { t \mid s } \mathbf { z } _ { t } + \left( 1 - \alpha _ { t \mid s } \right) \left. \mathbf { z } _ { t } , \pi \right. \mathbf { 1 } \right] \odot \mathbf { p } _ { s } ( \mathbf { x } ) } { \left. \mathbf { z } _ { t } , \mathbf { p } _ { t } ( \mathbf { x } ) \right. } .\tag{4}
$$

Standard discrete diffusion training minimizes the categorical reverse KL divergence

$$
\mathcal { L } _ { z _ { s } | z _ { t } } ( \mathbf { z } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) = D _ { \mathrm { K L } } \big [ q ( \mathbf { z } _ { s } \mid \mathbf { z } _ { t } , \mathbf { x } ) \big \| q ( \mathbf { z } _ { s } \mid \mathbf { z } _ { t } , \hat { \mathbf { x } } _ { \theta } ) \big ] .\tag{5}
$$

where $\hat { \mathbf { x } } _ { \theta } = f _ { \theta } ( \mathbf { z } _ { t } , t ) \in \Delta ^ { K - 1 }$ is the model prediction of the clean-token distribution. We use the shorthand $\mathbf { p } _ { t } : = \mathbf { p } _ { t } ( \mathbf { x } ) , \hat { \mathbf { p } } _ { t } : = \mathbf { p } _ { t } ( \hat { \mathbf { x } } _ { \theta } )$

## 3 Method

We construct Simplax by augmenting the uniform discrete diffusion process with an auxiliary simplex-valued variable. The construction leaves the categorical corruption process unchanged, but introduces an exact Dirichlet–categorical hierarchy around each corrupted state. We first define this hierarchy and derive its reverse bridge identities. We then use these identities to obtain a tractable Rao–Blackwellized training objective and a sampler induced by the same bridge structure.

## 3.1 Simplex relaxation

Building upon (2), we consider the following joint factorization for two times $s < t { : }$

$$
q ( \mathbf { x } , \mathbf { z } _ { s } , \mathbf { z } _ { t } , \mathbf { w } _ { s } , \mathbf { w } _ { t } ) = q ( \mathbf { x } ) q ( \mathbf { z } _ { s } \mid \mathbf { x } ) q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \mathbf { x } ) q ( \mathbf { z } _ { t } \mid \mathbf { z } _ { s } ) q ( \mathbf { w } _ { t } \mid \mathbf { z } _ { t } , \mathbf { x } ) .\tag{6}
$$

Figure 2 illustrates the graphical model implied by this factorization.

For $t \in ( 0 , 1 ]$ , we introduce a simplex-valued variable $\mathbf { w } _ { t } \in \Delta ^ { K - 1 }$ through

$$
q ( \mathbf { w } _ { t } \mid \mathbf { z } _ { t } , \mathbf { x } ) = \operatorname { D i r } ( \mathbf { w } _ { t } ; \boldsymbol { \eta } _ { t } \mathbf { p } _ { t } + \mathbf { z } _ { t } ) ,\tag{7}
$$

where $\eta _ { t } > 0$ is a concentration parameter. The mean of this Dirichlet distribution is centered at the diffusiontime categorical marginal, while the additive one-hot count $\mathbf { z } _ { t }$ anchors the relaxed state to the sampled discrete token.

![](images/f851c1a8b30b4ac4ed511475fad4e0d4269e7343428005f0a9fdd8bfc0da7119.jpg)  
Figure 2: Graphical model corresponding to the factorization in (6).

This construction yields an exact Dirichlet–categorical hierarchy.

Proposition 1. Assume $t > 0 .$ . The Dirichlet–categorical hierarchy satisfies the following properties; statements involving ${ \bf w } _ { s }$ additionally require $s > 0$

1. The marginal distribution ofthe relaxed state is

$$
q ( \mathbf { w } _ { t } \mid \mathbf { x } ) = \operatorname { D i r } ( \mathbf { w } _ { t } ; \eta _ { t } \mathbf { p } _ { t } ) .\tag{8}
$$

2. Given $\mathbf { w } _ { t } ,$ , the variables x and $\mathbf { z } _ { t }$ are conditionally independent, and the discrete state can be recoveredfrom the relaxed state via

$$
q ( \mathbf { z } _ { t } \mid \mathbf { w } _ { t } , \mathbf { x } ) = q ( \mathbf { z } _ { t } \mid \mathbf { w } _ { t } ) = \operatorname { C a t } ( \mathbf { z } _ { t } ; \mathbf { w } _ { t } ) .\tag{9}
$$

3. For $s < t ,$ , the reverse conditional posterior of $\mathbf { \dot { z } } _ { s }$ given $\mathbf { w } _ { t }$ and x is categorical:

$$
q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) = \operatorname { C a t } \left( \mathbf { z } _ { s } ; \pmb { \rho } _ { s \mid t } ( \mathbf { x } , \mathbf { w } _ { t } ) \right) ,\tag{10}
$$

where

$$
\begin{array} { r } { \rho _ { s | t } ( \mathbf { x } , \mathbf { w } _ { t } ) : = \mathbf { p } _ { s } \odot \left[ \alpha _ { t | s } \big ( \mathbf { w } _ { t } \oslash \mathbf { p } _ { t } \big ) + ( 1 - \alpha _ { t | s } ) \ \langle \mathbf { w } _ { t } , \pi \oslash \mathbf { p } _ { t } \rangle \ \mathbf { 1 } \right] . } \end{array}\tag{11}
$$

4. For $0 < s < t$ , the reverse conditional posterior of ${ \bf w } _ { s }$ given $\mathbf { w } _ { t }$ and x is a Dirichlet mixture:

$$
q ( \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) = \sum _ { k = 1 } ^ { K } \rho _ { s \mid t , k } ( \mathbf { x } , \mathbf { w } _ { t } ) \operatorname { D i r } ( \mathbf { w } _ { s } ; \eta _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k } ) .\tag{12}
$$

See Section A for the proof. These identities show that $\mathbf { w } _ { t }$ is not an ad hoc surrogate. It is an exact auxiliary variable whose marginal remains native to the simplex and whose decoder back to $\mathbf { z } _ { t }$ is simply categorical sampling from $\mathbf { w } _ { t }$

## 3.2 Training objective

A natural starting point is to match the simplex bridge directly:

$$
\mathcal { L } _ { w _ { s } \mid w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) = D _ { \mathrm { K L } } [ q ( \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) \parallel q ( \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ] .\tag{13}
$$

This is the most direct objective associated with the relaxed bridge, but it is generally intractable because $q ( \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } )$ is a Dirichlet mixture, as shown in (12).

At training time, we sample the augmented noisy state as

$$
\mathbf { w } _ { t } \sim q ( \mathbf { w } _ { t } \mid \mathbf { x } ) , \qquad \mathbf { z } _ { t } \sim q ( \mathbf { z } _ { t } \mid \mathbf { w } _ { t } ) = { \mathrm { C a t } } ( \mathbf { z } _ { t } ; \mathbf { w } _ { t } ) ,
$$

and predict the clean-token distribution from the categorical state:

$$
\widehat { \mathbf { x } } _ { \theta } = f _ { \theta } ( \mathbf { z } _ { t } , t ) .
$$

To define the relaxed discrete bridge objective, let $\widetilde { \mathbf { z } } _ { t }$ denote a second categorical variable satisfying

$$
\widetilde { \mathbf { z } } _ { t } \sim q \big ( \widetilde { \mathbf { z } } _ { t } \mid \mathbf { w } _ { t } \big ) = \mathrm { C a t } \big ( \widetilde { \mathbf { z } } _ { t } ; \mathbf { w } _ { t } \big ) ,
$$

conditionally independently of the network input $\mathbf { z } _ { t }$ given $\mathbf { w } _ { t } .$ . We optimize

$$
\begin{array} { r } { \bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \mathbf { z } _ { t } , \mathbf { x } ; s , t ) : = \mathbb { E } _ { q ( \widetilde { \mathbf { z } } _ { t } | \mathbf { w } _ { t } ) } \Big [ \mathrm { K L } \left( q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { x } ) \parallel q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \widehat { \mathbf { x } } _ { \theta } ) \right) \Big ] , } \end{array}\tag{14}
$$

where $\widehat { \mathbf { x } } _ { \theta } = f _ { \theta } ( \mathbf { z } _ { t } , t )$ . This objective averages the standard discrete reverse KL divergence over an auxiliary decoder sample from $\mathbf { w } _ { t } ,$ , while retaining $\mathbf { z } _ { t }$ as the denoiser input (Figure 1).

Rao–Blackwellized form. The expectation over the auxiliary decoder sample $\widetilde { \mathbf { z } } _ { t }$ in (14) can be marginalized exactly. Equivalently, the resulting expression is the Rao–Blackwellized form of the Monte Carlo estimator obtained by first sampling $\widetilde { \mathbf { z } } _ { t } \sim q ( \widetilde { \mathbf { z } } _ { t } \mid \mathbf { w } _ { t } )$ and then evaluating the discrete reverse KL. The independently sampled $\mathbf { z } _ { t }$ remains the denoiser input and is not marginalized by this step.

Proposition 2. The relaxed discrete bridge objective in (14) admits the exact closedform

$$
\bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) = \langle \mathbf { w } _ { t } , \log \hat { \mathbf { p } } _ { t } - \log \mathbf { p } _ { t } \rangle + \left. \rho _ { s | t } ( \mathbf { x } , \mathbf { w } _ { t } ) , \log \mathbf { p } _ { s } - \log \hat { \mathbf { p } } _ { s } \right. .\tag{15}
$$

The proof is given in Section B. Equation (15) is fully tractable and eliminates sampling noise associated with the auxiliary decoder sample $\widetilde { \mathbf { z } } _ { t }$ . Operationally, it shows that the reverse-bridge loss depends on $\mathbf { w } _ { t }$ only through two quantities: the decoded current-time marginal term $\left. \mathbf { w } _ { t } , \log \hat { \mathbf { p } } _ { t } \right.$ and the induced reverse posterior $\rho _ { s | t } ( \mathbf { x } , \mathbf { w } _ { t } )$ . The network prediction itself remains conditioned on the separately sampled categorical input $\mathbf { z } _ { t }$

Continuous-time limit. The closed form in (15) yields a non-degenerate infinitesimal limit. Let $s = t - \Delta$ with $\Delta \downarrow 0$

Proposition 3. Up to θ-independent additive terms, the relaxed discrete bridge objective satisfies

$$
\bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; t - \Delta , t ) = \Delta \ell _ { \mathrm { c t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } , t ) + o ( \Delta ) ,\tag{16}
$$

where

$$
\begin{array} { r } { \ell _ { \mathrm { c t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } , t ) = \lambda ( t ) \Bigg [ \left. \mathbf { w } _ { t } , \pi \mathcal { O } \hat { \mathbf { p } } _ { t } \right. - \left. \mathbf { w } _ { t } , \pi \mathcal { O } \mathbf { p } _ { t } \right. \left. \mathbf { p } _ { t } , \log \hat { \mathbf { p } } _ { t } \right. + \left. \pi \mathcal { O } \left( \mathbf { w } _ { t } \mathcal { O } \mathbf { p } _ { t } \right) , \log \hat { \mathbf { p } } _ { t } \right. \Bigg ] . } \end{array}\tag{17}
$$

and $\begin{array} { r } { \lambda ( t ) : = - \frac { \mathrm { d } } { \mathrm { d } t } \log \alpha ( t ) } \end{array}$ . Consequently, the corresponding continuous-time objective is

$$
\mathcal { L } _ { \mathrm { c t } } = \int _ { 0 } ^ { 1 } \mathbb { E } _ { q ( \mathbf { x } ) } \left[ \mathbb { E } _ { q ( \mathbf { w } _ { t } \mid \mathbf { x } ) } [ \ell _ { \mathrm { c t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } , t ) ] \right] ~ \mathrm { d } t .\tag{18}
$$

The proof is deferred to Section C.1. This proposition identifies (18) as the continuous-time counterpart of (14).

The structure of (14) and (18) is closely related to UDLM [Schiff et al., 2025]. UDLM derives a continuous-time reverse-KL objective directly from the discrete corrupted state (5). Our construction introduces the exact auxiliary variable $\mathbf { w } _ { t } .$ , averages the same reverse-KL bridge over an auxiliary draw $\widetilde { \mathbf { z } } _ { t } \sim q ( \widetilde { \mathbf { z } } _ { t } \mid \mathbf { w } _ { t } ) = \mathrm { C a t } ( \mathbf { w } _ { t } )$ , and then takes the infinitesimal limit. In this sense, (18) can be understood as a simplex-relaxed continuous-time analogue of the UDLM objective.

## 3.3 Sampling

At inference time, we run the reverse process on a grid $1 = t _ { N } > t _ { N - 1 } > \cdot \cdot \cdot > t _ { 0 } = 0$ . For a denoiser conditioned on $\mathbf { z } _ { t } ,$ , the Dirichlet–categorical hierarchy yields a stochastic sampler that maintains the augmented state $\left( \mathbf { z } _ { t } , \mathbf { w } _ { t } \right)$ at positive times. Since the endpoint marginal is $q ( \mathbf { w } _ { 1 } ) =$ $\mathrm { D i r } ( \mathbf { w } _ { 1 } ; \eta _ { 1 } \pmb { \pi } )$ and the exact decoder is $q ( \mathbf { z } _ { 1 } \mid \mathbf { w } _ { 1 } ) = \mathrm { C a t } ( \mathbf { z } _ { 1 } ; \mathbf { w } _ { 1 } )$ , generation starts from

$$
\begin{array} { r } { { \bf w } _ { t _ { N } } \sim \mathrm { D i r } ( \eta _ { t _ { N } } \pi ) , \qquad { \bf z } _ { t _ { N } } \sim \mathrm { C a t } ( { \bf w } _ { t _ { N } } ) . } \end{array}\tag{19}
$$

Given adjacent times $s = t _ { n - 1 } < t = t _ { n }$ and the current pair $\left( \mathbf { z } _ { t } , \mathbf { w } _ { t } \right)$ , the denoiser predicts

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { \theta } = f _ { \theta } ( \mathbf { z } _ { t } , t ) , } \end{array}\tag{20}
$$

which induces the bridge marginals $\hat { \bf p } _ { s }$ and $\hat { \mathbf { p } } _ { t }$ and the reverse categorical posterior $\rho _ { s | t } \big ( \hat { \mathbf { x } } _ { \theta } , \mathbf { w } _ { t } \big )$ . Our default sampler is the stochastic ancestral sampler implied by the Dirichlet–categorical hierarchy. Each reverse step draws

$$
\begin{array} { r } { \mathbf { z } _ { s } \sim \operatorname { C a t } \bigl ( \rho _ { s | t } \bigl ( \hat { \mathbf { x } } _ { \theta } , \mathbf { w } _ { t } \bigr ) \bigr ) , \qquad \mathbf { w } _ { s } \sim \operatorname { D i r } \bigl ( \eta _ { s } \hat { \mathbf { p } } _ { s } + \mathbf { z } _ { s } \bigr ) . } \end{array}\tag{21}
$$

The sampled $\mathbf { z } _ { s }$ is used as the network input at the next reverse step, while ${ \bf w } _ { s }$ carries the auxiliary bridge information required by the next reverse posterior. Repeating (21) from $t _ { N } = 1 \mathrm { t o } t _ { 0 } = 0$ yields the final categorical sample $\mathbf { z } _ { 0 }$ . The categorical input also has a computational advantage. In a standard token model, $\mathbf { z } _ { t }$ is stored as an integer token index and its embedding is obtained by lookup. Feeding the dense simplex vector $\mathbf { w } _ { t }$ instead requires computing $\mathbf { w } _ { t } ^ { \mathsf { T } } E$ for the vocabulary embedding matrix $E$ at every sequence position, adding a vocabulary-sized dense matrix multiplication and the associated memory traffic. The main experiments therefore use the $\mathbf { z } _ { t }$ -input formulation above.

## 4 Experiments

We evaluate Simplax on unconditional text generation with OpenWebText [Gokaslan and Cohen, 2019] and constrained categorical generation with Sudoku [Lee et al., 2026, Deschenaux and Gulcehre, 2026]. We first report compact design diagnostics on OpenWebText and then present the main comparisons.

## 4.1 Experimental setup

OpenWebText. We tokenize OpenWebText with the GPT-2 BPE tokenizer [Radford et al., 2019], giving $| \mathcal { V } | = 5 0 \AA , 2 5 7 \mathrm { \Omega }$ , and use sequence length $L = 1 , 0 2 4$ . All methods use the 179M-parameter diffusion transformer of Sahoo et al. [2024]: 12 transformer blocks, rotary position embeddings [Su et al., 2024], AdaLN time conditioning [Peebles and Xie, 2023], and a softmax output head. Models are trained with Adam [Kingma and Ba, 2014], learning rate $\bar { 3 } \times 1 0 ^ { - 4 }$ , batch size 512, and a total budget of 1M iterations. Unless stated otherwise, Simplax uses $\mathbf { z } _ { t }$ as the denoiser input and a constant concentration $\eta _ { t } \equiv 0 . 0 1$

Sudoku. We build on the Sudoku benchmark of Deschenaux and Gulcehre [2026], while using a cross-clue generalization protocol in which all models are trained only on puzzles with 30 clues. The dataset contains 48,000 training and 2,000 validation puzzles, each constructed to have a unique solution. A Sudoku instance is represented as a 180-token sequence consisting of a 91-token puzzle prefix and an 89-token solution. The puzzle prefix contains a BOS token, all 81 cells with unobserved cells represented by a blank token, eight row separators, and a second BOS token. The solution contains the 81 completed cells and eight row separators. The training loss is applied only to the solution tokens.

All methods use Transformer backbones with eight blocks, hidden dimension 512, eight attention heads, and dropout 0.1. Their parameter counts range from 25.21M to 28.59M; the principal differences are the training objective, time conditioning, and inference procedure. Models are trained for 20,000 steps using Adam with learning rate $3 \stackrel { - } { \times } 1 0 ^ { - 4 }$ and global batch size 256. Further architectural and optimization details are provided in Section E.1.

At inference time, the same 30-clue-trained checkpoint is evaluated with 40, 35, 30, 25, 20, and 17 clues. The 30-clue setting matches the training distribution, while the remaining settings evaluate transfer across clue densities. The 40- and 35-clue settings provide more conditioning information than observed during training, whereas the 25-, 20-, and 17-clue settings provide progressively less conditioning information. In particular, 17 is the minimum number of clues for which a standard $9 \times 9$ Sudoku puzzle can admit a unique solution [McGuire et al., 2014, Lin et al., 2013], making the 17-clue setting the most sparsely conditioned regime in our evaluation. We additionally evaluate generation from an all-blank puzzle prefix, which contains no clue information and is treated as unconditional Sudoku generation.

![](images/3ab78ed07c60c3b9fce5f43a5b37d45bab008d3c03fce0b7b87eaa2dc360accc.jpg)  
Figure 3: Design diagnostics on OpenWebText. The rows show temperature-swept generation frontiers at NFE = 16 and 128. The columns compare self-conditioning (a, d), denoiser input $\mathbf { w } _ { t }$ versus ${ \bf z } _ { t } \left( { \bf b } , { \bf e } \right) ;$ , and initialization from a pretrained UDLM checkpoint under a matched 1M-iteration budget (c, f). The dotted line marks the OpenWebText entropy, 5.44. Gen. PPL is evaluated with GPT-2 Large; lower is better at comparable Gen. ENT.

Metrics. For OpenWebText, we draw 1, 024 sequences and report generative unigram entropy and generative perplexity under GPT-2 Large, GPT-2 XL [Radford et al., 2019], and Llama-2 7B [Touvron et al., 2023]. The OpenWebText unigram entropy is 5.44 nats. For conditional Sudoku, we report solving accuracy. For unconditional Sudoku, we report validity, the fraction of generated boards satisfying all Sudoku constraints.

## 4.2 Design diagnostics on OpenWebText

The input and self-conditioning diagnostics use 50k-step runs with the same tokenizer, sequence length, and backbone as the main experiment. The initialization comparison matches the total training budget at 1M iterations. These experiments characterize individual design choices rather than provide the main method comparison.

Self-conditioning. For the $\mathbf { w } _ { t }$ -input diagnostic, the preferred setting depends on the inference budget: omitting self-conditioning is better near the data-entropy operating point at $\mathrm { N F E } = 1 6$ whereas using it is better at NFE = 128 Figure 3(a, d). The experiment therefore does not support a budget-independent conclusion.

Denoiser input. The bridge and objective do not require the auxiliary simplex state itself to be the denoiser input. With the same $\mathbf { w } _ { t }$ -based objective, the $\mathbf { z } _ { t }$ -input model attains lower Gen. PPL at comparable Gen. ENT at both NFE values Figure 3(b, e). It also avoids an additional dense projection at the input layer: token indices use embedding lookup, whereas a simplex input requires $\dot { \mathbf { w } } _ { t } ^ { \mathsf { T } } \dot { E }$ with the vocabulary embedding matrix E at every sequence position. We therefore use $\mathbf { z } _ { t }$ as the denoiser input in the main experiments while retaining $\mathbf { w } _ { t }$ in the objective and reverse update.

Table 1: OpenWebText unconditional generation at selected NFE values. Gen. ENT is generative unigram entropy and should be compared with the data entropy 5.44. Gen. PPL is evaluated by the indicated external language model. The best and second-best values in each column are shown in bold and underlined, respectively.
<table><tr><td></td><td colspan="4">NFE = 16</td><td colspan="4">NFE = 128</td><td colspan="4">NFE = 1,024</td></tr><tr><td>Method</td><td></td><td>Ent. GPT-2 L GPT-2 XL Llama-2</td><td></td><td></td><td>Ent.</td><td>GPT-2 L GPT-2 XL Llama-2</td><td></td><td></td><td></td><td>Ent. GPT-2 L GPT-2 XL Llama-2</td><td></td><td></td></tr><tr><td>CANDI [Pynadath et al., 2025]</td><td>5.43</td><td>97.2</td><td>99.6</td><td>56.0</td><td>5.45</td><td>67.2</td><td>69.2</td><td>36.8</td><td>5.46</td><td>74.3</td><td>76.6</td><td>39.2</td></tr><tr><td>UDLM [Schiff et al., 2025]</td><td>5.53</td><td>186.0</td><td>190.1</td><td>79.0</td><td>5.49</td><td>62.9</td><td>64.9</td><td>34.6</td><td>5.46</td><td>59.2</td><td>61.2</td><td>32.2</td></tr><tr><td>MDLM [Sahoo et al., 2024]</td><td>5.47</td><td>117.9</td><td>120.6</td><td>63.8</td><td>5.46</td><td>65.4</td><td>67.2</td><td>37.3</td><td>5.40</td><td>55.1</td><td>56.6</td><td>33.9</td></tr><tr><td>Duo [Sahoo et al., 2025]</td><td>5.45</td><td>166.1</td><td>168.3</td><td>87.1</td><td>5.45</td><td>93.9</td><td>95.9</td><td>53.3</td><td>5.43</td><td>88.9</td><td>90.6</td><td>50.4</td></tr><tr><td>FLM [Lee et al., 2026]</td><td>5.58</td><td>259.5</td><td>262.6</td><td>139.3</td><td>5.42</td><td>112.0</td><td>113.7</td><td>66.5</td><td>5.45</td><td>125.6</td><td>127.2</td><td>74.4</td></tr><tr><td>LangFlow [Chen et al., 2026]</td><td>5.42</td><td>115.1</td><td>116.6</td><td>59.9</td><td>5.42</td><td>60.2</td><td>61.7</td><td>30.0</td><td>5.41</td><td>68.3</td><td>70.0</td><td>28.2</td></tr><tr><td>S-FLM [Deschenaux and Gulcehre, 2026] 5.45</td><td></td><td>124.6</td><td>126.4</td><td>62.9</td><td>5.43</td><td>103.4</td><td>105.1</td><td>52.4</td><td>5.46</td><td>108.5</td><td>110.2</td><td>53.7</td></tr><tr><td>Simplax</td><td>5.46</td><td>90.5</td><td>93.1</td><td>49.3</td><td>5.45</td><td>56.9</td><td>58.9</td><td>31.4</td><td>5.44</td><td>45.1</td><td>46.8</td><td>25.5</td></tr></table>

![](images/bf9df8fa9692334f163bf850508967ce8e9dce7e31a534526fcc959f1109dbc1.jpg)  
Figure 4: Llama-2 7B generative frontiers on OpenWebText for NFE ∈ {8, 16, 32, 64, 128, 256, 512, 1, 024}. Each panel shows the temperature-swept Gen. PPL– Gen. ENT tradeoff. Lower Gen. PPL is better, and the reference data entropy is 5.44.

UDLM initialization. We compare Simplax trained from scratch for 1M iterations with a model trained as UDLM for 800k iterations and then with the Simplax objective for 200k iterations. UDLM initialization improves the Gen. PPL–Gen. ENT frontier at both NFE values Figure 3(c, f).

## 4.3 Unconditional generation on OpenWebText

We compare Simplax with MDLM [Sahoo et al., 2024], UDLM [Schiff et al., 2025], Duo [Sahoo et al., 2025], CANDI [Pynadath et al., 2025], FLM [Lee et al., 2026], LangFlow [Chen et al., 2026], and S-FLM [Deschenaux and Gulcehre, 2026]. For each method and NFE budget, we sweep 15 temperatures from 0.84 to 1.12 in increments of 0.02 and select the operating point whose generated entropy is closest to 5.44 nats.

Simplax has the lowest Gen. PPL under all three evaluators at NFE = 16 and 1, 024. At NFE = 128, it is best under GPT-2 Large and $\mathrm { { G P T } } { \cdot } 2 \mathrm { { X L } } ,$ while LangFlow is best under Llama-2 7B. The temperature-swept Llama-2 7B frontiers across all evaluated NFE budgets are shown in Figure 4.

Table 2: Conditional Sudoku solving accuracy and unconditional Sudoku validity in percent. All models are trained with 30 clues. The 40- and 35-clue settings evaluate transfer to more heavily conditioned inputs, the 30-clue setting matches the training clue density, and the 25-, 20-, and 17-clue settings evaluate transfer to progressively less heavily conditioned inputs. The best and second-best results in each column are shown in bold and underlined, respectively.
<table><tr><td></td><td colspan="6">Conditional accuracy (%)</td><td>Unconditional validity (%)</td></tr><tr><td>Method</td><td>40 clues</td><td>35 clues</td><td>30 clues</td><td>25 clues</td><td>20 clues</td><td>17 clues</td><td>0 clues</td></tr><tr><td>AR</td><td>16.00</td><td>4.05</td><td>0.70</td><td>0.05</td><td>0.00</td><td>0.00</td><td>8.15</td></tr><tr><td>Duo [Sahoo et al., 2025]</td><td>97.00</td><td>84.15</td><td>48.85</td><td>16.00</td><td>4.80</td><td>0.40</td><td>80.95</td></tr><tr><td>FLM [Lee et al., 2026]</td><td>96.45</td><td>83.80</td><td>48.30</td><td>14.05</td><td>3.30</td><td>0.45</td><td>34.05</td></tr><tr><td>MDLM [Sahoo et al., 2024]</td><td>98.45</td><td>85.15</td><td>48.00</td><td>11.80</td><td>2.70</td><td>0.20</td><td>22.35</td></tr><tr><td>S-FLM</td><td>77.70</td><td>42.40</td><td>10.75</td><td>1.15</td><td>0.05</td><td>0.00</td><td>1.25</td></tr><tr><td>S-FLM (truncated-adaptive)</td><td>94.70</td><td>78.70</td><td>41.60</td><td>11.90</td><td>2.95</td><td>0.25</td><td>3.45</td></tr><tr><td>Simplax</td><td>98.55</td><td>91.05</td><td>61.75</td><td>25.90</td><td>8.80</td><td>1.20</td><td>95.85</td></tr></table>

## 4.4 Constrained categorical generation on Sudoku

Simplax achieves the highest performance across all conditional and unconditional settings in Table 2. Its advantage extends beyond the 30-clue training distribution to both more and less conditioned inputs, including the challenging low-clue regimes. For unconditional generation, Simplax achieves 95.85% validity, compared with 80.95% for the strongest baseline.

## 5 Related Work

Discrete diffusion for categorical data. Discrete diffusion for categorical data was developed through early multinomial formulations and later unified and substantially generalized by D3PM, which introduced structured transition kernels such as uniform and absorbing corruptions and established the standard variational training recipe [Hoogeboom et al., 2021, Austin et al., 2021]. This framework was subsequently extended to continuous-time formulations and alternative reverse objec tives [Campbell et al., 2022, Lou et al., 2024, Zhang et al., 2025], and has since supported a broad line of language-modeling work covering masked, absorbing, and uniform-state diffusion [Sahoo et al., 2024, Shi et al., 2024, Ou et al., 2025, Zheng et al., 2025, Sahoo et al., 2025, Deschenaux et al., 2026]. Our method stays within this discrete-diffusion lineage: we keep the original categorical forward process and reverse posterior, and do not replace the primary generative state.

Auxiliary-variable and hybrid formulations. A parallel line of work enriches discrete diffusion through auxiliary variables or structured reverse distributions. Di4C [Hayakawa et al., 2025] uses mixtures of product models to capture dimensional correlations, VADD [Xie et al., 2026] introduces a Gaussian latent into masked denoising, and CoDD couples factorized outputs with probabilistic circuits [Li et al., 2026]. Continuous and hybrid constructions include Gaussian-relaxed views in Duo and Duo++ [Sahoo et al., 2025, Deschenaux et al., 2026], Euclidean denoising over one-hot states in FLM [Lee et al., 2026], and discrete–continuous diffusion in CADD and CANDI [Zheng et al., 2026, Pynadath et al., 2025]. Simplax instead introduces a simplex-valued auxiliary variable while preserving categorical diffusion. Unlike methods that use the auxiliary variable as the denoiser input or primary generative state, the z -input Simplax formulation retains the categorical state as the network input and uses the simplex variable to define the reverse-bridge objective and sampler.

Diffusion and flow on the simplex. Our method is also related to work that defines the generative process itself on the simplex. This includes simplex diffusion based on softmax-transformed continuous processes [Floto et al., 2023], simplex diffusion via categorical SDEs and Cox–Ingersoll–Ross dynamics [Richemond et al., 2023], Dirichlet-based score models such as DDSM [Avdeyev et al., 2023], Dirichlet Flow Matching [Stark et al., 2024], and recent unifying views of discrete, Gaussian, and simplicial diffusion [Chandra et al., 2026]. These methods are close to ours in geometry, since they treat simplex-valued states as first-class objects, but differ in role: in our formulation, the simplex variable is not the primary generative state, but an exact auxiliary bridge attached to a standard discrete diffusion process.

## 6 Conclusion and Limitations

We introduced Simplax, an exact Dirichlet–categorical augmentation of uniform discrete diffusion. Simplax preserves the categorical forward process while introducing an auxiliary simplex state to derive a Rao–Blackwellized reverse-bridge objective and stochastic ancestral sampler. It improves the Gen. PPL–Gen. ENT tradeoff on OpenWebText and achieves the highest Sudoku performance among the compared methods across all evaluated clue densities, from 40 to 17 clues, as well as in unconditional generation.

Limitation. The present formulation is specialized to uniform categorical corruption and introduces an auxiliary simplex-valued state whose computational overhead relative to standard discrete diffusion has not been fully characterized. Moreover, the concentration schedule remains an additional design choice rather than being determined by the theory. Extending the construction to broader categorical corruption kernels and developing more efficient reverse solvers are important directions for future work.

## Acknowledgments

We thank Chanhyuk Lee, Jaehoon Yoo, and Jinwoo Kim for insightful discussions.

## References

Ali Alp. Sudoku puzzle generator. https://github.com/alicommit-malp/sudoku, 2024.

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems, 2021.

Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, and Jian Zhou. Dirichlet diffusion score model for biological sequence generation. In International Conference on Machine Learning, 2023.

Andrew Campbell, Joe Benton, Valentin De Bortoli, Tom Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. In Advances in Neural Information Processing Systems, 2022.

Nuria Alina Chandra, Yucen Lily Li, Alan Nawzad Amin, Alex Ali, Joshua Rollins, Sebastian W. Ober, Aniruddh Raghu, and Andrew Gordon Wilson. A unification of discrete, gaussian, and simplicial diffusion. In The Fourteenth International Conference on Learning Representations, 2026.

Yuxin Chen, Chumeng Liang, Hangke Sui, Ruihan Guo, Chaoran Cheng, Jiaxuan You, and Ge Liu. Langflow: Continuous diffusion rivals discrete in language modeling, 2026.

Thomas M. Cover and Joy A. Thomas. Elements ofInformation Theory. 2006.

Justin Deschenaux and Caglar Gulcehre. Language modeling with hyperspherical flows, 2026.

Justin Deschenaux, Caglar Gulcehre, and Subham Sekhar Sahoo. The diffusion duality, chapter II: \$\psi\$-samplers and efficient curriculum. In The Fourteenth International Conference on Learning Representations, 2026.

Griffin Floto, Thorsteinn Jonsson, Mihai Nica, Scott Sanner, and Eric Zhengyu Zhu. Diffusion on the probability simplex. In ICML 2023 Workshop: Sampling and Optimization in Discrete Space, 2023.

Aaron Gokaslan and Vanya Cohen. OpenWebText corpus, 2019.

Satoshi Hayakawa, Yuhta Takida, Masaaki Imaizumi, Hiromi Wakaki, and Yuki Mitsufuji. Distillation of discrete diffusion through dimensional correlations. In Proceedings of the 42nd International Conference on Machine Learning, 2025.

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. In Advances in Neural Information Processing Systems, 2021.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2014.

Solomon Kullback and Richard A. Leibler. On information and sufficiency. The Annals of Mathematical Statistics, 1951.

Chanhyuk Lee, Jaehoon Yoo, Manan Agarwal, Sheel Shah, Jerry Huang, Aditi Raghunathan, Seunghoon Hong, Nicholas M. Boffi, and Jinwoo Kim. One-step language modeling via continuous denoising, 2026.

Ian Li, Zilei Shao, Benjie Wang, Rose Yu, Guy Van den Broeck, and Anji Liu. Breaking the factorization barrier in diffusion language models. In Forty-third International Conference on Machine Learning, 2026.

Hung-Hsuan Lin, I-Chen Wu, and Tinghan Wei. On specific 17-clue sudoku puzzles. ICGA Journal, 2013.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In International Conference on Machine Learning, 2024.

Gary McGuire, Bastian Tugemann, and Gilles Civario. There is no 16-clue sudoku: Solving the sudoku minimum number of clues problem via hitting set enumeration. Experimental Mathematics, 2014.

Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. In The Thirteenth International Conference on Learning Representations, 2025.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, 2023.

Patrick Pynadath, Jiaxin Shi, and Ruqi Zhang. Candi: Hybrid discrete-continuous diffusion models, 2025.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI Technical Report, 2019.

Pierre Harvey Richemond, Sander Dieleman, and Arnaud Doucet. Categorical SDEs with simplex diffusion. In ICML 2023 Workshop: Sampling and Optimization in Discrete Space, 2023.

Subham Sekhar Sahoo, Marianne Arriola, Aaron Gokaslan, Edgar Mariano Marroquin, Alexander M Rush, Yair Schiff, Justin T Chiu, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin T Chiu, and Volodymyr Kuleshov. The diffusion duality. In Forty-second International Conference on Machine Learning, 2025.

Subham Sekhar Sahoo, Jean-Marie Lemercier, Zhihan Yang, Justin Deschenaux, Jingyu Liu, John Thickstun, and Ante Jukic. Scaling beyond masked diffusion language models. In The Forty-Third International Conference on Machine Learning, 2026.

Yair Schiff, Subham Sekhar Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dalla-torre, Bernardo P de Almeida, Alexander M Rush, Thomas PIERROT, and Volodymyr Kuleshov. Simple guidance mechanisms for discrete diffusion models. In The Thirteenth International Conference on Learning Representations, 2025.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Hannes Stark, Bowen Jing, Chenyu Wang, Gabriele Corso, Bonnie Berger, Regina Barzilay, and Tommi Jaakkola. Dirichlet flow matching with applications to DNA sequence design. In Forty-first International Conference on Machine Learning, 2024.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568, 2024.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Dimitri von Rütte, Janis Fluri, Yuhui Ding, Antonio Orvieto, Bernhard Schölkopf, and Thomas Hofmann. Generalized interpolating discrete diffusion. In Forty-second International Conference on Machine Learning, 2025.

Dimitri von Rütte, Janis Fluri, Omead Pooladzandi, Bernhard Schölkopf, Thomas Hofmann, and Antonio Orvieto. Scaling behavior of discrete diffusion language models. In The Fourteenth International Conference on Learning Representations, 2026.

Tianyu Xie, Shuchen Xue, Zijin Feng, Tianyang Hu, Jiacheng Sun, Zhenguo Li, and Cheng Zhang. Variational autoencoding discrete diffusion with enhanced dimensional correlations modeling. In The Fourteenth International Conference on Learning Representations, 2026.

Ruixiang Zhang, Shuangfei Zhai, Yizhe Zhang, James Thornton, Zijing Ou, Joshua M. Susskind, and Navdeep Jaitly. Target concrete score matching: A holistic framework for discrete diffusion. In Forty-second International Conference on Machine Learning, 2025.

Huangjie Zheng, Shansan Gong, Ruixiang Zhang, Tianrong Chen, Jiatao Gu, Mingyuan Zhou, Navdeep Jaitly, and Yizhe Zhang. Continuously augmented discrete diffusion model for categorical generative modeling. In The Fourteenth International Conference on Learning Representations, 2026.

Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. In The Thirteenth International Conference on Learning Representations, 2025.

# Simplex Relaxation for Discrete Diffusion

# Supplementary Material

## A Auxiliary Identities and Full Dirichlet–Categorical Hierarchy

This appendix develops the exact probabilistic structure behind the simplex relaxation. The formulas below apply at positive diffusion times, where $\mathbf { p } _ { t }$ is strictly positive under the assumptions in Section 2. We define the clean endpoint separately as the categorical state $\mathbf { z } _ { 0 } = \mathbf { x } ;$ the hierarchy does not introduce a Dirichlet variable $\mathbf { w } _ { 0 }$

The key point is that the auxiliary state $\mathbf { w } _ { t }$ is not introduced as a heuristic soft surrogate. Rather, once we specify the Dirichlet bridge

$$
q ( \mathbf { w } _ { t } \mid \mathbf { z } _ { t } , \mathbf { x } ) = \operatorname { D i r } ( \mathbf { w } _ { t } ; \boldsymbol { \eta } _ { t } \mathbf { p } _ { t } + \mathbf { z } _ { t } ) ,
$$

the resulting joint model admits a closed hierarchy in both directions: the relaxed state has an exact Dirichlet marginal, the discrete state can be decoded exactly from $\mathbf { w } _ { t }$ , and the reverse bridge remains tractable after marginalizing either $\mathbf { z } _ { t }$ or $\mathbf { z } _ { s }$ . We begin with two elementary identities that make these cancellations possible.

## A.1 Useful identities

The first identity is the basic shift formula for Dirichlet densities. It shows that adding a one-hot count to the concentration vector simply multiplies the base Dirichlet density by the corresponding simplex coordinate.

Lemma 1. Let $\pmb { \alpha } \in \mathbb { R } _ { > 0 } ^ { K }$ and let $\textstyle \alpha _ { 0 } = \sum _ { i = 1 } ^ { K } \alpha _ { i }$ . Then, for any $k \in \{ 1 , \ldots , K \}$

$$
\operatorname { D i r } ( \mathbf { w } ; \alpha + \mathbf { e } _ { k } ) = { \frac { \alpha _ { 0 } } { \alpha _ { k } } } w _ { k } \operatorname { D i r } ( \mathbf { w } ; \alpha ) .\tag{22}
$$

Proof. By definition,

$$
\mathrm { D i r } ( \mathbf { w } ; \pmb { \alpha } ) = \frac { 1 } { B ( \pmb { \alpha } ) } \prod _ { i = 1 } ^ { K } w _ { i } ^ { \alpha _ { i } - 1 } , \qquad B ( \pmb { \alpha } ) = \frac { \prod _ { i = 1 } ^ { K } \Gamma ( \alpha _ { i } ) } { \Gamma ( \alpha _ { 0 } ) } .
$$

Hence

$$
\mathrm { D i r } ( \mathbf { w } ; { \pmb \alpha } + { \mathbf e } _ { k } ) = \frac { 1 } { B ( { \pmb \alpha } + { \mathbf e } _ { k } ) } w _ { k } \prod _ { i = 1 } ^ { K } w _ { i } ^ { \alpha _ { i } - 1 } .
$$

It remains to compare the normalizing constants:

$$
{ \frac { B ( \alpha + { \bf e } _ { k } ) } { B ( \alpha ) } } = { \frac { \Gamma ( \alpha _ { k } + 1 ) } { \Gamma ( \alpha _ { k } ) } } { \frac { \Gamma ( \alpha _ { 0 } ) } { \Gamma ( \alpha _ { 0 } + 1 ) } } = { \frac { \alpha _ { k } } { \alpha _ { 0 } } } .
$$

Therefore

$$
\frac { 1 } { B ( \pmb { \alpha } + \mathbf { e } _ { k } ) } = \frac { \alpha _ { 0 } } { \alpha _ { k } } \frac { 1 } { B ( \pmb { \alpha } ) } ,
$$

which proves (22).

Specializing this identity to concentrations of the form $\eta \mathbf { p }$ yields the cancellation that will be used throughout the appendix.

Corollary 1. Let $\mathbf { p } \in \Delta ^ { K - 1 }$ satisfy $p _ { k } > 0$ for all $k ,$ let $\eta > 0 ,$ , and define $\mathbf { \alpha } \alpha = \eta \mathbf { p }$ . Then

$$
p _ { k } \mathrm { D i r } ( \mathbf { w } ; \eta \mathbf { p } + \mathbf { e } _ { k } ) = w _ { k } \mathrm { D i r } ( \mathbf { w } ; \eta \mathbf { p } ) .\tag{23}
$$

Proof. Apply Lemma 1 with $\alpha = \eta \mathbf { p }$ . Since $\alpha _ { 0 } = \eta$ and $\alpha _ { k } = \eta p _ { k }$

$$
\operatorname { D i r } ( \mathbf { w } ; \eta \mathbf { p } + \mathbf { e } _ { k } ) = \frac { \eta } { \eta p _ { k } } w _ { k } \operatorname { D i r } ( \mathbf { w } ; \eta \mathbf { p } ) = \frac { w _ { k } } { p _ { k } } \operatorname { D i r } ( \mathbf { w } ; \eta \mathbf { p } ) .
$$

Multiplying both sides by $p _ { k }$ gives (23).

The content of Corollary 1 is simple but important: a categorical mixture over one-hot shifts of a Dirichlet distribution collapses back to the unshifted Dirichlet density. This is precisely the mechanism that makes the simplex relaxation exact rather than approximate.

## A.2 Full Dirichlet–categorical hierarchy

We now return to the joint factorization

$$
q ( \mathbf { x } , \mathbf { z } _ { s } , \mathbf { z } _ { t } , \mathbf { w } _ { s } , \mathbf { w } _ { t } ) = q ( \mathbf { x } ) q ( \mathbf { z } _ { s } \mid \mathbf { x } ) q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \mathbf { x } ) q ( \mathbf { z } _ { t } \mid \mathbf { z } _ { s } ) q ( \mathbf { w } _ { t } \mid \mathbf { z } _ { t } , \mathbf { x } ) ,\tag{24}
$$

together with

$$
q ( \mathbf { w } _ { t } \mid \mathbf { z } _ { t } , \mathbf { x } ) = \operatorname { D i r } ( \mathbf { w } _ { t } ; \boldsymbol { \eta } _ { t } \mathbf { p } _ { t } + \mathbf { z } _ { t } ) .\tag{25}
$$

The next proposition summarizes the full hierarchy induced by this construction. The first two statements identify the exact marginal and exact decoder at time t. The third lifts the standard discrete reverse posterior from $\mathbf { z } _ { t }$ to $\mathbf { w } _ { t }$ . The last two show that, once this lift is performed, the reverse bridge over relaxed states becomes a mixture of shifted Dirichlet components.

Proposition 4. Assume $t > 0 .$ . The Dirichlet–categorical hierarchy satisfies the following properties; statements involving ${ \bf w } _ { s }$ additionally require $s > 0$

1. The marginal distribution ofthe relaxed state is

$$
q ( \mathbf { w } _ { t } \mid \mathbf { x } ) = \operatorname { D i r } ( \mathbf { w } _ { t } ; \eta _ { t } \mathbf { p } _ { t } ) .\tag{26}
$$

2. Given $\mathbf { w } _ { t } ,$ , the variables x and $\mathbf { z } _ { t }$ are conditionally independent, and the discrete state can be recoveredfrom the relaxed state via

$$
q ( \mathbf { z } _ { t } \mid \mathbf { w } _ { t } , \mathbf { x } ) = q ( \mathbf { z } _ { t } \mid \mathbf { w } _ { t } ) = \operatorname { C a t } ( \mathbf { z } _ { t } ; \mathbf { w } _ { t } ) .\tag{27}
$$

3. For $s < t ,$ the reverse conditional posterior of $\mathbf { \dot { z } } _ { s }$ given $\mathbf { w } _ { t }$ and x is categorical:

$$
q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) = \operatorname { C a t } \left( \mathbf { z } _ { s } ; \pmb { \rho } _ { s \mid t } ( \mathbf { x } , \mathbf { w } _ { t } ) \right) ,\tag{28}
$$

where

$$
\begin{array} { r } { \rho _ { s | t } ( \mathbf { x } , \mathbf { w } _ { t } ) : = \mathbf { p } _ { s } \odot \left[ \alpha _ { t | s } \big ( \mathbf { w } _ { t } \oslash \mathbf { p } _ { t } \big ) + ( 1 - \alpha _ { t | s } ) \ \langle \mathbf { w } _ { t } , \pmb \pi \oslash \mathbf { p } _ { t } \rangle \mathbf { 1 } \right] . } \end{array}\tag{29}
$$

4. For $0 < s < t ,$ the reverse conditional posterior of $\mathbf { \dot { w } } _ { i }$ <sub>s</sub> given $\mathbf { z } _ { t }$ and x is a Dirichlet mixture:

$$
q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { t } , \mathbf { x } ) = \sum _ { k = 1 } ^ { K } r _ { s \mid t , k } ( \mathbf { x } , \mathbf { z } _ { t } ) \operatorname { D i r } ( \mathbf { w } _ { s } ; \boldsymbol { \eta } _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k } ) ,\tag{30}
$$

where $r _ { s | t , k } ( \mathbf { x } , \mathbf { z } _ { t } )$ denotes the k-th component of $\mathbf { \dot { r } } _ { s | t } ( \mathbf { x } , \mathbf { z } _ { t } )$ defined in Equation (4).

5. For $0 < s < t ,$ the reverse conditional posterior of ${ \bf w } _ { s }$ given $\mathbf { w } _ { t }$ and $\mathbf { x }$ is a Dirichlet mixture:

$$
q ( \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) = \sum _ { k = 1 } ^ { K } \rho _ { s \mid t , k } ( \mathbf { x } , \mathbf { w } _ { t } ) \operatorname { D i r } ( \mathbf { w } _ { s } ; \eta _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k } ) .\tag{31}
$$

Proof. 1. We begin with the marginal law of $\mathbf { w } _ { t }$ . Marginalizing the discrete latent $\mathbf { z } _ { t } \sim \mathrm { C a t } ( \mathbf { p } _ { t } )$ from the conditional bridge

$$
q ( \mathbf { w } _ { t } \mid \mathbf { z } _ { t } , \mathbf { x } ) = \operatorname { D i r } ( \mathbf { w } _ { t } ; \eta _ { t } \mathbf { p } _ { t } + \mathbf { z } _ { t } )
$$

gives

$$
\begin{array} { l } { { \displaystyle q ( \mathbf w _ { t } \mid \mathbf x ) = \sum _ { k = 1 } ^ { K } q ( \mathbf w _ { t } \mid \mathbf z _ { t } = \mathbf e _ { k } , \mathbf x ) q ( \mathbf z _ { t } = \mathbf e _ { k } \mid \mathbf x ) } } \\ { { \displaystyle \ = \sum _ { k = 1 } ^ { K } \mathrm { D i r } ( \mathbf w _ { t } ; \eta _ { t } \mathbf p _ { t } + \mathbf e _ { k } ) p _ { t , k } ( \mathbf x ) } . } \end{array}\tag{32}
$$

Now Corollary 1 applies termwise:

$$
p _ { t , k } ( { \bf x } ) \mathrm { D i r } ( { \bf w } _ { t } ; \eta _ { t } { \bf p } _ { t } + { \bf e } _ { k } ) = w _ { t , k } \mathrm { D i r } ( { \bf w } _ { t } ; \eta _ { t } { \bf p } _ { t } ) .
$$

Substituting this into (32) collapses the mixture:

$$
\begin{array} { l } { { \displaystyle q ( { \bf w } _ { t } \mid { \bf x } ) = \sum _ { k = 1 } ^ { K } w _ { t , k } \mathrm { D i r } ( { \bf w } _ { t } ; \eta _ { t } { \bf p } _ { t } ) } \ ~ } \\ { ~ = \left( \sum _ { k = 1 } ^ { K } w _ { t , k } \right) \mathrm { D i r } ( { \bf w } _ { t } ; \eta _ { t } { \bf p } _ { t } ) } \\ { ~ = \mathrm { D i r } ( { \bf w } _ { t } ; \eta _ { t } { \bf p } _ { t } ) , } \end{array}\tag{33}
$$

which proves (26).

2. We next derive the exact decoder from $\mathbf { w } _ { t }$ back to $\mathbf { z } _ { t }$ . Fix $k \in \{ 1 , \ldots , K \}$ . By Bayes’ rule,

$$
q ( \mathbf { z } _ { t } = \mathbf { e } _ { k } \mid \mathbf { w } _ { t } , \mathbf { x } ) = { \frac { q ( \mathbf { w } _ { t } \mid \mathbf { z } _ { t } = \mathbf { e } _ { k } , \mathbf { x } ) q ( \mathbf { z } _ { t } = \mathbf { e } _ { k } \mid \mathbf { x } ) } { q ( \mathbf { w } _ { t } \mid \mathbf { x } ) } } .\tag{34}
$$

Using the previous computation together with Corollary 1, the numerator becomes

$$
q ( \mathbf { w } _ { t } \mid \mathbf { z } _ { t } = \mathbf { e } _ { k } , \mathbf { x } ) q ( \mathbf { z } _ { t } = \mathbf { e } _ { k } \mid \mathbf { x } ) = w _ { t , k } \operatorname { D i r } ( \mathbf { w } _ { t } ; \boldsymbol { \eta } _ { t } \mathbf { p } _ { t } ) ,
$$

while the denominator is exactly $\mathrm { D i r } \big ( \mathbf { w } _ { t } ; \eta _ { t } \mathbf { p } _ { t } \big )$ . Hence

$$
q ( \mathbf { z } _ { t } = \mathbf { e } _ { k } \mid \mathbf { w } _ { t } , \mathbf { x } ) = w _ { t , k } .
$$

Since this holds for every $k ,$ we obtain

$$
q ( \mathbf { z } _ { t } \mid \mathbf { w } _ { t } , \mathbf { x } ) = \operatorname { C a t } ( \mathbf { z } _ { t } ; \mathbf { w } _ { t } ) ,
$$

and the right-hand side no longer depends on x. This proves (27).

3. We now lift the discrete reverse posterior from $\mathbf { z } _ { t }$ to $\mathbf { w } _ { t }$ . Marginalizing over $\mathbf { z } _ { t }$ gives

$$
q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) = \sum _ { j = 1 } ^ { K } q ( \mathbf { z } _ { s } \mid \mathbf { z } _ { t } = \mathbf { e } _ { j } , \mathbf { w } _ { t } , \mathbf { x } ) q ( \mathbf { z } _ { t } = \mathbf { e } _ { j } \mid \mathbf { w } _ { t } , \mathbf { x } ) .\tag{35}
$$

Conditioned on $( \mathbf { z } _ { t } , \mathbf { x } )$ , the variable $\mathbf { z } _ { s }$ is independent of $\mathbf { w } _ { t } .$ so the first factor is just the usual reverse posterior

$$
q ( \mathbf { z } _ { s } \mid \mathbf { z } _ { t } = \mathbf { e } _ { j } , \mathbf { w } _ { t } , \mathbf { x } ) = q ( \mathbf { z } _ { s } \mid \mathbf { z } _ { t } = \mathbf { e } _ { j } , \mathbf { x } ) = \operatorname { C a t } \bigl ( \mathbf { z } _ { s } ; \mathbf { r } _ { s \mid t } ( \mathbf { x } , \mathbf { e } _ { j } ) \bigr ) .
$$

The second factor is the exact decoder derived above:

$$
q ( \mathbf { z } _ { t } = \mathbf { e } _ { j } \mid \mathbf { w } _ { t } , \mathbf { x } ) = w _ { t , j } .
$$

Therefore

$$
q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) = \sum _ { j = 1 } ^ { K } w _ { t , j } { \mathrm { C a t } } { \bigl ( } \mathbf { z } _ { s } ; \mathbf { r } _ { s \mid t } ( \mathbf { x } , \mathbf { e } _ { j } ) { \bigr ) } .\tag{36}
$$

A mixture of categorical distributions is again categorical, with parameter vector equal to the same convex combination of the component parameters. Thus

$$
q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) = \operatorname { C a t } \big ( \mathbf { z } _ { s } ; \rho _ { s \mid t } ( \mathbf { x } , \mathbf { w } _ { t } ) \big ) , \qquad \rho _ { s \mid t } ( \mathbf { x } , \mathbf { w } _ { t } ) = \sum _ { j = 1 } ^ { K } w _ { t , j } \mathbf { r } _ { s \mid t } ( \mathbf { x } , \mathbf { e } _ { j } ) .
$$

To obtain the closed form, substitute Equation (4) with $\mathbf { z } _ { t } = \mathbf { e } _ { j }$

$$
\begin{array} { r } { \mathbf { r } _ { s \mid t } ( \mathbf { x } , \mathbf { e } _ { j } ) = \frac { \left[ \alpha _ { t \mid s } \mathbf { e } _ { j } + ( 1 - \alpha _ { t \mid s } ) \mathbf { \alpha } \langle \mathbf { e } _ { j } , \pi \rangle \textbf { 1 } \right] \odot \mathbf { p } _ { s } } { \langle \mathbf { e } _ { j } , \mathbf { p } _ { t } \rangle } } \\ { = \mathbf { p } _ { s } \odot \left[ \alpha _ { t \mid s } \frac { \mathbf { e } _ { j } } { p _ { t , j } ( \mathbf { x } ) } + ( 1 - \alpha _ { t \mid s } ) \frac { \alpha _ { j } } { p _ { t , j } ( \mathbf { x } ) } \mathbf { 1 } \right] . } \end{array}\tag{37}
$$

Averaging this expression under the weights $w _ { t , j }$ yields

$$
\begin{array} { l } { \displaystyle \rho _ { s | t } ( \mathbf { x } , \mathbf { w } _ { t } ) = \mathbf { p } _ { s } \odot \left[ \alpha _ { t | s } \sum _ { j = 1 } ^ { K } w _ { t , j } \frac { \mathbf { e } _ { j } } { p _ { t , j } ( \mathbf { x } ) } + ( 1 - \alpha _ { t | s } ) \sum _ { j = 1 } ^ { K } w _ { t , j } \frac { \pi _ { j } } { p _ { t , j } ( \mathbf { x } ) } \mathbf { 1 } \right] } \\ { = \mathbf { p } _ { s } \odot \left[ \alpha _ { t | s } \left( \mathbf { w } _ { t } \oslash \mathbf { p } _ { t } \right) + ( 1 - \alpha _ { t | s } ) \left. \mathbf { w } _ { t } , \pi \oslash \mathbf { p } _ { t } \right. \mathbf { 1 } \right] , } \end{array}\tag{38}
$$

which proves (28) and (29).

4. We next derive the reverse bridge over relaxed states conditioned on $\mathbf { z } _ { t }$ . Since ${ \bf w } _ { s }$ is conditionally independent of $\mathbf { z } _ { t }$ given $( { \bf z } _ { s } , { \bf x } )$ , marginalizing over $\mathbf { z } _ { s }$ gives

$$
\begin{array} { r } { q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { t } , \mathbf { x } ) = \displaystyle \sum _ { k = 1 } ^ { K } q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } = \mathbf { e } _ { k } , \mathbf { z } _ { t } , \mathbf { x } ) q ( \mathbf { z } _ { s } = \mathbf { e } _ { k } \mid \mathbf { z } _ { t } , \mathbf { x } ) } \\ { = \displaystyle \sum _ { k = 1 } ^ { K } q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } = \mathbf { e } _ { k } , \mathbf { x } ) q ( \mathbf { z } _ { s } = \mathbf { e } _ { k } \mid \mathbf { z } _ { t } , \mathbf { x } ) . } \end{array}\tag{39}
$$

The first factor is exactly the shifted Dirichlet bridge at time s:

$$
q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } = \mathbf { e } _ { k } , \mathbf { x } ) = \mathrm { D i r } ( \mathbf { w } _ { s } ; \eta _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k } ) .
$$

The second factor is the discrete reverse posterior

$$
q ( \mathbf { z } _ { s } = { \mathbf { e } } _ { k } \mid \mathbf { z } _ { t } , \mathbf { x } ) = r _ { s \mid t , k } ( \mathbf { x } , \mathbf { z } _ { t } ) .
$$

Substituting these identities into (39) yields

$$
q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { t } , \mathbf { x } ) = \sum _ { k = 1 } ^ { K } r _ { s \mid t , k } ( \mathbf { x } , \mathbf { z } _ { t } ) \operatorname { D i r } ( \mathbf { w } _ { s } ; \boldsymbol { \eta } _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k } ) ,
$$

which proves (30).

5. Finally, we lift this mixture from $\mathbf { z } _ { t }$ to $\mathbf { w } _ { t }$ . Since ${ \bf w } _ { s }$ is also conditionally independent of $\mathbf { w } _ { t }$ given $( { \bf z } _ { s } , { \bf x } )$ , we obtain

$$
\begin{array} { r } { q ( \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) = \displaystyle \sum _ { k = 1 } ^ { K } q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } = \mathbf { e } _ { k } , \mathbf { w } _ { t } , \mathbf { x } ) q ( \mathbf { z } _ { s } = \mathbf { e } _ { k } \mid \mathbf { w } _ { t } , \mathbf { x } ) } \\ { = \displaystyle \sum _ { k = 1 } ^ { K } q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } = \mathbf { e } _ { k } , \mathbf { x } ) q ( \mathbf { z } _ { s } = \mathbf { e } _ { k } \mid \mathbf { w } _ { t } , \mathbf { x } ) . } \end{array}\tag{40}
$$

The first factor is again Dir $\left( \mathbf { w } _ { s } ; \eta _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k } \right)$ , while the second factor is now the lifted reverse posterior:

$$
q ( \mathbf { z } _ { s } = \mathbf { e } _ { k } \mid \mathbf { w } _ { t } , \mathbf { x } ) = \rho _ { s \mid t , k } ( \mathbf { x } , \mathbf { w } _ { t } ) .
$$

Therefore

$$
q ( \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) = \sum _ { k = 1 } ^ { K } \rho _ { s \mid t , k } ( \mathbf { x } , \mathbf { w } _ { t } ) \operatorname { D i r } ( \mathbf { w } _ { s } ; \eta _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k } ) ,
$$

which proves (31).

Taken together, these identities show that the simplex variable sits inside the discrete diffusion process in a fully coherent way. The relaxed state has the correct Dirichlet marginal, the discrete state can be decoded from it exactly, and the reverse bridge remains available in closed form after replacing either $\mathbf { z } _ { t }$ or $\mathbf { z } _ { s }$ by their simplex-conditioned posteriors. This is the structural reason that the later training objectives and samplers can be written directly in terms of $\mathbf { w } _ { t }$ without abandoning the original categorical process.

## B Rao–Blackwellized Reverse-Bridge Objective

This appendix proves the closed form of the relaxed discrete bridge objective used in the main text. The auxiliary decoder sample $\widetilde { \mathbf { z } } _ { t }$ is marginalized analytically; the independently sampled denoiser input $\mathbf { z } _ { t }$ is not marginalized.

Proposition 5. For $0 \leq s < t \leq 1$ with $t > 0 ,$ , the relaxed discrete bridge objective satisfies

$$
\bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) = \langle \mathbf { w } _ { t } , \log \hat { \mathbf { p } } _ { t } - \log \mathbf { p } _ { t } \rangle + \left. \rho _ { s | t } ( \mathbf { x } , \mathbf { w } _ { t } ) , \log \mathbf { p } _ { s } - \log \hat { \mathbf { p } } _ { s } \right. .\tag{41}
$$

Proof. We first compute the Rao–Blackwellized categorical term $\bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } }$ . By definition,

$$
\bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } = \mathbb { E } _ { q ( \widetilde { \mathbf { z } } _ { t } | \mathbf { w } _ { t } ) } [ D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { x } ) \parallel q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ] ] .\tag{42}
$$

Using (27), we have

$$
q ( \widetilde { \mathbf { z } } _ { t } = \mathbf { e } _ { j } \mid \mathbf { w } _ { t } ) = w _ { t , j } .
$$

For fixed $\widetilde { \mathbf { z } } _ { t } = \mathbf { e } _ { j }$ , both reverse conditionals are categorical:

$$
q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } = \mathbf { e } _ { j } , \mathbf { x } ) = \operatorname { C a t } \left( \mathbf { z } _ { s } ; \mathbf { r } _ { s \mid t } ( \mathbf { x } , \mathbf { e } _ { j } ) \right) , \qquad q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } = \mathbf { e } _ { j } , \hat { \mathbf { x } } _ { \theta } ) = \operatorname { C a t } \left( \mathbf { z } _ { s } ; \mathbf { r } _ { s \mid t } ( \hat { \mathbf { x } } _ { \theta } , \mathbf { e } _ { j } ) \right) .
$$

Expanding the expectation over $\widetilde { \mathbf { z } } _ { t }$ and the KL divergence between categorical distributions gives

$$
\bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } = \sum _ { j = 1 } ^ { K } w _ { t , j } \sum _ { k = 1 } ^ { K } r _ { s | t , k } ( \mathbf { x } , \mathbf { e } _ { j } ) \log \frac { r _ { s | t , k } ( \mathbf { x } , \mathbf { e } _ { j } ) } { r _ { s | t , k } ( \hat { \mathbf { x } } _ { \theta } , \mathbf { e } _ { j } ) } .\tag{43}
$$

Substituting the explicit form of the reverse posterior components, we obtain

$$
\begin{array} { r l } & { \frac { r _ { s | t , k } ( \mathbf { x } , \mathbf { e } _ { j } ) } { r _ { s | t , k } ( \hat { \mathbf { x } } _ { \theta } , \mathbf { e } _ { j } ) } = \frac { \left[ \alpha _ { t | s } \delta _ { j , k } + ( 1 - \alpha _ { t | s } ) \pi _ { j } \right] p _ { s , k } ( \mathbf { x } ) / p _ { t , j } ( \mathbf { x } ) } { \left[ \alpha _ { t | s } \delta _ { j , k } + ( 1 - \alpha _ { t | s } ) \pi _ { j } \right] p _ { s , k } ( \hat { \mathbf { x } } _ { \theta } ) / p _ { t , j } ( \hat { \mathbf { x } } _ { \theta } ) } } \\ & { \quad \quad \quad \quad \quad \quad \quad = \frac { p _ { s , k } ( \mathbf { x } ) } { p _ { s , k } ( \hat { \mathbf { x } } _ { \theta } ) } \frac { p _ { t , j } ( \hat { \mathbf { x } } _ { \theta } ) } { p _ { t , j } ( \mathbf { x } ) } . } \end{array}\tag{44}
$$

Hence

$$
\log \frac { r _ { s | t , k } ( \mathbf { x } , \mathbf { e } _ { j } ) } { r _ { s | t , k } ( \hat { \mathbf { x } } _ { \theta } , \mathbf { e } _ { j } ) } = \log \frac { p _ { t , j } ( \hat { \mathbf { x } } _ { \theta } ) } { p _ { t , j } ( \mathbf { x } ) } + \log \frac { p _ { s , k } ( \mathbf { x } ) } { p _ { s , k } ( \hat { \mathbf { x } } _ { \theta } ) } .\tag{45}
$$

Substituting (45) into (43), the first contribution is

$$
\sum _ { j = 1 } ^ { K } w _ { t , j } \sum _ { k = 1 } ^ { K } r _ { s | t , k } ( \mathbf { x } , \mathbf { e } _ { j } ) \log \frac { p _ { t , j } ( \hat { \mathbf { x } } _ { \theta } ) } { p _ { t , j } ( \mathbf { x } ) } = \sum _ { j = 1 } ^ { K } w _ { t , j } \log \frac { p _ { t , j } ( \hat { \mathbf { x } } _ { \theta } ) } { p _ { t , j } ( \mathbf { x } ) } ,\tag{46}
$$

because $r _ { s \mid t } ( \mathbf { x } , \mathbf { e } _ { j } )$ is a probability vector. For the second contribution, exchanging the order of summation gives

$$
\sum _ { j = 1 } ^ { K } w _ { t , j } \sum _ { k = 1 } ^ { K } r _ { s | t , k } ( \mathbf { x } , \mathbf { e } _ { j } ) \log \frac { p _ { s , k } ( \mathbf { x } ) } { p _ { s , k } ( \hat { \mathbf { x } } _ { \theta } ) } = \sum _ { k = 1 } ^ { K } \left( \sum _ { j = 1 } ^ { K } w _ { t , j } r _ { s | t , k } ( \mathbf { x } , \mathbf { e } _ { j } ) \right) \log \frac { p _ { s , k } ( \mathbf { x } ) } { p _ { s , k } ( \hat { \mathbf { x } } _ { \theta } ) } .\tag{47}
$$

The coefficient in parentheses is exactly the k-th component of $q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } )$ , namely

$$
\sum _ { j = 1 } ^ { K } w _ { t , j } r _ { s | t , k } ( \mathbf { x } , \mathbf { e } _ { j } ) = \rho _ { s | t , k } ( \mathbf { x } , \mathbf { w } _ { t } ) .
$$

Therefore

$$
\begin{array} { r l r } {  { \bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } = \sum _ { j = 1 } ^ { K } w _ { t , j } \log \frac { p _ { t , j } ( \hat { \mathbf { x } } _ { \theta } ) } { p _ { t , j } ( \mathbf { x } ) } + \sum _ { k = 1 } ^ { K } \rho _ { s | t , k } ( \mathbf { x } , \mathbf { w } _ { t } ) \log \frac { p _ { s , k } ( \mathbf { x } ) } { p _ { s , k } ( \hat { \mathbf { x } } _ { \theta } ) } } } \\ & { } & { = \langle \mathbf { w } _ { t } , \log \hat { \mathbf { p } } _ { t } - \log \mathbf { p } _ { t } \rangle +  \rho _ { s | t } ( \mathbf { x } , \mathbf { w } _ { t } ) , \log \mathbf { p } _ { s } - \log \hat { \mathbf { p } } _ { s }  , } \end{array}\tag{48}
$$

which proves (41).

Equation (41) depends on the auxiliary state only through the current-time average $\langle \mathbf { w } _ { t } , \log \hat { \mathbf { p } } _ { t } \rangle$ and the lifted reverse posterior $\rho _ { s | t } ( \mathbf { x } , \mathbf { w } _ { t } )$ . This is the expression used in the main objective and in the continuous-time analysis.

## C Continuous-Time Limit of the Main Objective

The main text shows that the relaxed discrete bridge objective $\bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t )$ admits a non-degenerate first-order continuous-time limit. This appendix gives the full proof of the first-order local limit stated in the main text.

## C.1 Proof of the main continuous-time limit

Proposition 1. Up to θ-independent additive terms, the relaxed discrete bridge objective satisfies

$$
\bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; t - \Delta , t ) = \Delta \ell _ { \mathrm { c t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } , t ) + o ( \Delta ) ,\tag{49}
$$

where

$$
\begin{array} { r } { \ell _ { \mathrm { c t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } , t ) = \lambda ( t ) \left[ \left. \mathbf { w } _ { t } , \pi \mathcal { O } \hat { \mathbf { p } } _ { t } \right. - \left. \mathbf { w } _ { t } , \pi \mathcal { O } \mathbf { p } _ { t } \right. \left. \mathbf { p } _ { t } , \log \hat { \mathbf { p } } _ { t } \right. + \left. \pi \mathcal { O } \left( \mathbf { w } _ { t } \mathcal { O } \mathbf { p } _ { t } \right) , \log \hat { \mathbf { p } } _ { t } \right. \right] . } \end{array}\tag{50}
$$

and $\begin{array} { r } { \lambda ( t ) : = - \frac { d } { d t } } \end{array}$ log α(t). Consequently, the corresponding continuous-time objective is

$$
\mathcal { L } _ { \mathrm { c t } } = \int _ { 0 } ^ { 1 } \mathbb { E } _ { q ( \mathbf { x } ) } \left[ \mathbb { E } _ { q ( \mathbf { w } _ { t } \mid \mathbf { x } ) } \big [ \ell _ { \mathrm { c t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } , t ) \big ] \right] d t ,\tag{51}
$$

with $q ( \mathbf { w } _ { t } \mid \mathbf { x } ) = \operatorname { D i r } ( \mathbf { w } _ { t } ; \eta _ { t } \mathbf { p } _ { t } )$

ProofofProposition 3. Up to θ-independent additive terms, Equation (41) can be written as

$$
\begin{array} { r } { \bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) \equiv \left. \mathbf { w } _ { t } , \log \hat { \mathbf { p } } _ { t } \right. - \left. \pmb { \rho } _ { s | t } ( \mathbf { x } , \mathbf { w } _ { t } ) , \log \hat { \mathbf { p } } _ { s } \right. . } \end{array}\tag{52}
$$

We now set $s = t - \Delta$ and let $\Delta \downarrow 0 .$

First, by definition,

$$
\lambda ( t ) = - \frac { d } { d t } \log \alpha ( t ) ,\tag{53}
$$

so that

$$
\alpha _ { t | t - \Delta } = \frac { \alpha ( t ) } { \alpha ( t - \Delta ) } = 1 - \Delta \lambda ( t ) + o ( \Delta ) .\tag{54}
$$

Moreover,

$$
\partial _ { t } \mathbf { p } _ { t } = \partial _ { t } ( \alpha ( t ) \mathbf { x } + ( 1 - \alpha ( t ) ) \pmb { \pi } ) = - \lambda ( t ) \big ( \mathbf { p } _ { t } - \pmb { \pi } \big ) ,\tag{55}
$$

hence

$$
{ \bf p } _ { t - \Delta } = { \bf p } _ { t } + \Delta \lambda ( t ) ( { \bf p } _ { t } - \pi ) + o ( \Delta ) .\tag{56}
$$

Next, we expand $\rho _ { t - \Delta | t } ( \mathbf { x } , \mathbf { w } _ { t } )$ using Equation (29). Substituting Equation (54) and Equation (56) gives

$$
\begin{array} { r l } & { \rho _ { t - \Delta \mid t } ( \mathbf { x } , \mathbf { w } _ { t } ) } \\ & { \ = \mathbf { p } _ { t - \Delta } \odot \left[ \alpha _ { t \mid t - \Delta } \big ( \mathbf { w } _ { t } \oslash \mathbf { p } _ { t } \big ) + \big ( 1 - \alpha _ { t \mid t - \Delta } \big ) \left. \mathbf { w } _ { t } , \pi \oslash \mathbf { p } _ { t } \right. \mathbf { 1 } \right] } \\ & { \ = \Big ( \mathbf { p } _ { t } + \Delta \lambda ( t ) ( \mathbf { p } _ { t } - \pi ) \Big ) \odot \Big [ \mathbf { w } _ { t } \oslash \mathbf { p } _ { t } + \Delta \lambda ( t ) \Big ( \left. \mathbf { w } _ { t } , \pi \oslash \mathbf { p } _ { t } \right. \mathbf { 1 } - \mathbf { w } _ { t } \oslash \mathbf { p } _ { t } \Big ) \Big ] + o ( \Delta ) } \\ & { \ = \mathbf { w } _ { t } + \Delta \lambda ( t ) \left[ \left. \mathbf { w } _ { t } , \pi \oslash \mathbf { p } _ { t } \right. \mathbf { p } _ { t } - \pi \odot \left( \mathbf { w } _ { t } \oslash \mathbf { p } _ { t } \right) \right] + o ( \Delta ) . } \end{array}\tag{57}
$$

We now expand log $\hat { \bf p } _ { t - \Delta }$ . Since $\hat { \mathbf { x } } _ { \theta }$ is held fixed in the local limit,

$$
\begin{array} { r } { \hat { \bf p } _ { t } = \alpha ( t ) \hat { \bf x } _ { \theta } + ( 1 - \alpha ( t ) ) { \boldsymbol \pi } , } \end{array}\tag{58}
$$

and therefore

$$
\begin{array} { r } { \partial _ { t } \hat { \bf p } _ { t } = - \lambda ( t ) \big ( \hat { \bf p } _ { t } - \pi \big ) . } \end{array}\tag{59}
$$

Dividing componentwise by $\hat { \mathbf { p } } _ { t }$ yields

$$
\begin{array} { r } { \partial _ { t } \log \hat { \mathbf { p } } _ { t } = - \lambda ( t ) \left( \mathbf { 1 } - \pmb { \pi } \oslash \hat { \mathbf { p } } _ { t } \right) . } \end{array}\tag{60}
$$

Hence

$$
\log \hat { \mathbf { p } } _ { t - \Delta } = \log \hat { \mathbf { p } } _ { t } - \Delta \partial _ { t } \log \hat { \mathbf { p } } _ { t } + o ( \Delta ) = \log \hat { \mathbf { p } } _ { t } + \Delta \lambda ( t ) \left( \mathbf { 1 } - \pi \oslash \hat { \mathbf { p } } _ { t } \right) + o ( \Delta ) .\tag{61}
$$

For compactness, define

$$
A _ { t } : = \left. \mathbf { w } _ { t } , \pi \oslash \hat { \mathbf p } _ { t } \right. ,\tag{62}
$$

$$
B _ { t } : = \left. \mathbf { w } _ { t } , \pi \oslash \mathbf { p } _ { t } \right. \left. \mathbf { p } _ { t } , \log \hat { \mathbf { p } } _ { t } \right. ,\tag{63}
$$

$$
C _ { t } : = \left. \pmb { \pi } \odot ( \mathbf { w } _ { t } \oslash \mathbf { p } _ { t } ) , \log \hat { \mathbf { p } } _ { t } \right. .\tag{64}
$$

Substituting Equations (57) and (61) into Equation (52) and collecting first-order terms gives

$$
\begin{array} { r } { \bar { \mathcal { L } } _ { z _ { t - \Delta } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; t - \Delta , t ) \equiv \Delta \lambda ( t ) ( A _ { t } - B _ { t } + C _ { t } - 1 ) + o ( \Delta ) . } \end{array}\tag{65}
$$

The term $- \Delta \lambda ( t )$ is independent of θ, so it can be discarded. Therefore,

$$
\bar { \mathcal { L } } _ { z _ { t - \Delta } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; t - \Delta , t ) = \Delta \ell _ { \mathrm { c t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } , t ) + o ( \Delta ) ,\tag{66}
$$

where, up to θ-independent additive terms,

$$
\begin{array} { r } { \ell _ { \mathrm { c t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } , t ) \equiv \lambda ( t ) \left[ \left. \mathbf { w } _ { t } , \pi \mathcal { O } \hat { \mathbf { p } } _ { t } \right. - \left. \mathbf { w } _ { t } , \pi \mathcal { O } \mathbf { p } _ { t } \right. \left. \mathbf { p } _ { t } , \log \hat { \mathbf { p } } _ { t } \right. + \left. \pi \mathcal { O } \left( \mathbf { w } _ { t } \mathcal { O } \mathbf { p } _ { t } \right) , \log \hat { \mathbf { p } } _ { t } \right. \right] . } \end{array}\tag{67}
$$

This proves Equation (16) and Equation (17). Finally, integrating the density over time and taking expectation under $q ( \mathbf { x } )$ and $q ( \mathbf { w } _ { t } \mid \mathbf { x } )$ yields Equation (18). □

## D Alternative Surrogate Objectives

In the main text, we optimize the relaxed discrete bridge objective $\mathcal { \bar { L } } _ { z _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t )$ . The denoiser prediction $\hat { \mathbf { x } } _ { \theta } = f _ { \theta } ( \mathbf { z } _ { t } , t )$ uses an independently sampled categorical network input. In this appendix, $\widetilde { \mathbf { z } } _ { t }$ denotes only the auxiliary categorical decode that is averaged inside the objective. This choice is best understood relative to a broader family of surrogate objectives induced by the same simplex relaxation. The natural starting point is the exact relaxed bridge

$$
\mathcal { L } _ { w _ { s } \mid w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) : = D _ { \mathrm { K L } } [ q ( \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) \parallel q ( \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ] .\tag{68}
$$

This objective is the most direct one, but is generally intractable because $q ( \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } )$ is a Dirichlet mixture. The tractable surrogates introduced below differ in two orthogonal ways: first, whether they match only the discrete reverse state $\mathbf { z } _ { s }$ or the joint state $\left( { \bf z } _ { s } , { \bf w } _ { s } \right)$ , and second, whether they condition directly on $\mathbf { w } _ { t }$ or first decode $\widetilde { \mathbf { z } } _ { t } \sim q ( \widetilde { \mathbf { z } } _ { t } \mid \mathbf { w } _ { t } )$ and average.

## D.1 A broader surrogate family

The simplest tractable surrogate matches only the lifted reverse posterior over $\mathbf { z } _ { s } \mathrm { . }$

$$
\begin{array} { r } { \mathcal { L } _ { z _ { s } | w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) : = D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) \parallel q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ] . } \end{array}\tag{69}
$$

This objective preserves the full relaxed conditioning information in $\mathbf { w } _ { t } .$ , but discards the relaxed target ${ \bf w } _ { s }$

The objective used in the main text instead decodes $\widetilde { \mathbf { z } } _ { t }$ from $\mathbf { w } _ { t }$ and averages the standard discrete reverse KL:

$$
\bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) : = \mathbb { E } _ { q ( \widetilde { \mathbf { z } } _ { t } | \mathbf { w } _ { t } ) } [ D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { x } ) \parallel q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ] ] .\tag{70}
$$

Compared with (69), this objective is looser because $\mathbf { w } _ { t }$ influences the reverse matching step only through the decoded categorical latent $\widetilde { \mathbf { z } } _ { t }$ . Its advantage is that it remains close to the standard discrete-diffusion objective and admits the non-degenerate continuous-time limit derived in the main text.

A richer alternative is to match the joint bridge over $\left( \mathbf { z } _ { s } , \mathbf { w } _ { s } \right)$ directly under the relaxed conditioning state:

$$
\mathcal { L } _ { z _ { s } , w _ { s } | w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) : = D _ { \mathrm { K L } } \big [ q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) \big \| q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } ) \big ] .\tag{71}
$$

This objective is richer than the discrete surrogates because it also matches the relaxed bridge at time s.

Finally, one may combine the decoded conditioning of (70) with the joint target of (71):

$$
\bar { \mathcal { L } } _ { z _ { s } , w _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) : = \mathbb { E } _ { q ( \widetilde { \mathbf { z } } _ { t } | \mathbf { w } _ { t } ) } [ D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } , \mathbf { x } ) \parallel q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ] ] .\tag{72}
$$

This is the richest tractable surrogate in the family: it keeps the relaxed target w while also conditioning through the decoded latent $\widetilde { \mathbf { z } } _ { t }$

The loose joint objective admits a useful chain-rule decomposition. It shows that the main-text objective $\bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } }$ is precisely the categorical part of the loose joint bridge.

Proposition 6. The tight and loose joint objectives admit the decompositions

$$
\begin{array} { r } { \mathcal { L } _ { z _ { s } , w _ { s } | w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) = \mathcal { L } _ { z _ { s } | w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) + \bar { \mathcal { L } } _ { w _ { s } | z _ { s } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) , } \end{array}\tag{73}
$$

$$
\begin{array} { r } { \bar { \mathcal { L } } _ { z _ { s } , w _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) = \bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) + \bar { \mathcal { L } } _ { w _ { s } | z _ { s } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) , } \end{array}\tag{74}
$$

where

$$
\bar { \mathcal { L } } _ { w _ { s } | z _ { s } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) : = \mathbb { E } _ { q ( \mathbf { z } _ { s } | \mathbf { w } _ { t } , \mathbf { x } ) } [ D _ { \mathrm { K L } } [ q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \mathbf { x } ) \parallel q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \hat { \mathbf { x } } _ { \theta } ) ] ] .\tag{75}
$$

Proof. We first derive the tight decomposition (73).

By the joint graphical model, we have the conditional independences

$$
\begin{array} { r } { \mathbf { w } _ { s } \perp \perp \mathbf { w } _ { t } \mid ( \mathbf { z } _ { s } , \mathbf { x } ) , \qquad q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) = q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \mathbf { x } ) , } \end{array}
$$

and similarly on the model side,

$$
q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , { \hat { \mathbf { x } } } _ { \theta } ) = q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , { \hat { \mathbf { x } } } _ { \theta } ) q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , { \hat { \mathbf { x } } } _ { \theta } ) .
$$

Substituting these factorizations into $\mathcal { L } _ { z _ { s } , w _ { s } | w _ { t } }$ gives

$$
\mathcal { L } _ { z _ { s } , w _ { s } | w _ { t } } = D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \mathbf { x } ) \mid \mid q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } ) q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \hat { \mathbf { x } } _ { \theta } ) ] .\tag{76}
$$

Applying the chain rule of KL yields

$$
\begin{array} { r l } & { \mathcal { L } _ { z _ { s } , w _ { s } | w _ { t } } = D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) \parallel q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ] } \\ & { \qquad + \mathbb { E } _ { q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) } [ D _ { \mathrm { K L } } [ q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \mathbf { x } ) \parallel q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \hat { \mathbf { x } } _ { \theta } ) ] ] . } \end{array}\tag{77}
$$

The first term is exactly $\mathcal { L } _ { z _ { s } | w _ { t } }$ , while the second term is $\bar { \mathcal { L } } _ { w _ { s } | z _ { s } , w _ { t } }$ by definition. This proves (73).   
We next derive the loose decomposition (74).

The same graphical model implies the conditional independences

$$
\begin{array} { r } { \mathbf { z } _ { s } \perp \perp \mathbf { w } _ { t } \mid ( \widetilde { \mathbf { z } } _ { t } , \mathbf { x } ) , \qquad \mathbf { w } _ { s } \perp \perp ( \widetilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } ) \mid ( \mathbf { z } _ { s } , \mathbf { x } ) , } \end{array}
$$

and therefore

$$
q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } , \mathbf { x } ) = q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { x } ) q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \mathbf { x } ) ,\tag{78}
$$

$$
q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \mathbf { \widetilde { z } } _ { t } , \mathbf { w } _ { t } , \boldsymbol { \hat { \mathbf { x } } } _ { \theta } ) = q ( \mathbf { z } _ { s } \mid \mathbf { \widetilde { z } } _ { t } , \boldsymbol { \hat { \mathbf { x } } } _ { \theta } ) q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \boldsymbol { \hat { \mathbf { x } } } _ { \theta } ) .\tag{79}
$$

Substituting (78) and (79) into $\bar { \mathcal { L } } _ { z _ { s } , w _ { s } | z _ { t } , w _ { t } }$ gives

$$
\bar { \mathcal { L } } _ { z _ { s } , w _ { s } | z _ { t } , w _ { t } } = \mathbb { E } _ { q ( \widetilde { \mathbf { z } } _ { t } | \mathbf { w } _ { t } ) } [ D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { x } ) q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \mathbf { x } ) \parallel q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \widehat { \mathbf { x } } _ { \theta } ) q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \widehat { \mathbf { x } } _ { \theta } ) ] ] .\tag{80}
$$

Applying the chain rule of KL inside the expectation yields

$$
\begin{array} { r l } & { \bar { \mathcal { L } } _ { z _ { s } , w _ { s } | z _ { t } , w _ { t } } = \mathbb { E } _ { q ( \widetilde { \mathbf { z } } _ { t } | \mathbf { w } _ { t } ) } [ D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { x } ) \parallel q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ] ] } \\ & { \qquad + \mathbb { E } _ { q ( \widetilde { \mathbf { z } } _ { t } | \mathbf { w } _ { t } ) } \left[ \mathbb { E } _ { q ( \mathbf { z } _ { s } | \widetilde { \mathbf { z } } _ { t } , \mathbf { x } ) } [ D _ { \mathrm { K L } } [ q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \mathbf { x } ) \parallel q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \hat { \mathbf { x } } _ { \theta } ) ] ] \right] . } \end{array}\tag{81}
$$

The first term is exactly $\bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } }$ . For the second term, apply the law of total expectation:

$$
\begin{array} { r } { \mathbb { E } _ { q ( \widetilde { \mathbf { z } } _ { t } | \mathbf { w } _ { t } ) } \left[ \mathbb { E } _ { q ( \mathbf { z } _ { s } | \widetilde { \mathbf { z } } _ { t } , \mathbf { x } ) } [ [ \cdot ] ] \right] = \mathbb { E } _ { q ( \mathbf { z } _ { s } | \mathbf { w } _ { t } , \mathbf { x } ) } [ [ \cdot ] ] . } \end{array}
$$

Hence

$$
\begin{array} { r l } & { \bar { \mathcal { L } } _ { z _ { s } , w _ { s } | z _ { t } , w _ { t } } = \bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } } \\ & { \phantom { \frac { 1 } { \mathcal { L } } } + \mathbb { E } _ { q ( \mathbf { z } _ { s } | \mathbf { w } _ { t } , \mathbf { x } ) } [ D _ { \mathrm { K L } } [ q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \mathbf { x } ) \parallel q ( \mathbf { w } _ { s } \mid \mathbf { z } _ { s } , \hat { \mathbf { x } } _ { \theta } ) ] ] , } \end{array}\tag{82}
$$

which is exactly (74).

The decomposition in (74) clarifies the role of the selected main-text objective. The relaxed discrete bridge keeps the categorical part of the joint bridge while discarding the additional simplex-matching term. This is exactly the simplification that later makes the continuous-time limit non-degenerate.

## D.2 Auxiliary KL inequalities

We next record three standard KL inequalities that will be used to compare the surrogate objectives. Lemma 2 (Data processing inequality for KL divergence). Let $P$ and $Q$ be two probability distributions on a space ${ \mathcal { X } } ,$ and let $\kappa ( \mathbf { y } \mid \mathbf { x } )$ be a stochastic kernelfrom X to Y. Define the pushforward distributions

$$
( P \mathcal { K } ) ( \mathbf { y } ) : = \mathbb { E } _ { P ( \mathbf { x } ) } [ K ( \mathbf { y } \mid \mathbf { x } ) ] , \qquad ( Q \mathcal { K } ) ( \mathbf { y } ) : = \mathbb { E } _ { Q ( \mathbf { x } ) } [ K ( \mathbf { y } \mid \mathbf { x } ) ] .\tag{83}
$$

Then

$$
D _ { \mathrm { K L } } [ P \mathcal { K } \| Q \mathcal { K } ] \leq D _ { \mathrm { K L } } [ P \| Q ] .\tag{84}
$$

Proof. This follows directly from Theorem 4.1 of Kullback and Leibler [1951] by taking $T$ to be the stochastic kernel $\kappa .$ □

A particularly important special case is marginalization.

Corollary 2 (Marginalization cannot increase KL). Let $P _ { U , V }$ and $Q _ { U , V }$ be two joint distributions, with marginals $P _ { U }$ and $Q _ { U }$ . Then

$$
D _ { \mathrm { K L } } [ P _ { U } \| Q _ { U } ] \leq D _ { \mathrm { K L } } [ P _ { U , V } \| Q _ { U , V } ] .\tag{85}
$$

Proof. Take $\kappa$ to be the projection kernel $( u , v ) \mapsto$ u in Lemma 2.

Lemma 3 (Joint convexity of KL divergence). Let $\lambda _ { i } \geq 0$ with $\Sigma _ { i = 1 } ^ { m } \lambda _ { i } = 1$ , and let $P _ { i } , Q _ { i }$ be probability distributions on a common space. Then

$$
D _ { \mathrm { K L } } [ \sum _ { i = 1 } ^ { m } \lambda _ { i } P _ { i }  \sum _ { i = 1 } ^ { m } \lambda _ { i } Q _ { i } ] \leq \sum _ { i = 1 } ^ { m } \lambda _ { i } D _ { \mathrm { K L } } [ P _ { i }  Q _ { i }  .\tag{86}
$$

Proof. This follows from the joint convexity of relative entropy; see Theorem 2.7.2 in Cover and Thomas [2006]. □

## D.3 Relations among the surrogate objectives

The valid relations among the surrogate objectives follow from marginalization and joint convexity of KL divergence. Marginalizing either $\mathbf { z } _ { s }$ or ${ \bf w } _ { s }$ from a joint bridge gives the corresponding categorical or simplex bound. In addition, averaging over the auxiliary decoded state $\widetilde { \mathbf { z } } _ { t } \sim q ( \widetilde { \mathbf { z } } _ { t } \mid \mathbf { w } _ { t } )$ gives the bounds from the direct categorical and direct joint objectives to their decoded counterparts.

To state the decoded simplex bound, define

$$
\bar { \mathcal { L } } _ { w _ { s } \left. z _ { t } , w _ { t } \right. } ( { \mathbf { w } } _ { t } , \hat { \mathbf { x } } _ { \theta } , { \mathbf { x } } ; s , t ) : = \mathbb { E } _ { q ( \widetilde { \mathbf { z } } _ { t } \mid { \mathbf { w } } _ { t } ) } [ D _ { \mathrm { K L } } [ q ( { \mathbf { w } } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , { \mathbf { w } } _ { t } , { \mathbf { x } } ) \parallel q ( { \mathbf { w } } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , { \mathbf { w } } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ] ] .\tag{87}
$$

Proposition 7. For all $\mathbf { w } _ { t } \in \Delta ^ { K - 1 } , \hat { \mathbf { x } } _ { \theta } \in \Delta ^ { K - 1 }$ , and $\mathbf { x } \in \mathcal { V } ,$

$$
\mathcal { L } _ { z _ { s } | w _ { t } } \leq \mathcal { L } _ { z _ { s } , w _ { s } | w _ { t } } ,
$$

$$
\mathcal { L } _ { w _ { s } | w _ { t } } \leq \mathcal { L } _ { z _ { s } , w _ { s } | w _ { t } } ,\tag{88}
$$

$$
\mathcal { L } _ { z _ { s } , w _ { s } | w _ { t } } \leq \bar { \mathcal { L } } _ { z _ { s } , w _ { s } | z _ { t } , w _ { t } } .
$$

Moreover,

$$
\begin{array} { r l } & { \mathcal { L } _ { z _ { s } | w _ { t } } \leq \bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } \leq \bar { \mathcal { L } } _ { z _ { s } , w _ { s } | z _ { t } , w _ { t } } , } \\ & { \bar { \mathcal { L } } _ { w _ { s } | z _ { t } , w _ { t } } \leq \bar { \mathcal { L } } _ { z _ { s } , w _ { s } | z _ { t } , w _ { t } } . } \end{array}\tag{89}
$$

Proof. We first prove (88).

1. The distributions $q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } )$ and $q ( \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } )$ are the corresponding marginals of $q ( \mathbf { z } _ { s } , \mathbf { w } _ { s }$ $\mathbf { w } _ { t } , \mathbf { x } )$ . The same statement holds on the model side. Hence, by Corollary $^ { 2 , }$

$$
\begin{array} { r l } & { \mathcal { L } _ { z _ { s } | w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) \leq D _ { \mathrm { K L } } \big [ q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) \big \| q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } ) \big ] } \\ & { \qquad = \mathcal { L } _ { z _ { s } , w _ { s } | w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) , } \end{array}\tag{90}
$$

$$
\begin{array} { r l } & { \mathcal { L } _ { w _ { s } | w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) \leq D _ { \mathrm { K L } } \big [ q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) \big \| q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } ) \big ] } \\ & { \qquad = \mathcal { L } _ { z _ { s } , w _ { s } | w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) . } \end{array}\tag{91}
$$

2. By marginalizing over $\widetilde { \mathbf { z } } _ { t }$ , we have

$$
q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) = \mathbb { E } _ { q ( \widetilde { \mathbf { z } } _ { t } \mid \mathbf { w } _ { t } ) } [ q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } , \mathbf { x } ) ] ,\tag{92}
$$

$$
q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , { \hat { \mathbf { x } } } _ { \theta } ) = \mathbb { E } _ { q ( \widetilde { \mathbf { z } } _ { t } \mid \mathbf { w } _ { t } ) } [ q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } , { \hat { \mathbf { x } } } _ { \theta } ) ] .\tag{93}
$$

The mixing law is shared by both sides because it is always $q ( \widetilde { \mathbf { z } } _ { t } \mid \mathbf { w } _ { t } )$ . Applying Lemma 3 yields

$$
\begin{array} { r l } & { \mathcal { L } _ { z _ { s } , w _ { s } | w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } }_ { \theta } , \mathbf { x } ) = D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) \parallel q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ] } \\ & { \qquad \leq \mathbb { E } _ { q ( \tilde { \mathbf { z } } _ { t } | \mathbf { w } _ { t } ) } [ D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \tilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } , \mathbf { x } ) \parallel q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \tilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ] ] } \\ & { \qquad = \tilde { \mathcal { L } } _ { z _ { s } , w _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) . } \end{array}\tag{94}
$$

Combining (90), (91), and (94) proves (88).

We now prove (89).

## 1. Using the conditional independence

$$
\mathbf { z } _ { s } \perp \perp \mathbf { w } _ { t } \mid ( \widetilde { \mathbf { z } } _ { t } , \mathbf { x } ) ,
$$

and marginalizing over $\widetilde { \mathbf { z } } _ { t }$ , we have

$$
q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , \mathbf { x } ) = \mathbb { E } _ { q ( \widetilde { \mathbf { z } } _ { t } \mid \mathbf { w } _ { t } ) } [ q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { x } ) ] ,\tag{95}
$$

$$
q ( \mathbf { z } _ { s } \mid \mathbf { w } _ { t } , { \hat { \mathbf { x } } } _ { \theta } ) = \mathbb { E } _ { q ( { \widetilde { \mathbf { z } } } _ { t } \mid \mathbf { w } _ { t } ) } [ q ( \mathbf { z } _ { s } \mid { \widetilde { \mathbf { z } } } _ { t } , { \hat { \mathbf { x } } } _ { \theta } ) ] .\tag{96}
$$

Applying Lemma 3 to these mixtures yields

$$
\mathcal { L } _ { z _ { s } | w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) \leq \bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) .\tag{97}
$$

2. For each fixed $\widetilde { \mathbf { z } } _ { t } , q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { x } )$ and $q \big ( \mathbf { w } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } , \mathbf { x } \big )$ are the corresponding marginals of

$$
q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } , \mathbf { x } ) ,
$$

where

$$
q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } , \mathbf { x } ) = q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { x } ) .
$$

The same statements hold on the model side. Therefore, by Corollary 2,

$$
D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { x } ) \parallel q ( \mathbf { z } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ]
$$

$$
\begin{array} { r } { \leq D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } , \mathbf { x } ) \mid \mid q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ] , } \end{array}\tag{98}
$$

$$
D _ { \mathrm { K L } } [ q ( \mathbf { w } _ { s } \mid \mathbf { \widetilde { z } } _ { t } , \mathbf { w } _ { t } , \mathbf { x } ) \parallel q ( \mathbf { w } _ { s } \mid \mathbf { \widetilde { z } } _ { t } , \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ]
$$

$$
\begin{array} { r } { \leq D _ { \mathrm { K L } } [ q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } , \mathbf { x } ) \mid \mid q ( \mathbf { z } _ { s } , \mathbf { w } _ { s } \mid \widetilde { \mathbf { z } } _ { t } , \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } ) ] . } \end{array}\tag{99}
$$

Averaging both inequalities over $q ( \widetilde { \mathbf { z } } _ { t } \mid \mathbf { w } _ { t } )$ gives

$$
\begin{array} { r } { \bar { \mathcal { L } } _ { z _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) \leq \bar { \mathcal { L } } _ { z _ { s } , w _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) , } \end{array}\tag{100}
$$

$$
\bar { \mathcal { L } } _ { w _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) \leq \bar { \mathcal { L } } _ { z _ { s } , w _ { s } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) .\tag{101}
$$

Combining (97), (100), and (101) proves (89).

The joint surrogate contains an additional simplex-matching term. The following identity and proposition give its closed form.

Lemma 4. For any $j , k \in \{ 1 , \dots , K \}$

$$
\mathbb { E } _ { \mathrm { D i r } ( \cdot ; \eta _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k } ) } [ \log w _ { j } ] = \psi ( \eta _ { s } p _ { s , j } ( \mathbf { x } ) + \delta _ { j , k } ) - \psi ( \eta _ { s } + 1 ) .\tag{102}
$$

Proof. For a Dirichlet random vector with parameter ${ \pmb { \alpha } } = ( \alpha _ { 1 } , \dots , \alpha _ { K } )$ , the standard identity is

$$
\mathbb { E } _ { \mathrm { { D i r } } ( \cdot ; \alpha ) } [ \log w _ { j } ] = \psi ( \alpha _ { j } ) - \psi \left( \sum _ { m = 1 } ^ { K } \alpha _ { m } \right) .
$$

Applying this with

$$
\alpha = \eta _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k }
$$

gives

$$
\alpha _ { j } = \eta _ { s } p _ { s , j } ( \mathbf { x } ) + \delta _ { j , k } , \qquad \sum _ { m = 1 } ^ { K } \alpha _ { m } = \eta _ { s } \sum _ { m = 1 } ^ { K } p _ { s , m } ( \mathbf { x } ) + 1 = \eta _ { s } + 1 ,
$$

which proves (102).

Proposition 8. For $0 < s < t \leq 1$ , the simplex-matching term satisfies

$$
\begin{array} { r l } & { \bar { \mathcal { L } } _ { w _ { s } | z _ { s } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; s , t ) = D _ { \mathrm { K L } } [ \mathrm { D i r } ( \cdot ; \eta _ { s } \mathbf { p } _ { s } ) \| \mathrm { D i r } ( \cdot ; \eta _ { s } \hat { \mathbf { p } } _ { s } ) ] } \\ & { \qquad + \left. \rho _ { s | t } ( \mathbf { x } , \mathbf { w } _ { t } ) , \log \hat { \mathbf { p } } _ { s } - \log \mathbf { p } _ { s } + \mathbf { 1 } - \hat { \mathbf { p } } _ { s } \mathcal { O } \mathbf { p } _ { s } \right. . } \end{array}\tag{103}
$$

Proof. We compute the simplex term $\bar { \mathcal { L } } _ { w _ { s } | z _ { s } , w _ { t } }$ . By definition,

$$
\bar { \mathcal { L } } _ { w _ { s } | z _ { s } , w _ { t } } = \sum _ { k = 1 } ^ { K } \rho _ { s | t , k } \bigl ( \mathbf { x } , \mathbf { w } _ { t } \bigr ) D _ { \mathrm { K L } } \bigl [ \mathrm { D i r } \bigl ( \cdot ; \eta _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k } \bigr ) \bigr | \bigr ] \mathrm { D i r } \bigl ( \cdot ; \eta _ { s } \hat { \mathbf { p } } _ { s } + \mathbf { e } _ { k } \bigr ) \bigr ] .\tag{104}
$$

Fix $k \in \{ 1 , \ldots , K \}$ . By Lemma 1,

$$
\mathrm { D i r } ( \mathbf { w } ; \eta _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k } ) = \frac { w _ { k } } { p _ { s , k } ( \mathbf { x } ) } \mathrm { D i r } ( \mathbf { w } ; \eta _ { s } \mathbf { p } _ { s } ) ,\tag{105}
$$

$$
\mathrm { D i r } ( { \bf w } ; \eta _ { s }        \hat { \bf p } _ { s } + { \bf e } _ { k } ) = \frac { w _ { k } } { p _ { s , k } ( \hat { \bf x } _ { \theta } ) } \mathrm { D i r } ( { \bf w } ; \eta _ { s } \hat { \bf p } _ { s } ) .\tag{106}
$$

Taking the logarithm of the ratio gives

$$
\log \frac { \mathrm { D i r } ( \mathbf { w } ; \eta _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k } ) } { \mathrm { D i r } ( \mathbf { w } ; \eta _ { s } \hat { \mathbf { p } } _ { s } + \mathbf { e } _ { k } ) } = \log \frac { \mathrm { D i r } ( \mathbf { w } ; \eta _ { s } \mathbf { p } _ { s } ) } { \mathrm { D i r } ( \mathbf { w } ; \eta _ { s } \hat { \mathbf { p } } _ { s } ) } + \log p _ { s , k } ( \hat { \mathbf { x } } _ { \theta } ) - \log p _ { s , k } ( \mathbf { x } ) .\tag{107}
$$

We now compare the expectation of the unshifted log-ratio under the shifted Dirichlet law with the KL between the unshifted Dirichlet distributions. Expanding the Dirichlet density gives

$$
\log \frac { \mathrm { D i r } ( \mathbf { w } ; \eta _ { s } \mathbf { p } _ { s } ) } { \mathrm { D i r } ( \mathbf { w } ; \eta _ { s } \hat { \mathbf { p } } _ { s } ) } = \log \frac { B ( \eta _ { s } \hat { \mathbf { p } } _ { s } ) } { B ( \eta _ { s } \mathbf { p } _ { s } ) } + \eta _ { s } \sum _ { j = 1 } ^ { K } \bigl ( p _ { s , j } ( \mathbf { x } ) - p _ { s , j } ( \hat { \mathbf { x } } _ { \theta } ) \bigr ) \log w _ { j } .\tag{108}
$$

Taking expectation under $\mathrm { D i r } ( \cdot ; \eta _ { s } { \bf p } _ { s } + { \bf e } _ { k } )$ and applying Lemma 4 gives

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathrm { D i } ( \cdot \cdot \eta _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k } ) } \left[ \log \frac { \mathrm { D i r } ( \mathbf { w } ; \eta _ { s } \mathbf { p } _ { s } ) } { \mathrm { D i r } ( \mathbf { w } ; \eta _ { s } \hat { \mathbf { p } } _ { s } ) } \right] } \\ & { \quad \quad \quad = \log \frac { B ( \eta _ { s } \hat { \mathbf { p } } _ { s } ) } { B ( \eta _ { s } \mathbf { p } _ { s } ) } + \eta _ { s } \sum _ { i = 1 } ^ { K } ( p _ { s , j } ( \mathbf { x } ) - p _ { s , j } ( \hat { \mathbf { x } } _ { \theta } ) \big ) \big ( \psi ( \eta _ { s } p _ { s , j } ( \mathbf { x } ) + \delta _ { j , k } ) - \psi ( \eta _ { s } + 1 ) \big ) . } \end{array}\tag{109}
$$

On the other hand, the KL divergence between the unshifted Dirichlet distributions is

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } [ \mathrm { D i r } ( \cdot ; \eta _ { s } \mathbf { p } _ { s } ) \| \mathrm { D i r } ( \cdot ; \eta _ { s } \hat { \mathbf { p } } _ { s } ) ] } \\ & { \mathrm { ~ } = \log \displaystyle \frac { B ( \eta _ { s } \hat { \mathbf { p } } _ { s } ) } { B ( \eta _ { s } \mathbf { p } _ { s } ) } + \eta _ { s } \sum _ { j = 1 } ^ { K } \bigl ( p _ { s , j } ( \mathbf { x } ) - p _ { s , j } ( \hat { \mathbf { x } } _ { \theta } ) \bigr ) \left( \psi ( \eta _ { s } p _ { s , j } ( \mathbf { x } ) ) - \psi ( \eta _ { s } ) \right) . } \end{array}\tag{110}
$$

Subtracting (110) from (109), and using

$$
\psi ( a + 1 ) = \psi ( a ) + \frac { 1 } { a } , \qquad \psi ( \eta _ { s } + 1 ) = \psi ( \eta _ { s } ) + \frac { 1 } { \eta _ { s } } ,
$$

yields

$$
\begin{array} { r l } {  { \mathbb { E } _ { \mathrm { D i r } ( \cdot ; \eta _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k } ) } [ \log \frac { \mathrm { D i r } ( \mathbf { w } ; \eta _ { s } \mathbf { p } _ { s } ) } { \mathrm { D i r } ( \mathbf { w } ; \eta _ { s } \hat { \mathbf { p } } _ { s } ) } ] } } \\ & { = D _ { \mathrm { K L } } [ \mathrm { D i r } ( \cdot ; \eta _ { s } \mathbf { p } _ { s } ) \| \operatorname { D i r } ( \cdot ; \eta _ { s } \hat { \mathbf { p } } _ { s } ) ] + 1 - \frac { p _ { s , k } ( \hat { \mathbf { x } } _ { \theta } ) } { p _ { s , k } ( \mathbf { x } ) } . } \end{array}\tag{111}
$$

Combining (107) with (111), we obtain

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } [ \mathrm { D i r } ( \cdot ; \eta _ { s } \mathbf { p } _ { s } + \mathbf { e } _ { k } ) \| \mathrm { D i r } ( \cdot ; \eta _ { s } \hat { \mathbf { p } } _ { s } + \mathbf { e } _ { k } ) ] } \\ & { \ = D _ { \mathrm { K L } } [ \mathrm { D i r } ( \cdot ; \eta _ { s } \mathbf { p } _ { s } ) \| \mathrm { D i r } ( \cdot ; \eta _ { s } \hat { \mathbf { p } } _ { s } ) ] + \log p _ { s , k } ( \hat { \mathbf { x } } _ { \theta } ) - \log p _ { s , k } ( \mathbf { x } ) } \\ & { \quad + 1 - \displaystyle \frac { p _ { s , k } ( \hat { \mathbf { x } } _ { \theta } ) } { p _ { s , k } ( \mathbf { x } ) } . } \end{array}\tag{112}
$$

Substituting (112) into (104), and using $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \rho _ { s | t , k } ( \mathbf { x } , \mathbf { w } _ { t } ) = 1 } \end{array}$ , yields

$$
\begin{array} { r l } & { \bar { \mathcal L } _ { w _ { s } \mid z _ { s } , w _ { t } } = D _ { \mathrm { K L } } [ \mathrm { D i r } ( \cdot ; \eta _ { s } \mathbf p _ { s } ) \parallel \mathrm { D i r } ( \cdot ; \eta _ { s } \hat { \mathbf p } _ { s } ) ] } \\ & { \qquad + \displaystyle \sum _ { k = 1 } ^ { K } \rho _ { s | t , k } ( \mathbf x , \mathbf w _ { t } ) \left( \log p _ { s , k } ( \hat { \mathbf x } _ { \theta } ) - \log p _ { s , k } ( \mathbf x ) + 1 - \frac { p _ { s , k } ( \hat { \mathbf x } _ { \theta } ) } { p _ { s , k } ( \mathbf x ) } \right) } \\ & { \qquad = D _ { \mathrm { K L } } [ \mathrm { D i r } ( \cdot ; \eta _ { s } \mathbf p _ { s } ) \parallel \mathrm { D i r } ( \cdot ; \eta _ { s } \hat { \mathbf p } _ { s } ) ] + \left. \rho _ { s | t } ( \mathbf x , \mathbf w _ { t } ) , \log \hat { \mathbf p } _ { s } - \log \mathbf p _ { s } + \mathbf 1 - \hat { \mathbf p } _ { s } \oslash \mathbf p _ { s } \right. , } \end{array}\tag{113}
$$

which proves (103).

## D.4 Why the other surrogate objectives do not yield suitable continuous-time objectives

The relaxed discrete bridge is distinguished by its first-order scaling. We now show that the remaining surrogates behave differently in the local limit: the tight discrete objective vanishes at second order, whereas the joint objectives retain an $O ( 1 )$ simplex-matching term.

Proposition 9. Let $s = t - \Delta$ with $\Delta \downarrow 0 ;$ , and assume that $\eta _ { t }$ is continuous in $t .$

1. The tight discrete objective satisfies

$$
\begin{array} { r } { \mathcal { L } _ { z _ { t - \Delta } | w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; t - \Delta , t ) = O ( \Delta ^ { 2 } ) . } \end{array}\tag{114}
$$

2. Define

$$
\begin{array} { r } { \mathcal { S } _ { t } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) : = D _ { \mathrm { K L } } [ \mathrm { D i r } ( \cdot ; \eta _ { t } \mathbf { p } _ { t } ) \| \mathrm { D i r } ( \cdot ; \eta _ { t } \hat { \mathbf { p } } _ { t } ) ] + \langle \mathbf { w } _ { t } , \log \hat { \mathbf { p } } _ { t } - \log \mathbf { p } _ { t } + \mathbf { 1 } - \hat { \mathbf { p } } _ { t } \mathcal { O } \mathbf { p } _ { t } \rangle . } \end{array}\tag{115}
$$

Then the simplex term satisfies

$$
\bar { \mathcal { L } } _ { w _ { t - \Delta } | z _ { t - \Delta } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; t - \Delta , t ) = S _ { t } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) + o ( 1 ) .\tag{116}
$$

3. Consequently,

$$
\begin{array} { r } { \mathcal { L } _ { z _ { t - \Delta } , w _ { t - \Delta } | w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; t - \Delta , t ) = S _ { t } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) + o ( 1 ) , } \end{array}\tag{117}
$$

$$
\begin{array} { r } { \bar { \mathcal { L } } _ { z _ { t - \Delta } , w _ { t - \Delta } | z _ { t } , w _ { t } } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ; t - \Delta , t ) = { \mathcal { S } } _ { t } ( \mathbf { w } _ { t } , \hat { \mathbf { x } } _ { \theta } , \mathbf { x } ) + o ( 1 ) . } \end{array}\tag{118}
$$

Proof. 1. We first analyze the tight discrete objective. Using Equation (29) together with the same expansions (54) and (56) as above, we obtain

$$
\begin{array} { r } { \pmb { \rho } _ { t - \Delta | t } ( \mathbf { x } , \mathbf { w } _ { t } ) = \mathbf { w } _ { t } + \Delta \mathbf { a } _ { t } ( \mathbf { x } , \mathbf { w } _ { t } ) + o ( \Delta ) , } \end{array}\tag{119}
$$

$$
\begin{array} { r } { \pmb { \rho } _ { t - \Delta | t } \big ( \hat { \mathbf { x } } _ { \theta } , \mathbf { w } _ { t } \big ) = \mathbf { w } _ { t } + \Delta \hat { \mathbf { a } } _ { t } \big ( \hat { \mathbf { x } } _ { \theta } , \mathbf { w } _ { t } \big ) + o \big ( \Delta \big ) , } \end{array}\tag{120}
$$

where

$$
\mathbf { a } _ { t } ( \mathbf { x } , \mathbf { w } _ { t } ) = \lambda ( t ) \left[ \left. \mathbf { w } _ { t } , \pi \oslash \mathbf { p } _ { t } \right. \mathbf { p } _ { t } - \pi \odot \left( \mathbf { w } _ { t } \oslash \mathbf { p } _ { t } \right) \right] ,\tag{121}
$$

$$
\hat { \mathbf { a } } _ { t } \big ( \hat { \mathbf { x } } _ { \theta } , \mathbf { w } _ { t } \big ) = \lambda ( t ) \left[ \left. \mathbf { w } _ { t } , \pi \oslash \hat { \mathbf { p } } _ { t } \right. \hat { \mathbf { p } } _ { t } - \pi \odot \left( \mathbf { w } _ { t } \oslash \hat { \mathbf { p } } _ { t } \right) \right] .\tag{122}
$$

Since both vectors in (119) and (120) are probability vectors, their first-order perturbations satisfy

$$
\langle \mathbf { 1 } , \mathbf { a } _ { t } ( \mathbf { x } , \mathbf { w } _ { t } ) \rangle = 0 , \qquad \langle \mathbf { 1 } , \hat { \mathbf { a } } _ { t } ( \hat { \mathbf { x } } _ { \theta } , \mathbf { w } _ { t } ) \rangle = 0 .
$$

Therefore, expanding the categorical KL around the common base point $\mathbf { w } _ { t }$ shows that the first-order term cancels:

$$
\begin{array} { r l } & { \mathcal { L } _ { z _ { t - \Delta } | w _ { t } } = D _ { \mathrm { K L } } [ \mathrm { C a t } ( \cdot ; \mathbf { w } _ { t } + \Delta \mathbf { a } _ { t } + o ( \Delta ) ) \parallel \mathrm { C a t } ( \cdot ; \mathbf { w } _ { t } + \Delta \hat { \mathbf { a } } _ { t } + o ( \Delta ) ) ] } \\ & { \qquad = O ( \Delta ^ { 2 } ) . } \end{array}\tag{123}
$$

This proves (114).

2. We next analyze the simplex term using its closed form (103). Since $\eta _ { t }$ is continuous,

$$
\eta _ { t - \Delta } = \eta _ { t } + o ( 1 ) .
$$

Moreover, by (56) and its analogue for $\hat { \bf p } _ { t - \Delta } .$

$$
{ \bf p } _ { t - \Delta } = { \bf p } _ { t } + O ( \Delta ) , \qquad { \hat { \bf p } } _ { t - \Delta } = { \hat { \bf p } } _ { t } + O ( \Delta ) .
$$

Finally, Equation (57) gives

$$
\begin{array} { r } { \pmb { \rho } _ { t - \Delta | t } ( \mathbf { x } , \mathbf { w } _ { t } ) = \mathbf { w } _ { t } + O ( \Delta ) . } \end{array}
$$

Substituting these expansions into (103) yields

$$
\begin{array} { r l } & { \bar { \mathcal { L } } _ { w _ { t - \Delta } | z _ { t - \Delta } , w _ { t } } = D _ { \mathrm { K L } } [ \mathrm { D i r } ( \cdot ; \eta _ { t } \mathbf { p } _ { t } ) \| \mathrm { D i r } ( \cdot ; \eta _ { t } \hat { \mathbf { p } } _ { t } ) ] } \\ & { \qquad + \left. \mathbf { w } _ { t } , \log \hat { \mathbf { p } } _ { t } - \log \mathbf { p } _ { t } + \mathbf { 1 } - \hat { \mathbf { p } } _ { t } \oslash \mathbf { p } _ { t } \right. + o ( 1 ) , } \end{array}\tag{124}
$$

which is exactly (116).

3. The asymptotics of the two joint objectives now follow from the decompositions in Equations (73) and (74). For the tight joint objective,

$$
\mathcal { L } _ { z _ { t - \Delta } , w _ { t - \Delta } | w _ { t } } = \mathcal { L } _ { z _ { t - \Delta } | w _ { t } } + \bar { \mathcal { L } } _ { w _ { t - \Delta } | z _ { t - \Delta } , w _ { t } } ,
$$

and combining (114) with (116) gives (117).

For the loose joint objective,

$$
\bar { \mathcal { L } } _ { z _ { t - \Delta } , w _ { t - \Delta } | z _ { t } , w _ { t } } = \bar { \mathcal { L } } _ { z _ { t - \Delta } | z _ { t } , w _ { t } } + \bar { \mathcal { L } } _ { w _ { t - \Delta } | z _ { t - \Delta } , w _ { t } } .
$$

The first term is $o ( 1 )$ by Proposition 3, while the second is given by (116). This yields (118).

The proposition makes the selection of the main-text objective precise. The tight discrete objective is too small in the local limit: after dividing by $\Delta ,$ , it vanishes. The joint objectives behave in the opposite way: they contain a generally nonzero $O ( 1 )$ simplex-matching term, so they do not reduce to a finite first-order training density. The relaxed discrete bridge sits exactly between these two extremes, which is why it is the natural objective for the continuous-time formulation.

## E OpenWebText Experimental Details

This appendix provides preprocessing, optimization, checkpoint, sampling, and evaluation details that are omitted from the main text. The dataset, tokenizer, sequence length, backbone, primary optimization settings, and entropy-matched evaluation protocol are summarized in Sections 4.1 and 4.3.

Data preprocessing. We use the openwebtext-train and openwebtext-valid splits. Documents are concatenated with an end-of-sequence token inserted between adjacent documents and packed into fixed-length blocks of 1,024 GPT-2 tokens.

Optimization details. Models trained in our common codebase use Adam with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } =$ 0.999, numerical constant $1 0 ^ { - 8 }$ , and gradient-norm clipping at 1.0. Training uses bfloat16 precision and a linear warmup over the first 2,500 optimizer steps, followed by a constant learning rate. We maintain an EMA of the parameters with decay 0.9999 and use the averaged parameters for generation.

Simplax checkpoint. The Simplax model used in the main OpenWebText comparison is initialized from a UDLM checkpoint trained for 800,000 optimizer steps and is subsequently trained with the Simplax objective for an additional 200,000 steps. The resulting checkpoint therefore has a total optimization history of 1,000,000 steps.

The network predicts the clean-token distribution and receives the categorical state $\mathbf { z } _ { t }$ as input, while the relaxed state $\mathbf { w } _ { t }$ remains in the training objective. We use a uniform time schedule, constant Dirichlet concentration $\eta = 0 . 0 1$ , and constant loss weighting. The reported checkpoint does not use an auxiliary self-conditioning input.

Dirichlet sampling and concentration-dependent computations are performed in float64. Concentration values are restricted to $[ 1 0 ^ { - 1 0 } , 1 0 ^ { 8 } ]$ for numerical stability. Before normalization, the output logits ℓ are softly bounded as $\ell \gets 3 0 \mathrm { t a n h } \left( \ell / 3 0 \right)$ .

Table 3: Representative unconditional OpenWebText generations at NFE = 16. Entropy is the generative unigram entropy in nats per token.
<table><tr><td>Method</td><td>Ent.</td><td>Generated text</td></tr><tr><td>CANDI</td><td>5.44</td><td>July on our records and the song in July, and we&#x27;re signing people just for the next album, but it didn&#x27;t couple with laser or anything. I had just a video. I thought they were crazy, because there were a lot of people out there who did it right. They felt like it was completely under my radar. [... ]</td></tr><tr><td>UDLM</td><td>5.53</td><td>What with talk of a new bus un the altitude in the first place. In South Bend in November came the focus upon new cyclists – with an being builder. The other duck, Alabama. In total at least 82 bikes were donated to the department. [... ]</td></tr><tr><td>MDLM</td><td>5.43</td><td>Select Rally and Comm Rally was a great game with strong designed scenarios, but I did think about no-one should play it as middle of the road. You&#x27;ve already started to love my opinion on Two vs Two. You should check it, and stick and play this once again! We again stack you. This is the final year for the year. [...]</td></tr><tr><td>Duo</td><td>5.45</td><td>I ever thought that it would be torrent be progression to try something new like my team base&#x27;s football department. While I would have been optimistic about both the br of young players that hockey at the University at which went down in international football over the last of years and the way paths kept me focused so well I was pretty. [. .. ]</td></tr><tr><td>FLM</td><td>5.58</td><td>Dec 16, 2015 Edit: I cant let my reader now clear that this painting looks like a collegecommunication.&quot; On Z. Thanks the box. that by the way I am out over to the river to buy your best chance to date for such a horrible year. But at which point after year they release the windows of 3,100 by 11 inches. St. [... ]</td></tr><tr><td>LangFlow</td><td>5.42</td><td>God, will thrive. No others, and not at all, will determine our faith. We live in transient despair. But we don&#x27;t know how to do it tomorrow. We want to live. We are sinners, we do not have God. God. we have identified the freedom of the one Being, is made to the only kind that governs our nation. [... ]</td></tr><tr><td>S-FLM</td><td>5.45</td><td>I feel like, how much does that have to work out there? Are you going into a different development company? I think speaking about the timing and the reputation of what I mean as something that have really enjoyed my career doing. I kind of think the company can change things. [... ]</td></tr><tr><td>Simplax</td><td>5.46</td><td>It was basically a percentage league, is there going to have to be one a year? They always were because they had them. If they knew that rule, they got to play one for their bodies. Then, when they voted for a &quot;bolt 1&quot; rule and what they got was, they want the fifth best defender to attack you, so they added [.. .]</td></tr></table>

Qualitative generations. Tables 3 to 5 show representative generations at NFE = 16, 128, and 1, 024. For each method, the example is selected from the operating point determined by the entropymatching procedure used in the main experiment. The generated text is not manually rewritten. The excerpts are truncated at the positions marked by [. . . ], and line wrapping is applied only for presentation.

Table 4: Representative unconditional OpenWebText generations at NFE = 128. Entropy is the generative unigram entropy in nats per token.
<table><tr><td>Method</td><td>Ent.</td><td>Generated text</td></tr><tr><td>CANDI</td><td>5.43</td><td>I think they should hear about their jobs more than they think. They know that they to hear about a huge percentage of young adult Americans that are helping create jobs. I realize that what our advisors and architects believe is true. They&#x27;re in fact engaging in those American jobs, whether or not they&#x27;ve been doing it for several years. [. .. ]</td></tr><tr><td>UDLM</td><td>5.50</td><td>The show has been going on for years, and people who tell the community about the end, and what way it works are already very exciting. But it is difficult to know whether you will disagree, even whether you feel the show expects a lot. So tell other friends and friends who are just around you, the producers and the media what [...]</td></tr><tr><td>MDLM</td><td>5.45</td><td>It&#x27;s been difficult times with, for me and Brett. It&#x27;s important to tackle this issue on the long haul. I plan on putting it on hold again once I have my business, but committed to showing up again by the deadline. There&#x27;s a lot of effort involved in an issue about a loan for $20k, and it was in the middle, I [.. . ]</td></tr><tr><td>Duo</td><td>5.44</td><td>America&#x27;s social justice system. The ability of America&#x27;s people to deal with those that are alike and those different is unclear. But when tough and bad, the outcomes are not very different.&quot; Congressman Kathy Lee interviewed for this article told me what she called her priority business when serving as lieutenant senator of New York. [...]</td></tr><tr><td>FLM</td><td>5.43</td><td>I think that we would have to bring back aggression by the usual course, with war with the past, on a situation. If it was done, the fascists were inside them or there was nobody stirred up. People were part of the invasion by the Nazis, then they shot, they went on. [.. .]</td></tr><tr><td>LangFlow</td><td>5.42</td><td>The body bloates with a layer of wood, black air that brings in your skin and exites aging. When reading about your life in the late 90s, what was the puzzle you were plotting in your life? I wanted to tell the secret of being an ideal person, and that everything that relies on that secret still exists. [.. . ]</td></tr><tr><td>S-FLM</td><td>5.43</td><td>“You,&quot; asked Reeves, who had a light conversation with her. “You shouldn&#x27;t sell a media conference. You have been around for weeks and you haven&#x27;t come out to 83 game,&quot; Khanco said, &quot;WHAT?&quot; “&quot;Foreby, you have been making a fortune with free agency!&quot; she replied. “I have to wonder what it was your secret. I plan to write everything with this team. [.. . ]</td></tr><tr><td>Simplax</td><td>5.45</td><td>&quot;I thought it was odd,&quot; Anna told me. “I wanted to know,&quot; Elsa said nervously. It&#x27;s a lovely thing when you use words the way they mean something. You seem to like it. A lot.&quot; That was nice. &quot;It needed to be answered,&quot; Anna said. &quot;I just didn&#x27;t say it since I grew up.&quot; I said. [. . . ]</td></tr></table>

Table 5: Representative unconditional OpenWebText generations at NFE = 1, 024. Entropy is the generative unigram entropy in nats per token.
<table><tr><td>Method</td><td>Ent.</td><td>Generated text</td></tr><tr><td>CANDI</td><td>5.46</td><td>Tip: Every single mistake you face, you don&#x27;t pay for it all the time. This is lost confidence. Of course, in this article, you want to be booking your own grooming option, just to pay at your local time schedule. You have to have a monthly with the brand Indigo Tattoo Shop for the monthly fee. [... ]</td></tr><tr><td>UDLM</td><td>5.45</td><td>Yeah: That&#x27;s what I understand... and I&#x27;d presumably have come on board. I think it would be foolish to give something to him, but I guess he wouldn&#x27;t want to write them, because he&#x27;d actually want it to remain in front of him for a few years. [. . . ]</td></tr><tr><td>MDLM</td><td>5.44</td><td>No last of the songs have been put together – we really want the whole development going. I really hope it turns out, or the clock falls on its own itself, and if we start putting them together, things will just blow up almost completely. That&#x27;s pretty scary. This will be our first time back in the studio. [... ]</td></tr><tr><td>Duo</td><td>5.43</td><td>If for rats and pies. I&#x27;ve been digging for a hard break. With Ann – she&#x27;d have a long, bitter it by trying on. It never got there, I- She was just shy of puberty or a man. I wouldn&#x27;t even want to tell my grandmother what was going on in her life. Those things went somewhere, baby. [.. . ]</td></tr><tr><td>FLM</td><td>5.45</td><td>There can be something if you don&#x27;t know you&#x27;re someones not squad, that&#x27;s got an entirely different breed but neither do instinctively believe what or everyone born with will do the thing. There are some of that kind of good will. You only do that stuff for nothing but fun. [...]</td></tr><tr><td>LangFlow</td><td>5.41</td><td>50. &quot;My messy experience is everything&quot; he said.&quot; Something that has been on for all my life, and I think I&#x27;m living with it, but that&#x27;s tough when I start losing people&#x27;s attention.&quot; He said he had accepted divorce for most of his family&#x27;s care because he now relies on current degree&#x27;s rent, healthcare, and social assistance children. [...]</td></tr><tr><td>S-FLM</td><td>5.45</td><td>I can&#x27;t say no they. She&#x27;s definitely a stockwoman. She&#x27;s got a huge attitude. We don&#x27;t make her look her the way she&#x27;s someone that&#x27;s popular. There&#x27;s just a hint of that.&quot; If you would give us more time to the public on our #1 gender department, you can tell us your thoughts in the comments below. [... ]</td></tr><tr><td>Simplax</td><td>5.44</td><td>How could I take it to him? Would I have a chat in my life? I didn&#x27;t freak out. Even still, he did not have enough strength, and I was only millimeters away, and not even touching his chador. Only a rag with lain liquid it around his hair, and he could feel the innocence of a Playboy. [... ]</td></tr></table>

## E.1 Sudoku experimental details

Dataset. All Sudoku puzzles used in our experiments have unique solutions. Following Deschenaux and Gulcehre [2026], we use the greedy Sudoku puzzle generator of Alp [2024] to construct the training data and the evaluation sets with 25 or more clues. The training set contains 48,000 puzzles with 30 observed cells, and all models are trained exclusively on this 30-clue training set. For evaluation, we use 2,000 puzzles for each clue count. The training set and the 40-, 35-, 30-, and 25-clue evaluation sets are generated with seed 42.

For the 20- and 17-clue settings, we instead use uniquely solvable 17-clue Sudoku puzzles studied by Lin et al. [2013]. The 17-clue evaluation set uses these puzzles directly, while the 20-clue evaluation set is constructed by augmenting each 17-clue puzzle with three additional clues from its unique solution.

At evaluation time, we therefore consider conditional completion with 40, 35, 30, 25, 20, and 17 clues. The 30-clue setting matches the training clue density. The 40- and 35-clue settings provide more conditioning information than observed during training, whereas the 25-, 20-, and 17-clue settings progressively reduce the available conditioning information. For the no-clue evaluation, all 81 cells in the puzzle prefix are replaced with the blank token.

Sequence representation. The vocabulary contains 12 symbols: a blank-cell token, the digits 1–9, a row-separator token, and a BOS token. Each example is represented by 180 tokens:

$$
[ \mathrm { B } 0 \mathrm { S } ] \ + \ 8 9 \mathrm { - t o k e n \ p u z z l e \ + \ } [ \mathrm { B } 0 \mathrm { S } ] \ + \ 8 9 \mathrm { - t o k e n \ s o l u t i o n } .
$$

Each 89-token board representation contains 81 cell tokens and eight row separators. The resulting puzzle prefix has length 91, and the generated solution has length 89. Unobserved cells are represented explicitly by the blank token, so the prefix length remains 91 for every clue count. The training objective is evaluated only on the solution portion of the sequence.

Shared architecture. All methods use Transformer models with hidden dimension 512, eight Transformer blocks, eight attention heads of dimension 64, and dropout probability 0.1. The models use learned token embeddings without embedding–output weight tying. The autoregressive model uses causal attention and no time conditioning. The remaining models use bidirectional attention and AdaLN-based time conditioning with conditioning dimension 128.

<table><tr><td rowspan=1 colspan=9>Input puzzle        25 clues</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5</td></tr></table>

<table><tr><td rowspan=1 colspan=9>Ground truth      reference</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>7</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>5</td></tr></table>

<table><tr><td rowspan=1 colspan=9>AR           invalid 42errors</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>7</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>5</td></tr></table>

<table><tr><td rowspan=1 colspan=9>MDLM        invalid ·8errors</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>7</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>5</td></tr></table>

![](images/65ca3fa121847204e31ad8b1a212c3a91aefb9d5ca469d62fb925be69024d5f7.jpg)

<table><tr><td rowspan=1 colspan=9>FLM         invalid26errors</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>7</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>5</td></tr></table>

Figure 5: Generation results for a representative 25-clue Sudoku puzzle at NFE = 89. Blue entries are given clues, black entries agree with the unique solution, and red entries differ from it. All methods receive the same puzzle prefix. Simplax produces the valid solution in this example, whereas the other methods violate at least one Sudoku constraint.

Optimization. All models are trained for 20,000 optimization steps with global batch size 256 using bfloat16 precision. We use Adam with learning rate $3 \times 1 0 ^ { - 4 } , \beta _ { 1 } = \mathsf { \bar { 0 } } . 9 , \beta _ { 2 } = 0 . 9 9 9$ , and gradient-norm clipping at 1.0. The learning rate is linearly warmed up for 2,500 steps and then held constant. We maintain an EMA of the parameters with decay 0.9999 and use the averaged parameters for evaluation. Antithetic time sampling is enabled. All training runs use random seed 1, and the reported checkpoints are taken at step 20,000, corresponding to epoch 106.

Qualitative Sudoku generation. As shown in Figure 5, the baseline models often recover many individual entries while failing to form a globally consistent grid. MDLM comes closest in this case, differing from the unique solution in only eight cells, but even these sparse errors are sufficient to invalidate the board. In contrast, Simplax satisfies the coupled row, column, and subgrid constraints simultaneously. This example was selected from the held-out 25-clue evaluation set to illustrate the distinction between local agreement and global validity; aggregate accuracy over the full evaluation sets is reported separately.