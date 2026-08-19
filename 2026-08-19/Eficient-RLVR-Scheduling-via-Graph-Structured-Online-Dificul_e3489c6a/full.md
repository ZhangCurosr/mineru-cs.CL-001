# Eficient RLVR Scheduling via Graph-Structured Online Dificulty Estimation

Zhizhao Liu, Zhiliang Tian, Xi Wang, Zhihua Wen, Yihang Xiong, Zhiquan Lai, Dongsheng Li,

<sup>1</sup>PDL Lab, College of Computer Science and Technology, National University of Defense Technology

No.109 Deya Road, Kaifu District, Changsha

Hunan 410073, P.R. China

tianzhiliang@nudt.edu.cn

dsli@nudt.edu.cn

## Abstract

Reinforcement learning with verifiable rewards (RLVR) improves the reasoning capabilities of large language models but relies on costly rollout exploration. Assigning the same exploration budget to samples with diferent dificulty levels is inefficient: easy samples may receive redundant rollouts, whereas dificult but learnable samples may receive too little exploration. Existing adaptive schedulers address this mismatch through curriculum-based sample selection or non-uniform rollout allocation based on estimated sample dificulty. However, obtaining reliable online dificulty estimates remains challenging: dedicated probing adds substantial generation overhead, whereas history-based estimators face a cold start with no initial observations and stale feedback, and typically ignore relations among samples. To address these limitations, we propose a plug-and-play graph-based online dificulty estimator that shares rollout feedback across related samples and continuously updates their dificulty estimates, mitigating cold start and staleness without dedicated probing. Specifically, we first construct a dificulty-aware sample graph based on semantic and reasoning similarities. Based on this graph, we introduce latent dificulty states and use a Potts prior to encourage neighboring samples to share the same state. We then employ a state-level Beta-Binomial model to aggregate the rollout outcomes associated with each state. Finally, we use an online mean-field variational algorithm to continuously update the latent-state assignments and state-level dificulty as new feedback arrives. Our framework can be integrated into sample-selection and rollout-allocation schedulers, enabling dificulty-adaptive exploration without dedicated probing. Experiments across multiple base models, RL schedulers, and benchmarks demonstrate that our framework achieves better performance under matched rollout budgets.

## Introduction

Reinforcement learning with verifiable rewards (RLVR) improves the reasoning capabilities of large language models through automatically verifiable feedback (Shao et al. 2024; Yu et al. 2026; Zheng et al. 2025). Group-based algorithms, such as Group Relative Policy Optimization (GRPO), generate multiple responses for each training sample and compare their rewards to update the policy (Shao et al. 2024). These rollouts explore alternative solutions but also incur substantial generation cost (Guo et al. 2025; Yao et al. 2026).

However, GRPO commonly assigns the same number of rollouts to every selected sample (Shao et al. 2024; Yao et al. 2026). This uniform allocation treats all samples as if they require the same amount of exploration, although the current policy may easily solve some samples and struggle with others. Additional rollouts on easy samples often repeat predictable outcomes, whereas dificult but learnable samples may receive too few attempts to discover useful solution trajectories (Nguyen et al. 2026a; Yao et al. 2026; Zeng et al. 2026). This budget mismatch motivates dificultyaware scheduling, which uses sample dificulty to decide how to distribute the available exploration budget.

Existing studies implement dificulty-aware scheduling in two ways. First, sample-selection methods adjust the training distribution and decide which samples receive exploration (Gao et al. 2026; Qu et al. 2026; Kong et al. 2026). Second, rollout-allocation methods assign diferent numbers of rollouts to selected samples and determine how much exploration each sample receives (Nguyen et al. 2026a; Yao et al. 2026; Zeng et al. 2026). Although these methods make diferent scheduling decisions, both need up-to-date dificulty estimates because the policy continually changes which samples it can solve. This requirement makes low-cost online dificulty estimation essential for adaptive RLVR scheduling.

However, obtaining such up-to-date and reliable estimates remains challenging. Some researchers explicitly estimate success probabilities for train examples under the current policy by using additional rollout probing (Zeng et al. 2026; Yao et al. 2026; Team et al. 2025). Although this approach provides estimation, it incurs substantial computational overhead and introduces statistical uncertainty due to the limited number of stochastic rollouts<sup>1</sup>. Further, some researchers estimate these success probabilities from outcomes collected during previous training steps (Nguyen et al. 2026a,b; Kong et al. 2026; Qu et al. 2026). Although this approach reduces the additional rollout overhead, it struggles with cold start and accuracy because historical results may introduce noise from the variance of stochastic outcomes, and the historical results may become outdated as the policy model evolves during training. Thus, these limitations motivate the following question: How can we continuously estimate the evolving dificulty of all training examples at low cost as the policy evolves during RLVR training?

We argue that semantic or reasoning structure related samples can provide mutually informative evidence for dificulty estimation. Samples with similar semantic often rely on similar model capabilities and may exhibit exploitable local statistical correlations. Thus, in this paper, we propose a graph-based online dificulty estimation framework for RLVR-based LLM training, which can be integrated into diferent RL training schedulers to improve their performance. Specifically, we first construct a dificulty-aware sample graph that connects samples with similar semantic and reasoning characteristics. Then, we model sample dificulty as graph-structured latent states and develop an online variational inference algorithm to update these states using rollout feedback collected during training. In this way, related samples can share historical evidence, enabling us to continuously estimate the success probability of all training examples without additional rollout probing. Finally, we integrate our estimator into sample selection or rollout allocation methods to improve RLVR training eficiency. Experiments across diferent models and datasets demonstrate the accuracy, eficiency, and generality of our method.

Our contributions are as follows: (1) We are the first to formulate dynamic dificulty estimation in RLVR as a graphstructured latent-variable inference problem. (2) We propose a low-cost online estimator that propagates rollout feedback among related samples and continuously estimates their success probabilities as the policy evolves. (3) We integrate our estimator into both sample selection and rollout allocation methods. Extensive experiments show that our method is cold start friendly and substantially reduces dificulty estimation overhead while consistently improving the downstream performance of RLVR schedulers.

## Related Work

## Adaptive Sampling and Rollout Allocation in RLVR

Curriculum learning order or reweight examples according to dificulty, competence, or learnability (Bengio et al. 2009; Mindermann et al. 2022). This principle is especially relevant to group-based RLVR, where prompts with nearly deterministic group rewards provide little group-relative learning signal (Shao et al. 2024; Yu et al. 2026; Zheng et al. 2025). Prompt-selection methods therefore use estimated dificulty, competence alignment, uncertainty, or reward dynamics to prioritize informative examples (Yu et al. 2026; Gao et al. 2026; Qu et al. 2026; Kong et al. 2026; Zheng et al. 2026; Bae et al. 2026). A complementary line allocates non-uniform rollout counts. GVM, VIP, and CurES use gradient-variance or stability objectives (Yao et al. 2026; Nguyen et al. 2026a; Zeng et al. 2026), whereas recent methods use sequential uncertainty reduction, Bernoulli-variance proxies, or posterior hit utility (Xiong et al. 2025; Fang et al. 2026). Although their scheduling objectives difer, these methods require estimates of current prompt behavior. Our work is complementary: it provides a reusable estimator rather than another scheduling rule.

## Dynamic Dificulty Estimation for Reasoning LLMs

Dificulty depends not only on an example but also on the policy model. Item response theory explicitly models the interaction between latent ability and item dificulty and has been adapted to NLP evaluation (Lalor, Wu, and Yu 2016; Rodriguez et al. 2021). Training-dynamics methods instead characterize examples through quantities such as confidence and variability across epochs, or prioritize examples that remain learnable and useful (Swayamdipta et al. 2020; Mindermann et al. 2022). RLVR further requires these modeldependent estimates to track an evolving policy. Existing RLVR estimators follow three broad strategies. Model-based methods learn a value or scoring model to predict dificulty without rolling out every candidate (Gao et al. 2026). Probing-based methods estimate current success probabilities from additional on-policy generations, obtaining timely but costly and noisy signals (Yao et al. 2026; Zeng et al. 2026; Team et al. 2025). History-based methods reuse training feedback: MoPPS performs streaming Bayesian inference (Qu et al. 2026); CDAS aggregates historical performance discrepancies (Kong et al. 2026); GRESO exploits temporal regularity in rewards (Zheng et al. 2026); and VIP fits a lightweight predictor to recent rollouts (Nguyen et al. 2026a). Complementary scheduling methods operate at different granularities: Re-Schedule derives a static structurallearnability score from ofline reasoning trees for easy-tohard curriculum scheduling (Wang et al. 2026), whereas ARRoL predicts final correctness from partial rollouts and prunes trajectories online to balance binary rewards (Xu et al. 2026). Learned predictors must remain calibrated under policy drift, probing and tree construction incur generation overhead, and historical observations may be sparse or stale. Our approach also draws on graph-based semi-supervised learning and probabilistic graphical models, using spectral structure, Potts priors, and mean-field inference to propagate sparse feedback across related samples; see Appendix H for further discussion. In contrast, our estimator couples related samples through an explicit graph, pools sparse evidence through shared latent dificulty states.

Our setting combines these ideas with policy-induced nonstationarity: rollout observations are sparse, while their generating policy changes throughout training. We construct a dificulty-aware graph, associate prompts with time-indexed latent states, aggregate binary outcomes through state-level Beta–Binomial factors, and carry the variational posterior across training steps. This yields sequential probabilistic reasoning over relational and evolving latent states, together with the success-probability estimates required by adaptive RLVR schedulers.

## Method

## 3.1 Overview

We propose a graph-based online framework for estimating the evolving dificulty of training examples in RLVR. By leveraging historical rollout outcomes from related examples, our framework predicts the success probability and difficulty of each example under the current policy before each training step. As illustrated in Fig. 1, the framework consists of four sequential modules. First, the Dificulty-aware Sample Graph jointly encodes the semantic and dificulty features of training examples and constructs a sparse train set graph (Sec. 3.3). Second, the Graph-Structured Latent Dificulty Model maps the graph into latent dificulty states that can be dynamically updated by rollouts feedback, with examples in the same state sharing a common success probability (Sec. 3.4). Third, the Online Mean-Field Variational Inference updates the latent-state distribution of each example and the posterior success probability of each latent state (Sec. 3.5). Finally, we use the updated posterior to estimate the success probability and dificulty of candidate samples for the next training step (Sec. 3.6).

![](images/443bd165a24c39472766837f13631eb94c4d92cda202e12859944f262bc97059.jpg)  
Figure 1: The overview of our framework. We first construct a dificulty-aware sample graph (a). During RL training, our graph structured latent model aggregates rollout feedback across related samples (b), while online mean-field variational inference updates their evolving dificulty (c). The resulting estimates guide adaptive sample selection and rollout allocation (d).

## 3.2 Problem Formulation

Let the training dataset for reinforcement learning be $\mathcal { D } =$ $\{ x _ { i } \} _ { i = 1 } ^ { N }$ , where $x _ { i }$ denotes the i-th sample. Following the actual training order, we let ${ \mathcal { O } } _ { t }$ represent the set of samples for which we generate rollouts at training step t. Under the current policy model $\pi _ { t } ,$ , each sample $i \in \mathcal { O } _ { t }$ generates $n _ { i , t }$ rollouts, and we denote the binary correctness outcome of the g-th rollout by $y _ { i , t , g } \in \{ 0 , 1 \}$ . We aggregate the rollout outcomes of the same sample into the number of successful rollouts, $\begin{array} { r } { s _ { i , t } = \sum _ { g = 1 } ^ { n _ { i , t } } y _ { i , t , g } } \end{array}$ , and define its empirical success rate as $r _ { i , t } = s _ { i , t } / n _ { i , t }$

Our goal is to accurately estimate the probability that the current policy correctly solves each sample before each RL training step: $\hat { p } _ { i , t } = \operatorname* { P r } _ { Y \sim \pi _ { t } } ( Y$ is correct $\mid x _ { i } , \mathbf { s } _ { < t } , \mathbf { n } _ { < t } )$ We then define the dificulty of sample i at step t as $d _ { i , t } = 1 -$ $p _ { i , t }$ . Finally, we feed these probability or dificulty estimates into existing RL scheduling frameworks to guide decisions such as sample selection and rollout allocation.

## 3.3 Dificulty-Aware Sample Graph

In this section, we construct a sparse similarity graph over the training dataset based on semantic and dificulty. This allows samples with few or no prior rollouts to estimate their current expected dificulty by leveraging historical feedback from semantically and dificulty-wise similar neighbors.

To encourage the encoder to capture semantic content, reasoning structure, and dificulty-related characteristics of each sample, we introduce a dificulty-aware instruction $I _ { \pi } { \mathrm { : } }$ Instruction: Given a prompt, encode it into a dense vector that captures both its semantic content and dificulty level,for the purpose of similarity comparison. \n Prompt: [SAMPLE] Let $f _ { \mathrm { e m b } }$ denote a pretrained embedding model. The representation of sample $x _ { i }$ is computed as $\widetilde { \phi } _ { i } = f _ { \mathrm { e m b } } ( I _ { \pi } , x _ { i } )$ and normalized to $\phi _ { i } = \widetilde { \phi } _ { i } / \lVert \widetilde { \phi } _ { i } \rVert _ { 2 }$ . For each pair of samples, we measure their similarity with cosine similarity: $c _ { i j } = \phi _ { i } ^ { \top } \phi _ { i }$

To capture the underlying manifold structure of highdimensional sample embeddings while reducing computational cost, we construct a k-nearest-neighbor graph by retaining the $k _ { \mathrm { n n } }$ nearest neighbors ofeach node. We then retain only edges where the two nodes are mutual k-nearest neighbors and remove those with negative similarities, yielding a sparse undirected sample graph $W \in \mathbb { R } ^ { N \times N }$ . Specifically, $W _ { i j } = c _ { i j }$ if $x _ { i }$ and $x _ { j }$ are mutual k-nearest neighbors and $c _ { i j } > 0$ , and 0 otherwise. A larger $W _ { i j }$ reflects greater similarity in semantic content, reasoning structure, and dificultyrelated features between $x _ { i }$ and $x _ { j }$

## 3.4 Graph-Structured Latent Dificulty Model

In this section, we aim to perform statistical inference of sample dificulty by leveraging the graph structure and the historical rollout outcomes collected during previous RL training. Specifically, we first construct a graph-structured prior over sample latent states, then model rollout feedback through shared state-level success probabilities, and finally derive the posterior distribution for subsequent dificulty estimation.

Graph Prior. Although W captures local similarity relationships among samples, it falls short in describing the overall dificulty structure of the training dataset. To further uncover similarities in both semantic features and dificulty levels across samples, we apply spectral clustering to the sparse undirected sample graph $\mathbf { \bar { \boldsymbol { W } } } ^ { \bullet } \in \mathbb { R } ^ { N \times N }$ , and it yields an initial set of static cluster labels $c _ { i } \in \{ 1 , \ldots , K \}$ However, spectral clustering relies solely on static sample representations, and its assignments can be afected by representation and clustering errors. More importantly, as we continuously update the policy during RL training, the dificulty of the same sample under the current policy may also change. Therefore, we convert each static cluster label into a smoothed prior distribution over latent states, we define $A _ { i k } \ = \ \left\{ \begin{array} { l l } { { 1 - \epsilon , } } & { { k = c _ { i } , } } \\ { { { \frac { \epsilon } { K - 1 } } , } } & { { k \neq c _ { i } , } } \end{array} \right.$ . This conversion retains the structure from spectral clustering, and it also allows subsequent rollout outcomes to refine each sample’s latent assignment. Then, inspired by Besag (1974), we introduce a Potts graph prior (Potts 1952) over the latent assignments to retain the initial clustering structure while encouraging neighboring samples to share the same latent state. Specifically, we define $p ( \mathbf { z } _ { t } \mid W , A ) =$ $\begin{array} { r } { \frac { 1 } { Z ( W , A ) } \left( \prod _ { i = 1 } ^ { N } A _ { i , z _ { i , t } } \right) \exp \left[ \frac { \beta } { 2 } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } W _ { i j } \mathbb { I } ( z _ { i , t } = z _ { j , t } ) \right] } \end{array}$ where $\mathbf { z } _ { t } = ( z _ { 1 , t } , \dots , z _ { N , t } )$ denotes the vector of latent-state assignments at step t, with $z _ { i , t } \in \{ 1 , \dots , K \}$ indicating the assignment for sample $x _ { i }$ . The hyperparameter $\beta$ controls the strength of graph propagation, and $Z ( W , A )$ is the partition function.

State-Level Success Model. To statistically infer sample dificulty and latent-state assignments, we further integrate their relationship with historical rollout feedback into a Bayesian framework. However, estimating an independent success probability for each sample would still require a large number of sample-level rollouts to yield stable estimates. To address this and improve the eficiency of historical feedback utilization, we assume that samples assigned to the same latent state share a common state-level success probability. For each latent state k, following Zeng et al. (2026) and Qu et al. (2026), we model its success probability at step t with a Beta distribution: $\theta _ { k , t } \sim \mathrm { B e t a } \left( a _ { k , t } ^ { \mathrm { h i s t } } , b _ { k , t } ^ { \mathrm { h i s \tilde { t } } } \right)$ . This distribution jointly captures the estimate of the state-level success probability while enabling sequential Bayesian updates. We set the prior parameters for step t by inheriting the posterior parameters from step $t - 1 , \mathrm { i . e . , } a _ { k , t } ^ { \mathrm { h i s t } } = a _ { k , t - 1 }$ and $b _ { k , t } ^ { \mathrm { h i s t } } ~ = ~ b _ { k , t - 1 }$ . Both parameters are initialized to 1: $a _ { k , 0 } ~ = ~ 1 , ~ b _ { k , 0 } ~ = ~ 1$ . Conditioned on the latent-state assignment $z _ { i , t } ~ = ~ k$ and the conditional independence of rollout outcomes, the number of successes $s _ { i , t }$ out of $n _ { i , t }$ rollouts follows a binomial distribution: $( s _ { i , t } \mid n _ { i , t } , z _ { i , t } =$ $k , \theta _ { k , t } ) \sim \mathrm { B i n o m i a l } ( n _ { i , t } , \theta _ { k , t } )$ . Let $\pmb { \theta } _ { t } = ( \theta _ { 1 , t } , \dots , \theta _ { K , t } )$ collect the success probabilities of all latent states. By combining the graph-structured prior over $\mathbf { z } _ { t }$ , the independent Beta priors over each $\theta _ { k , t }$ , and the sample-level binomial likelihoods, we obtain the joint distribution at step t: $\begin{array} { r } { p _ { t } ( \mathbf { s } _ { t } , \mathbf { z } _ { t } , \boldsymbol { \theta } _ { t } \mid \mathbf { n } _ { t } , W , A ) = ~ p ( \mathbf { z } _ { t } \mid W , A ) \prod _ { k = 1 } ^ { K } } \end{array}$ Beta θ<sub>k,t</sub> | $a _ { k , t } ^ { \mathrm { h i s t } } , b _ { k , t } ^ { \mathrm { h i s t } } ) \prod _ { i \in \mathcal { O } _ { t } }$ Binomial $\left( s _ { i , t } \mid n _ { i , t } , \theta _ { z _ { i , t } , t } \right)$

Posterior Inference. From this joint distribution, we apply Bayes’ rule to infer the posterior over the latent-state assignments and state-level success probabilities after observing the rollout outcomes. This posterior allows us to jointly update our beliefs about each sample’s latent state and the shared success probabilities, which is essential for subsequent dificulty estimation. It factorizes as $p _ { t } ( \mathbf { z } _ { t } , \pmb \theta _ { t } \mid \mathbf { s } _ { t } , \mathbf { n } _ { t } , W , A ) =$ $\begin{array} { r l } { \frac { p _ { t } ( \mathbf { s } _ { t } , \mathbf { z } _ { t } , \theta _ { t } | \mathbf { n } _ { t } , W , A ) } { \mathbf { \sigma } } = \frac { \mathbf { \sigma } _ { p _ { t } } ( \mathbf { s } _ { t } , \mathbf { z } _ { t } , \theta _ { t } | \mathbf { n } _ { t } , W , A ) } { \mathbf { \sigma } } } \end{array}$ $\begin{array} { r } { \overline { { p _ { t } ( \mathbf { s } _ { t } | \mathbf { n } _ { t } , W , A ) \mathrm { ~ \tiny ~ - ~ } } } \overline { { \sum _ { \mathbf { z } _ { t } } \int p _ { t } ( \mathbf { s } _ { t } , \mathbf { z } _ { t } , \boldsymbol { \theta } _ { t } | \mathbf { n } _ { t } , W , A ) \mathrm { d } \boldsymbol { \theta } _ { t } } } . } \end{array}$

## 3.5 Online Mean-Field Variational Inference

In this section, we employ online mean-field variational inference to support sample dificulty estimation for adaptive LLM RL scheduling. At each training step, only selected samples provide rollout feedback. The sample graph allows this feedback to update the latent-state distributions of related samples. Specifically, we approximate the joint posterior of the latent states $\mathbf { z } _ { t }$ and state-level success probabilities $\pmb \theta _ { t } .$ However, exact posterior inference is intractable. The Potts prior couples the latent states of neighboring samples. This coupling requires a summation over all $K ^ { \widetilde { N } }$ possible assignments. Its computational cost grows exponentially with the number of training samples. To obtain tractable updates, we follow prior work (Zhang 1992; Hofman et al. 2013) and adopt the mean-field approximation $\begin{array} { r } { q _ { t } ( \mathbf { z } _ { t } , \pmb { \theta } _ { t } ) \approx \prod _ { i = 1 } ^ { N } q _ { i } ( z _ { i , t } ) \prod _ { k = 1 } ^ { K } q ( \theta _ { k , t } ) } \end{array}$ .Here, $q _ { i } \big ( z _ { i , t } \big )$ represents the latent-state distribution of sample i. We denote its state-assignment probability by $q _ { i k , t } \equiv q _ { i } ( z _ { i , t } = k )$ . We restrict each state-level factor to the Beta family: $q ( \theta _ { k , t } ) =$ $\mathrm { B e t a } ( a _ { k , t } , b _ { k , t } )$

Let Q denote the variational family defined above. We seek the member of Q that minimizes the Kullback– Leibler (KL) divergence from the exact posterior: $q _ { t } ^ { \star } \ =$ arg $\begin{array} { r } { \operatorname* { m i n } _ { q _ { t } \in \mathcal { Q } } { \mathrm { K L } } \left[ q _ { t } ( \mathbf { z } _ { t } , \pmb { \theta } _ { t } ) \parallel , p _ { t } ( \mathbf { z } _ { t } , \pmb { \theta } _ { t } \mid \mathbf { s } _ { t } , \mathbf { n } _ { t } , W , A ) \right] } \end{array}$ The exact posterior contains the partition function $Z ( W , A )$ of the Potts prior. This makes the KL divergence dificult to evaluate directly. We instead use the evidence lower bound (ELBO): $\begin{array} { r l } { \mathcal { L } _ { t } ( q _ { t } ) } & { { } = } \end{array}$ $\mathbb { E } _ { q _ { t } } [ \log p _ { t } ( \mathbf { s } _ { t } , \mathbf { z } _ { t } , \boldsymbol { \theta } _ { t } \mid \mathbf { n } _ { t } , W , A ) ] - \mathbb { E } _ { q _ { t } } [ \log q _ { t } ( \mathbf { z } _ { t } , \boldsymbol { \theta } _ { t } ) ]$ . The log marginal likelihood satisfies log $\begin{array} { r } { { p } _ { t } ( \mathbf { s } _ { t } \mid \mathbf { n } _ { t } , W , A ) = } \end{array}$ $\mathcal { L } _ { t } ( q _ { t } ) \ + \ \mathrm { K L } \left[ q _ { t } ( \mathbf { z } _ { t } , \pmb { \theta } _ { t } ) \parallel , p _ { t } ( \mathbf { z } _ { t } , \pmb { \theta } _ { t } \mid \mathbf { s } _ { t } , \mathbf { n } _ { t } , W , A ) \right]$ The left-hand side does not depend on $q _ { t } .$ . Thus, minimizing the KL divergence is equivalent to maximizing the ELBO: $q _ { t } ^ { \star } = \arg \operatorname* { m a x } _ { q _ { t } \in \mathcal { Q } } \mathcal { L } _ { t } ( q _ { t } )$

We can evaluate the ELBO under the mean-field approximation. In particular, the expected Potts interaction becomes $\begin{array} { r } { \mathbb { E } _ { q _ { t } } \left[ \mathbb { I } ( z _ { i , t } = z _ { j , t } ) \right] = \sum _ { k = 1 } ^ { K } q _ { i k , t } q _ { j k , t } . } \end{array}$ . Substituting this result and the variational factors into the ELBO gives

$$
\begin{array} { r l } & { \mathcal { L } _ { t } ( q ) = ( \beta / 2 ) \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } W _ { i j } \sum _ { k = 1 } ^ { K } q _ { i k , t } q _ { j k , t } } \\ & { \qquad + \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } q _ { i k , t } \log A _ { i k } - \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } q _ { i k , t } \log q _ { i k , t } } \\ & { \qquad + \sum _ { i \in \mathcal { O } _ { t } } \sum _ { k = 1 } ^ { K } q _ { i k , t } \big \{ s _ { i , t } [ \psi ( a _ { k , t } ) - \psi ( a _ { k , t } + b _ { k , t } ) ] } \\ & { \qquad + ( n _ { i , t } - s _ { i , t } ) [ \psi ( b _ { k , t } ) - \psi ( a _ { k , t } + b _ { k , t } ) ] \big \} } \\ & { \qquad - \sum _ { k = 1 } ^ { K } \mathrm { K L } [ \mathrm { B e t a } ( a _ { k , t } , b _ { k , t } ) \| \mathrm { B e t a } ( a _ { k , t } ^ { \mathrm { h i s t } } , b _ { k , t } ^ { \mathrm { h i s t } } ) ] . } \end{array}
$$

We omit terms independent of $q _ { t }$ . Here, ψ(·) denotes the digamma function. Appendix A provides the detailed derivation. The ELBO contains two groups of variational parameters. The first group contains the assignment probabilities $q _ { i k , t }$ . The second contains the Beta parameters $( a _ { k , t } , b _ { k , t } )$ We optimize one group while keeping the other fixed. This procedure is known as coordinate ascent. Proposition1 derives the assignment update, and Proposition 2 derives the Beta update. We alternate these updates during inference. Proposition 3 establishes the convergence of this alternating procedure. Appendix A provides the proofs.

Proposition 1: Coordinate-Optimal Assignment Update Fix $q ( \pmb \theta _ { t } )$ and all $q _ { j } ( z _ { j , t } )$ for $j \quad \neq \quad i .$ The coordinate-optimal update is $\begin{array} { r l } { q _ { i k , t } ^ { \star } } & { { } = } \end{array}$ Softmax<sub>k</sub>[log $\begin{array} { r } { A _ { i k } + \beta \sum _ { j = 1 } ^ { N } W _ { i j } q _ { j k , t } + \mathbb { I } i \ \in \ \mathcal { O } _ { t } ) \ell _ { i k , t } ^ { \mathrm { V B } } ] } \end{array}$ where $\begin{array} { r c l } { \ell _ { i k , t } ^ { \mathrm { V B } } } & { = } & { s _ { i , t } \left[ \psi ( a _ { k , t } ^ { - } ) - \psi ( a _ { k , t } + b _ { k , t } ) \right] + ( n _ { i , t } - } \end{array}$ $s _ { i , t } ) \left[ \psi ( b _ { k , t } ) - \psi ( a _ { k , t } + b _ { k , t } ) \right]$

Proposition 2: Coordinate-Optimal Beta Update Fix $q ( \mathbf { z } _ { t } )$ . The coordinate-optimal factor is $\begin{array} { r l } { q ^ { \star } ( \theta _ { k , t } ) } & { { } = } \end{array}$ $\mathrm { B e t a } ( a _ { k , t } , b _ { k , t } )$ , where $\begin{array} { r } { a _ { k , t } = a _ { k , t } ^ { \mathrm { h i s t } } + \sum _ { i \in \mathcal { O } _ { t } } q _ { i k , t } s _ { i , t } } \end{array}$ , and $\begin{array} { r } { b _ { k , t } = b _ { k , t } ^ { \mathrm { h i s t } } + \sum _ { i \in \mathcal { O } _ { t } } q _ { i k , t } \big ( n _ { i , t } - s _ { i , t } \big ) } \end{array}$

Proposition 3: Monotonicity and Convergence At training step t, we keep $( a _ { k , t } ^ { \mathrm { h i s t } } , b _ { k , t } ^ { \mathrm { h i s t } } )$ fixed. Each update in Propositions 1 and 2 maximizes the ELBO with respect to one group of parameters. Therefore, neither update decreases $\mathcal { L } _ { t } ( \bar { q } )$ and the ELBO sequence converges. This convergence provides a stable estimator state at each LLM RL step (though it may not be globally optimal).

We now apply Propositions 1 and 2 in an online alternating procedure. After training step t produces rollout feedback for $\mathcal { O } _ { t } .$ , we initialize inference with the previous estimator state: $q _ { i k , t } ^ { ( 0 ) } = q _ { i k , t - 1 } , a _ { k , t } ^ { ( 0 ) } = a _ { k , t - 1 } , b _ { k , t } ^ { ( 0 ) } = b _ { k , t - 1 } .$ · We keep $( a _ { k , t } ^ { \mathrm { h i s t } } , b _ { k , t } ^ { \mathrm { h i s t } } )$ fixed during inference at step t. Each iteration first applies Proposition 1 to update $q _ { i k , t } .$ . It then applies Proposition 2 to update $( a _ { k , t } , \bar { b _ { k , t } } )$ . The first step updates the latent-state assignments and serves as an E-like update. The second updates the state-level parameters and serves as an M-like update. We repeat both steps until the ELBO increment falls below a preset threshold or the iteration count reaches its maximum.<sup>2</sup> Proposition 1 updates the assignments of both observed and unobserved samples. Proposition 2 updates the state-level success probabilities using the observed feedback. Together, they transform the feedback from ${ \mathcal { O } } _ { t }$ into an updated estimator state for the full training set: ${ \cal S } _ { t } = \{ q _ { t } , a _ { t } , b _ { t } \}$ . The next subsection converts $S _ { t }$ into sample-level dificulty estimates for adaptive scheduling. We also use $S _ { t }$ to initialize inference at the next training step.

## 3.6 Dificulty Estimation for Adaptive RL Schedulers

In this section, we aggregate the state-level posterior from the previous training step into sample-level estimates of success probability, dificulty. These estimates then guide adaptive sample selection and rollout allocation. At the beginning of RL training step t, the estimator loads the state $\bar { S _ { t - 1 } } ~ = ~ \left\{ q _ { t - 1 } , a _ { t - 1 } , \bar { b _ { t - 1 } } \right\}$ from the previous step. For each latent state k, we compute the posterior mean of its success probability as $\begin{array} { r } { \mu _ { k , t - 1 } ~ = ~ \frac { { \bf \ddot { \mu } } a _ { k , t - 1 } } { a _ { k , t - 1 } + b _ { k , t - 1 } } } \end{array}$ , We first form a model-based prediction for sample $x _ { i } \colon { \widehat { p } } _ { i , t } ^ { \mathrm { m o d e l } } \ =$ $\scriptstyle \sum _ { k = 1 } ^ { K } q _ { i k , t - 1 } \mu _ { k , t - 1 }$ . Then we adjust the prediction based on whether the sample has been rolled out before. If $x _ { i }$ has no prior rollouts, we directly use the model prediction: $\widehat { p } _ { i , t } = \bar { \widehat { p } } _ { i , t } ^ { \mathrm { m o d e l } }$ . If it has been sampled in earlier steps, we average the model prediction with its historical success rate $\overline { { r } } _ { i , < t } \colon \overline { { p } } _ { i , t } = \gamma \widehat { p } _ { i . t } ^ { \mathrm { m o d e l } } + ( 1 - \gamma ) \bar { r } _ { i , < t } ) ( \gamma = 0 . 5 $ , see discussion in Appendix E). Also, the estimated dificulty is then $\widehat { d } _ { i , t } = 1 - \widehat { p } _ { i , t }$

Overall, before training step t, the estimator predicts the dificulty of every sample. These outputs serve as a plugand-play component for most sample-level scheduling and resource-allocation frameworks in RLVR training.

## Experiment

## Implementation Details

Setting. We conduct all training and inference on 4×A100 80G GPUs. To avoid data leakage (Wu et al. 2026), following Yao et al. (2026) and Zeng et al. (2026), we adopt Qwen 2.5 1.5B Math (Yang et al. 2024) and Llama 3 1B-Instruct (Grattafiori et al. 2024) as base models. For the main experiments, we keep the environment strictly consistent with the original baselines. We integrate our framework as a plug-and-play component into the original code to demonstrate that it improves the final RL performance. For fairness, we use the same computational budget for all experiments, equivalent to one round of the standard GVM algorithm (∼1.5M rollouts, including pre-sampling). We use Qwen3-0.6B (Zhang et al. 2025) as the sentence embedding model, and set $K \stackrel { \bar { = } } { = } 3 2 0 , k _ { \mathrm { n n } } = 8 0 , \gamma = 0 . 5$ and $\beta = 2 .$ Guidelines for hyperparameter or embedding model selection across diferent setting are provided in Appendix E.

Dataset. Since all baselines we integrate are based on mathematical problems, we follow the experimental settings of previous works (Yao et al. 2026; Gao et al. 2026; Zheng et al. 2026) and conduct reinforcement learning on Numina-Math (Li et al. 2024; Cui et al. 2025). We evaluate the resulting models on MATH500 (Hendrycks et al. 2021; Lightman et al. 2024), AIME 2024 (Balunovic et al. 2026), AIME 2025 (Dekoninck et al. 2026), and OlympiadBench (He et al. 2024). We provide detailed descriptions of these datasets in Appendix B and discuss their generalizability to other domains in Appendix D.

<table><tr><td rowspan="2">Method</td><td colspan="4">Qwen-2.5-Math-1.5B (Average@8)</td><td colspan="4">Llama-3.2-1B-Instruct (Average@8)</td></tr><tr><td>MATH500</td><td>AIME24</td><td>AIME25</td><td>Olym</td><td>MATH500</td><td>AIME24</td><td>AIME25</td><td>Olym</td></tr><tr><td>Base</td><td>39.3</td><td>3.30</td><td>3.33</td><td>25.2</td><td>17.5</td><td>0.00</td><td>0.00</td><td>3.02</td></tr><tr><td>GVM</td><td>71.9</td><td>11.3</td><td>11.3</td><td>35.8</td><td>23.9</td><td>0.42</td><td>0.00</td><td>4.59</td></tr><tr><td>GVM+Ours</td><td>74.7</td><td>16.7</td><td>13.3</td><td>38.3</td><td>25.7</td><td>3.30</td><td>0.00</td><td>4.89</td></tr><tr><td>PCL</td><td>59.7</td><td>7.92</td><td>4.58</td><td>26.1</td><td>26.7</td><td>0.42</td><td>0.00</td><td>5.04</td></tr><tr><td>PCL+Ours</td><td>61.6</td><td>11.3</td><td>6.67</td><td>27.1</td><td>27.4</td><td>0.00</td><td>0.42</td><td>5.81</td></tr><tr><td>GRESO</td><td>66.8</td><td>10.0</td><td>7.50</td><td>30.3</td><td>30.0</td><td>2.50</td><td>0.00</td><td>7.59</td></tr><tr><td>GRESO+Ours</td><td>68.2</td><td>10.0</td><td>10.0</td><td>31.9</td><td>32.6</td><td>6.67</td><td>0.42</td><td>8.26</td></tr></table>

Table 1: The Average@8 accuracy of diferent RL scheduling methods on four mathematical reasoning benchmarks using two base models. The first column lists the original schedulers and their variants integrated with our dificulty estimator. Underlined results indicate improvements over the corresponding original scheduler $( p < 5 \times 1 0 ^ { - 5 }$ , see Appendix C).

Baseline. We evaluate our plug-and-play component on three representative baselines from diferent scheduling paradigms: GVM, a rollout-allocation method (Yao et al. 2026); PCL, a curriculum-learning method that trains a dedicated dificulty predictor (Gao et al. 2026); and GRESO, a curriculum-learning method that uses historical success rates as dificulty signals (Zheng et al. 2026). For each baseline, we replace only its original dificulty estimator with ours.

Metric. We use rule-based matching to judge whether the result matches the ground truth (Guo et al. 2025; Sheng et al. 2024). Because a single evaluation on small and challenging datasets such as AIME can be sensitive to sampling randomness, we follow prior work (Yao et al. 2026; Zheng et al. 2026) and sample eight independent rollouts per prompt, reporting Average@8 accuracy.

## Overall Performance

We integrate our estimator into three representative RL schedulers: GVM for rollout allocation, PCL for curriculum selection, and GRESO for selective rollout. We replace their original dificulty estimators while retaining their scheduling strategies, and compare all methods under the same total generation budget.

As shown in Table 1, our estimator improves the overall performance of all three schedulers, with gains in most model–dataset settings. For GVM, GVM+Ours retains presampling only for gradient-related estimation and replaces the original dificulty estimate with ours. It improves performance in seven of the eight settings and matches the baseline in the remaining one. We will provide a more indepth analysis in the next section. When integrated into PCL and GRESO, our estimator also improves the corresponding baselines in most settings. A plausible explanation is that the more stable dificulty estimates help the schedulers identify samples that are likely to yield both correct and incorrect responses within the same rollout group. Such samples produce nondegenerate group-relative advantages and potentially more informative optimization signals.

Although full-history aggregation may introduce temporal lag, graph propagation and shared state-level statistics make its efect largely systematic across related samples, afecting absolute calibration more than relative dificulty ranking. Since adaptive schedulers primarily rely on ranking for sample selection and rollout allocation, this bias has limited impact on scheduling, as supported by the high correlations and downstream gains above. Appendix G further examines a windowed variant that reduces stale historical influence.

Overall, these results suggest that our estimator can serve as a reusable component for diferent RL scheduling strategies and improve downstream performance under a matched generation budget.

## Can Our Method Continuously Track Sample Dificulty During RL Training?

We evaluate how accurately diferent methods track evolving sample dificulty along the same GRPO training trajectory. At each evaluation point, we use the empirical success rate from 8 independent rollouts generated by the corresponding policy as the reference empirical success probability. We report sample-level MAE for calibration and batch-level Pearson correlation r (batch size 256) for ranking consistency in Table 2, since sample-level empirical rates are discrete and sparse. Lower MAE and higher r indicate better estimation.

We compare two high-cost probing baselines, Sample Before RL (SBR) and Sample Before Step (SBS), with three representative low-cost approaches: PCL (LLM-based difficulty prediction), VIP (historical-feedback-based statistical estimation) , and MoPPS (history-based dynamic modeling). SBS probes samples repeatedly with the current policy; it therefore achieves the highest accuracy but requires roughly 45–48 hours of extra computation. SBR, however, estimates dificulty only once before training, so its predictions quickly become outdated as the policy evolves, highlighting the limitation of static estimates. Our approach performs best overall among low-cost methods, with a particularly clear advantage in the early sparse-feedback stage. Early in training, it achieves Pearson correlations of 0.482 for Qwen and 0.469 for Llama, together with the lowest MAE among the low-cost baselines. These results indicate that the graph structure alleviates observation sparsity by transferring feedback across related samples. As more rollout observations accumulate, the full-trajectory correlations increase to 0.836 and 0.776, while the total estimation overhead remains approximately 0.12h. Overall, our method mitigates early-stage observation sparsity and tracks policy-induced dificulty changes with low computational overhead.

<table><tr><td rowspan="2">Base Model</td><td rowspan="2">Methods</td><td colspan="2">Early Steps</td><td colspan="2">Middle Steps</td><td colspan="2">Last Steps</td><td colspan="2">Full Steps</td><td rowspan="2">Time (A100*h)</td></tr><tr><td>MAE (↓)</td><td>r (↑)</td><td>MAE (↓)</td><td>r (↑)</td><td>MAE (↓)</td><td>r (↑)</td><td>MAE (↓)</td><td>r (↑)</td></tr><tr><td rowspan="6">Qwen-2.5 Math-1.5B</td><td>Sample before RL</td><td>0.116</td><td>0.544</td><td>0.139</td><td>0.856</td><td>0.151</td><td>0.865</td><td>0.135</td><td>0.522</td><td>~29.6</td></tr><tr><td>Sample before Step</td><td>0.063</td><td>0.959</td><td>0.054</td><td>0.971</td><td>0.049</td><td>0.974</td><td>0.055</td><td>0.969</td><td>~45.3</td></tr><tr><td>VIP</td><td>0.373</td><td></td><td>0.094</td><td>0.794</td><td>0.084</td><td>0.885</td><td>0.184</td><td>0.423</td><td>~0.05</td></tr><tr><td>PCL</td><td>0.404</td><td>0.138</td><td>0.336</td><td>0.407</td><td>0.328</td><td>0.474</td><td>0.356</td><td>0.281</td><td>~7.36</td></tr><tr><td>MoPPS</td><td>0.373</td><td></td><td>0.149</td><td>0.910</td><td>0.118</td><td>0.929</td><td>0.214</td><td>0.754</td><td></td></tr><tr><td>Ours</td><td>0.290</td><td>0.482</td><td>0.133</td><td>0.912</td><td>0.123</td><td>0.943</td><td>0.183</td><td>0.836</td><td>~0.12</td></tr><tr><td rowspan="6">Llama-3.2 1B-Instruct</td><td>Sample before RL</td><td>0.116</td><td>0.374</td><td>0.141</td><td>0.648</td><td>0.154</td><td>0.671</td><td>0.137</td><td>0.394</td><td>~33.4</td></tr><tr><td>Sample before Step</td><td>0.044</td><td>0.986</td><td>0.046</td><td>0.967</td><td>0.046</td><td>0.967</td><td>0.045</td><td>0.986</td><td>~47.9</td></tr><tr><td>VIP</td><td>0.413</td><td></td><td>0.107</td><td>0.895</td><td>0.084</td><td>0.880</td><td>0.201</td><td>0.717</td><td>~0.05</td></tr><tr><td>PCL</td><td>0.204</td><td>0.467</td><td>0.181</td><td>0.404</td><td>0.187</td><td>0.525</td><td>0.191</td><td>-0.231</td><td>~4.38</td></tr><tr><td>MoPPS</td><td>0.413</td><td></td><td>0.133</td><td>0.864</td><td>0.109</td><td>0.869</td><td>0.218</td><td>0.606</td><td></td></tr><tr><td>Ours</td><td>0.197</td><td>0.469</td><td>0.100</td><td>0.868</td><td>0.075</td><td>0.913</td><td>0.131</td><td>0.776</td><td>~0.12</td></tr></table>

Table 2: Dificulty-estimation accuracy and overhead at diferent stages of GRPO training (batch size =256). MAE is computed at the sample level and Pearson correlation r at the batch level. Bold denotes the best overall result, underlining denotes the best result among low-cost methods, and Time (including embedding, clustering and inference) denotes additional estimation overhead for over 3 epochs on 150K training samples.

<table><tr><td rowspan="2">Base Model</td><td rowspan="2">Methods</td><td colspan="2">Full Steps</td></tr><tr><td>MAE (↓)</td><td>r (↑)</td></tr><tr><td rowspan="4">Qwen-2.5 Math-1.5B</td><td>Ours</td><td>0.183</td><td>0.836</td></tr><tr><td>w/o Spectral Init.</td><td>0.218</td><td>0.710</td></tr><tr><td>w/o Prior Smoothing</td><td>0.197</td><td>0.783</td></tr><tr><td>w/o Graph Sparsification</td><td>0.187</td><td>0.835</td></tr><tr><td rowspan="4">Llama-3.2 1B-Instruct</td><td>Ours</td><td>0.131</td><td>0.776</td></tr><tr><td>w/o Spectral Init.</td><td>0.197</td><td>0.686</td></tr><tr><td>w/o Prior Smoothing</td><td>0.144</td><td>0.736</td></tr><tr><td>w/o Graph Sparsification</td><td>0.134</td><td>0.742</td></tr></table>

Table 3: Full-trajectory ablation results for dificulty estimation on two base models. Lower MAE and higher Pearson correlation r indicate better estimation.

## Ablation Study

We ablate spectral initialization, prior smoothing, and graph sparsification. Table 3 reports results over the full training trajectory. Random initialization causes a substantial performance drop. Since our EM-style optimization only guarantees convergence to a local optimum, spectral clustering provides a high-quality starting point that guides inference toward a better solution, whereas random initialization can lead to poor local optima. Replacing the smoothed latentstate prior $A _ { i k }$ with hard assignments also degrades performance. As the policy evolves, sample dificulty and its latentstate assignment should evolve accordingly. The smoothed prior preserves the initial clustering structure while allowing rollout feedback to revise the assignments, whereas hard assignments prevent such adaptation. Using a dense graph causes a modest decline because weakly related edges introduce noise into feedback propagation. Graph sparsification retains more reliable relationships between samples.

## Do Spectral Clusters Capture Semantic and Dificulty Structure?

We use MATH (Hendrycks et al. 2021) for this analysis because it provides standardized annotations of both subject category and dificulty level, allowing us to evaluate the semantic and dificulty information captured by the clusters separately. Following Sec.3.3, we embed the MATH problems and perform spectral clustering. The resulting clusters are strongly associated with subject categories $( \chi ^ { 2 } = 1 0 , 9 9 5 . 8 4 , p \check { < } 1 0 ^ { - 3 0 0 }$ ; Cramér’s $V \overset { \cdot } { = } 0 . 4 9 4 )$ , indicating the spectral partition efectively captures semantic structure. After controlling for subject category, dificulty distributions still difer significantly across clusters (Kruskal–Wallis, $p _ { m a x } < 5 \times \mathrm { { i } 0 ^ { - 3 } ) }$ . These results show the clusters capture not only problem semantics but also dificulty information beyond subject identity, thereby providing an informative initial structure for subsequent online dificulty inference. We provide the full statistical details in Appendix F.

## Conclusion

In this work, we address ineficient exploration-budget allocation in RLVR through graph-based online dificulty estimation. Our framework connects samples with similar semantic content and reasoning structures and uses rollout feedback to track their evolving dificulty without dedicated probing. The resulting estimates support both sample-selection and rollout-allocation schedulers. Experiments across multiple base models, schedulers, and benchmarks show that our framework improves downstream reasoning performance in most settings under matched rollout budgets while adding little online computational overhead. These results demonstrate the potential of online dificulty estimation to support more eficient RLVR training.

## A. ELBO Derivation and Proofs

This section derives the expanded evidence lower bound (ELBO) in Sec. 3.5 and proves Propositions 1–3. For compactness, define $\delta _ { i k , t } = \mathbb { I } ( z _ { i , t } = k )$ and let $C _ { t }$ collect all terms that are independent of the variational distribution. The mutual-neighbor graph is undirected, so ${ \cal W } _ { i j } = { \cal W } _ { j i } ,$ and it has no self-loops, so $W _ { i i } = 0$

## A.1 Derivation of the Expanded ELBO

From the joint model in Sec. 3.4, its log density can be written, up to the constant $C _ { t }$ , as

$$
\begin{array} { r l } & { \displaystyle \log p _ { k } ( \mathbf { s } _ { t } , \mathbf { z } _ { t } , \theta _ { t } | \mathbf { \phi } _ { \mathbf { n } _ { k } , W _ { k } , A } ) } \\ & { = C _ { t } + \displaystyle \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \delta _ { i k , t } \log A _ { k t } + \displaystyle \frac { \beta } { 2 } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } W _ { i j } \displaystyle \sum _ { k = 1 } ^ { K } \delta _ { i k , t } \delta _ { j k , t } } \\ & { \quad + \displaystyle \sum _ { k = 1 } ^ { K } \left[ ( a \mathrm { h } _ { k t } ^ { \mathrm { l i e t } } - 1 ) \log \theta _ { k t } + ( b _ { k t } ^ { \mathrm { l i e t } } - 1 ) \log ( 1 - \theta _ { k t } ) \right] } \\ & { \quad + \displaystyle \sum _ { i \in O , t \leq i - 1 } ^ { K } \delta _ { i k , t } \log \theta _ { k t } } \\ & { \quad + \displaystyle \sum _ { i \in O , t } \sum _ { k = 1 } ^ { K } \delta _ { i k , t } ( \ln ( \boldsymbol { u } _ { i \cdot t } - \boldsymbol { s } _ { t , t } ) } \\ & { \quad + \displaystyle \sum _ { i \in O } \sum _ { k = 1 } ^ { K } \delta _ { i k , t } ( \ln ( \boldsymbol { u } _ { i \cdot t } - \boldsymbol { s } _ { t , t } ) } \end{array}\tag{1}
$$

Here, $C _ { t }$ includes − log $Z ( W , A )$ , the Beta normalizers, and the binomial coeficients, all of which are fixed with respect to $q _ { t }$

Under the mean-field distribution,

$$
\mathbb { E } _ { q _ { t } } [ \delta _ { i k , t } ] = q _ { i k , t } ,\tag{2}
$$

$$
\mathbb { E } _ { q _ { t } } [ \delta _ { i k , t } \delta _ { j k , t } ] = q _ { i k , t } q _ { j k , t } \quad ( i \neq j ) ,\tag{3}
$$

$$
\mathbb { E } _ { q _ { t } } [ \log \theta _ { k , t } ] = \psi ( a _ { k , t } ) - \psi ( a _ { k , t } + b _ { k , t } ) ,\tag{4}
$$

$$
\mathbb { E } _ { q _ { t } } [ \log ( 1 - \theta _ { k , t } ) ] = \psi ( b _ { k , t } ) - \psi ( a _ { k , t } + b _ { k , t } ) .\tag{5}
$$

The second identity is suficient for the graph term because $W _ { i i } = 0$ . The negative entropy contribution factorizes as

$$
\begin{array} { r l r } & { } & { - \mathbb { E } _ { q _ { t } } [ \log { q _ { t } } ] = - \displaystyle \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } q _ { i k , t } \log { q _ { i k , t } } } \\ & { } & { - \displaystyle \sum _ { k = 1 } ^ { K } \mathbb { E } _ { q _ { t } } [ \log { q ( \theta _ { k , t } ) } ] . } \end{array}\tag{6}
$$

Combining the expected log Beta prior with the corresponding Beta-factor entropy gives

$$
\begin{array} { r l } & { \mathbb { E } _ { q _ { t } } \left[ \log \mathrm { B e t a } ( \theta _ { k , t } \mid a _ { k , t } ^ { \mathrm { h i s t } } , b _ { k , t } ^ { \mathrm { h i s t } } ) - \log q ( \theta _ { k , t } ) \right] } \\ & { \qquad = - \mathrm { K L } \left[ \mathrm { B e t a } ( a _ { k , t } , b _ { k , t } ) \parallel \mathrm { B e t a } ( a _ { k , t } ^ { \mathrm { h i s t } } , b _ { k , t } ^ { \mathrm { h i s t } } ) \right] } \end{array}\tag{7}
$$

Substituting these identities into $\mathcal { L } _ { t } ( q _ { t } ) ~ = ~ \mathbb { E } _ { q _ { t } } [ \log p _ { t } ] ~ -$ $\mathbb { E } _ { q _ { t } } \big [ \mathrm { l o g } { q _ { t } } \big ]$ yields the expression below. To keep the display compact, let

$$
D _ { k , t } ^ { \mathrm { B e t a } } = \mathrm { K L } \big [ \mathrm { B e t a } ( a _ { k , t } , b _ { k , t } ) \big | \big | \mathrm { B e t a } ( a _ { k , t } ^ { \mathrm { h i s t } } , b _ { k , t } ^ { \mathrm { h i s t } } ) \big ] .\tag{8}
$$

Then

$$
\begin{array} { r l } { \mathcal { L } _ { i } ( q _ { k } ) = \displaystyle \frac { \beta } { 2 } \sum _ { i = 1 } ^ { N } \sum _ { s = 1 } ^ { N } } & { \displaystyle \sum _ { k = 1 } ^ { C } q _ { k , i } q _ { j , k } , } \\ & { + \displaystyle \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { C } q _ { k , i } \log A _ { i j k } } \\ & { - \displaystyle \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } q _ { i , k } \log 4 u _ { k , i } } \\ & { + \displaystyle \sum _ { i \in G } \sum _ { \alpha _ { k } = 1 } ^ { N } \sum _ { \alpha _ { k } = 1 } ^ { K } \frac { \alpha _ { \alpha _ { k } } } { \ell _ { k , i } ^ { \mathrm { v b } } } } \\ & { + \displaystyle \sum _ { i \in G } \sum _ { \alpha _ { k } = 1 } ^ { N } q _ { i , k } \ell _ { i \alpha _ { k } } ^ { \mathrm { v b } } } \\ & { - \displaystyle \sum _ { k = 1 } ^ { K } D _ { k , k } ^ { \mathrm { a r s } } + C _ { k } , } \end{array}\tag{9}
$$

where

$$
\begin{array} { r l } & { \ell _ { i k , t } ^ { \mathrm { V B } } = s _ { i , t } \left[ \psi ( a _ { k , t } ) - \psi ( a _ { k , t } + b _ { k , t } ) \right] } \\ & { ~ + \left( n _ { i , t } - s _ { i , t } \right) \left[ \psi ( b _ { k , t } ) - \psi ( a _ { k , t } + b _ { k , t } ) \right] . } \end{array}\tag{10}
$$

Dropping $C _ { t } .$ , which is independent of $q _ { t }$ , gives exactly the expanded ELBO reported in Sec. 3.5.

## A.2 Proof of Proposition 1

Fix $q ( \pmb \theta _ { t } )$ and every assignment factor except $q _ { i } \big ( z _ { i , t } \big )$ . Because $W$ is symmetric, the two appearances of node i in the double graph sum combine and cancel the factor $1 / 2$ . The terms of Eq. (9) that depend on $q _ { i k , t }$ are therefore

$$
\mathcal { L } _ { t } ( q _ { i } ) = \sum _ { k = 1 } ^ { K } q _ { i k , t } h _ { i k , t } - \sum _ { k = 1 } ^ { K } q _ { i k , t } \log q _ { i k , t } + C ,\tag{11}
$$

$$
h _ { i k , t } = \log A _ { i k } + \beta \sum _ { j = 1 } ^ { N } W _ { i j } q _ { j k , t } + \mathbb { I } ( i \in \mathcal { O } _ { t } ) \ell _ { i k , t } ^ { \mathrm { V B } } .\tag{12}
$$

Introduce a Lagrange multiplier $\lambda _ { i }$ for $\textstyle \sum _ { k = 1 } ^ { K } q _ { i k , t } = 1$ and define

$$
\widetilde { \mathscr { L } } _ { t } ( q _ { i } ) = \mathscr { L } _ { t } ( q _ { i } ) + \lambda _ { i } \left( \sum _ { r = 1 } ^ { K } q _ { i r , t } - 1 \right) .\tag{13}
$$

Diferentiating gives

$$
\frac { \partial \widetilde { \mathcal { L } } _ { t } ( q _ { i } ) } { \partial q _ { i k , t } } = h _ { i k , t } - \log q _ { i k , t } - 1 + \lambda _ { i } .\tag{14}
$$

Setting this derivative to zero and normalizing over k yields

$$
q _ { i k , t } ^ { \star } = \frac { \exp ( h _ { i k , t } ) } { \sum _ { r = 1 } ^ { K } \exp ( h _ { i r , t } ) } = \mathrm { S o f t m a x } _ { k } ( h _ { i k , t } ) ,\tag{15}
$$

which is the update in Proposition 1.

## A.3 Proof of Proposition 2

Fix $q ( \mathbf { z } _ { t } )$ and define $\begin{array} { r } { q _ { - k , t } = q ( \mathbf { z } _ { t } ) \prod _ { r \neq k } q ( \theta _ { r , t } ) } \end{array}$ . The standard mean-field coordinate identity gives

$$
\begin{array} { r } { \log q ^ { \star } ( \theta _ { k , t } ) = \mathbb { E } _ { q _ { - k , t } } \left[ \log p _ { t } ( \mathbf { s } _ { t } , \mathbf { z } _ { t } , \pmb { \theta } _ { t } } \\ { \mid \mathbf { n } _ { t } , W , A ) \right] + C . } \end{array}\tag{16}
$$

Retaining only terms involving $\theta _ { k , t }$ in $\operatorname { E q . }$ . (1) gives

$$
\alpha _ { k , t } = a _ { k , t } ^ { \mathrm { h i s t } } + \sum _ { i \in \mathcal { O } _ { t } } q _ { i k , t } s _ { i , t } ,\tag{17}
$$

$$
\gamma _ { k , t } = b _ { k , t } ^ { \mathrm { h i s t } } + \sum _ { i \in \mathcal { O } _ { t } } q _ { i k , t } ( n _ { i , t } - s _ { i , t } ) ,\tag{18}
$$

$$
\begin{array} { r l } & { \log q ^ { \star } ( \theta _ { k , t } ) = ( \alpha _ { k , t } - 1 ) \log \theta _ { k , t } } \\ & { \qquad + ( \gamma _ { k , t } - 1 ) \log ( 1 - \theta _ { k , t } ) + C . } \end{array}\tag{19}
$$

This is the log density of a Beta distribution. Hence

$$
q ^ { \star } ( \theta _ { k , t } ) = \mathrm { B e t a } ( a _ { k , t } , b _ { k , t } ) ,\tag{20}
$$

$$
a _ { k , t } = a _ { k , t } ^ { \mathrm { h i s t } } + \sum _ { i \in \mathcal { O } _ { t } } q _ { i k , t } s _ { i , t } ,\tag{21}
$$

$$
b _ { k , t } = b _ { k , t } ^ { \mathrm { h i s t } } + \sum _ { i \in \mathcal { O } _ { t } } q _ { i k , t } ( n _ { i , t } - s _ { i , t } ) ,\tag{22}
$$

which proves Proposition 2.

## A.4 Sequential and Synchronous Implementations

Proposition 1 gives the exact coordinate-optimal update for a single assignment factor $q _ { i } ( z _ { i , t } )$ while all other factors are held fixed. We consider two implementations of this update.

Sequential reference implementation. Our reference implementation performs a strict sequential Gauss–Seidel sweep over the samples. Under the update order $\begin{array} { r l } { i } & { { } = } \end{array}$ $1 , \ldots , N$ , the assignment update at iteration $m + 1$ is

$$
\begin{array} { c } { q _ { i k , t } ^ { ( m + 1 ) } = \mathrm { S o f t m a x } _ { k } \left[ \log A _ { i k } + \beta \displaystyle \sum _ { j < i } W _ { i j } q _ { j k , t } ^ { ( m + 1 ) } + \beta \displaystyle \sum _ { j > i } W _ { i j } q _ { j k , j } ^ { ( m ) } \right] } \\ { + \mathbb { I } ( i \in \mathcal { O } _ { t } ) \ell _ { i k , t } ^ { \mathrm { V B } , ( m ) } \displaystyle \Bigg ] . } \end{array}
$$

Thus, each update uses the most recently available values of the previously updated factors. After one complete assignment sweep, the Beta factors are updated according to Proposition 2. Because every assignment factor is optimized while the remaining factors are fixed, this implementation is an exact coordinate-ascent variational inference procedure. The monotonicity and convergence results in Proposition 3 apply to this sequential implementation.

Synchronous accelerated implementation. The strict sequential update is dificult to parallelize over samples. We therefore additionally implement a synchronous Jacobi-style update:

$$
\widetilde { q } _ { i k , t } ^ { ( m + 1 ) } = \mathrm { S o f t m a x } _ { k } \left[ \log A _ { i k } + \beta \sum _ { j = 1 } ^ { N } W _ { i j } q _ { j k , t } ^ { ( m ) } + \mathbb { I } ( i \in \mathcal { O } _ { t } ) \ell _ { i k , t } ^ { \mathrm { V B } , ( m ) } \right]\tag{24}
$$

All assignment factors in Eq. (24) are computed in parallel from the same previous iterate, after which the Beta factors are updated using Proposition 2.

Unlike the sequential implementation, a synchronous assignment sweep is not an exact block-coordinate maximization of the ELBO. Consequently, Proposition 3 does not directly guarantee that every synchronous sweep is ELBO non-decreasing. We treat the synchronous implementation as a parallel approximation to the theoretically grounded sequential procedure.

To evaluate this approximation, we compare the converged sample-level success-probability estimates produced by the two implementations. We define their discrepancy at training step t as

$$
\Delta _ { t } = \operatorname* { m a x } _ { 1 \leq i \leq N } \left| \widehat { p } _ { i , t } ^ { \mathrm { s y n c } } - \widehat { p } _ { i , t } ^ { \mathrm { s e q } } \right| .\tag{25}
$$

Across the evaluated training checkpoints, the discrepancy remains within $1 0 ^ { - 2 }$ . At the same time, the synchronous implementation allows the graph-message aggregation and assignment updates to be executed eficiently in parallel, resulting in substantially better wall-clock performance. We therefore use and recommend the synchronous implementation in practice, while retaining the sequential implementation as the reference algorithm for the theoretical analysis.

## A.5 Proof of Proposition 3

We prove the result for the strict sequential implementation described in Sec. A.4. Fix a training step t. Throughout the inference procedure at this step, the historical parameters $( a _ { k , t } ^ { \mathrm { h i s t } } , b _ { k , t } ^ { \mathrm { h i s t } } )$ , the graph $( W , A )$ , and all rollout observations are fixed.

We assume that

$$
A _ { i k } > 0 \quad \mathrm { f o r e v e r y } i , k ,\tag{26}
$$

$$
a _ { k , t } ^ { \mathrm { h i s t } } > 0 , \qquad b _ { k , t } ^ { \mathrm { h i s t } } > 0 \quad \mathrm { f o r } \mathrm { e v e r y } k ,\tag{27}
$$

$$
\vert W _ { i j } \vert < \infty , \qquad n _ { i , t } < \infty .\tag{28}
$$

he first condition is satisfied by the smoothed initialization hen $0 < \epsilon < 1$ . We further assume that the categorical ctors are updated cyclically and that every factor is updated once in each complete sequential sweep.

Monotonicity. Fix all variational factors except one categorical factor $q _ { i } ( z _ { i , t } )$ . Proposition 1 gives the unique maximizer of the ELBO with respect to this factor. Therefore, updating $q _ { i } \big ( z _ { i , t } \big )$ cannot decrease the ELBO.

Similarly, after fixing $q ( \mathbf { z } _ { t } )$ and all Beta factors except $q ( \theta _ { k , t } )$ , Proposition 2 gives the coordinate-optimal Beta factor. Updating this factor therefore cannot decrease the ELBO. Consequently, $\operatorname { i f } q _ { t } ^ { ( m ) }$ denotes the variational distribution after the m-th coordinate update, then

$$
\mathcal { L } _ { t } ( q _ { t } ^ { ( m + 1 ) } ) \geq \mathcal { L } _ { t } ( q _ { t } ^ { ( m ) } ) .\tag{29}
$$

Boundedness of the parameter sequence. Each categorparameter belongs to the probability simplex:

$$
( q _ { i 1 , t } , \dots , q _ { i K , t } ) \in \Delta _ { K } .\tag{30}
$$

The product of the N categorical simplexes is compact.

Define the total number of observed successes and failures at training step t as

$$
S _ { t } = \sum _ { i \in \mathcal { O } _ { t } } s _ { i , t } ,\tag{31}
$$

$$
F _ { t } = \sum _ { i \in \mathcal { O } _ { t } } ( n _ { i , t } - s _ { i , t } ) .\tag{32}
$$

By Proposition 2 and $0 \leq q _ { i k , t } \leq 1$ , the Beta parameters satisfy

$$
a _ { k , t } ^ { \mathrm { h i s t } } \leq a _ { k , t } \leq a _ { k , t } ^ { \mathrm { h i s t } } + S _ { t } ,\tag{33}
$$

$$
b _ { k , t } ^ { \mathrm { h i s t } } \leq b _ { k , t } \leq b _ { k , t } ^ { \mathrm { h i s t } } + F _ { t } .\tag{34}
$$

Therefore, all Beta parameters remain in compact positive intervals. The complete variational-parameter sequence consequently lies in the compact set

$$
\begin{array} { l } { \displaystyle \mathcal { X } _ { t } = \left( \displaystyle \prod _ { i = 1 } ^ { N } \Delta _ { K } \right) } \\ { \displaystyle \quad \times \prod _ { k = 1 } ^ { K } \big [ a _ { k , t } ^ { \mathrm { h i s t } } , a _ { k , t } ^ { \mathrm { h i s t } } + S _ { t } \big ] } \\ { \displaystyle \quad \times \prod _ { k = 1 } ^ { K } \big [ b _ { k , t } ^ { \mathrm { h i s t } } , b _ { k , t } ^ { \mathrm { h i s t } } + F _ { t } \big ] . } \end{array}\tag{35}
$$

In particular, the parameter sequence has at least one accumulation point.

Convergence of the ELBO values. Let

$$
p _ { t } ^ { \mathrm { p o s t } } = p _ { t } ( \mathbf { z } _ { t } , \pmb { \theta } _ { t } \mid \mathbf { s } _ { t } , \mathbf { n } _ { t } , W , A )\tag{36}
$$

denote the exact posterior. The ELBO satisfies

$$
\begin{array} { r } { \mathcal { L } _ { t } ( q _ { t } ) = \log p _ { t } ( \mathbf { s } _ { t } \mid \mathbf { n } _ { t } , W , A ) } \\ { - \mathrm { K L } \left[ q _ { t } \parallel p _ { t } ^ { \mathrm { p o s t } } \right] . } \end{array}\tag{37}
$$

Because the model has a finite number of latent assignments, proper Beta priors, and finite rollout counts, the log evidence is finite. Since the KL divergence is nonnegative,

$$
\begin{array} { r } { \mathcal { L } _ { t } ( q _ { t } ) \leq \log p _ { t } ( \mathbf { s } _ { t } \mid \mathbf { n } _ { t } , W , A ) . } \end{array}\tag{38}
$$

Combining this upper bound with Eq. (29) shows that the ELBO-value sequence is monotone and bounded above. Hence there exists a finite $\mathcal { L } _ { t } ^ { \star }$ such that

$$
\operatorname* { l i m } _ { m \to \infty } \mathcal { L } _ { t } ( q _ { t } ^ { ( m ) } ) = \mathcal { L } _ { t } ^ { \star } .\tag{39}
$$

Stationarity of accumulation points. Consider the iterates at the boundaries of complete sequential sweeps, and let $T _ { t }$ denote the mapping corresponding to one full sweep of all categorical and Beta-factor updates. Under Eqs. (26)– (28), the softmax assignment update, the Beta update, and the associated digamma expectations are continuous on $\mathcal { X } _ { t }$ Therefore, $T _ { t }$ is continuous.

Let $\bar { q } _ { t }$ be an accumulation point of the sweep-level variational sequence. Suppose, for contradiction, that $T _ { t } ( \bar { q } _ { t } ) \neq \bar { q } _ { t } .$ Because every coordinate update is the unique maximizer of its coordinate subproblem, at least one update in the sweep would produce a strict ELBO increase. Hence

![](images/81617fd67c24cdf31f3829b86327ac74faf094593bba82ce2f4c5169d67c3393.jpg)

RL Step 10  
![](images/7864f9e44dd6ea57509ac3c5d196dd10af285a8a4138d57c76efef92a132e48c.jpg)  
Figure 2: EM iterations

$$
\mathcal { L } _ { t } ( T _ { t } ( \bar { q } _ { t } ) ) > \mathcal { L } _ { t } ( \bar { q } _ { t } ) .\tag{40}
$$

By continuity, the same positive improvement would hold in a neighborhood of $\bar { q } _ { t }$ . This contradicts the convergence of the ELBO values in Eq. (39), according to which the ELBO improvement over a complete sweep must approach zero. Therefore,

$$
T _ { t } ( \bar { q } _ { t } ) = \bar { q } _ { t } .\tag{41}
$$

Thus, every accumulation point is a fixed point of all exact coordinate updates and is consequently coordinate-wise optimal. Since the categorical updates are strictly positive under $A _ { i k } ~ > ~ 0$ , and the Beta parameters remain strictly positive, the ELBO is diferentiable at such a fixed point subject to the simplex constraints. The corresponding first-order Karush–Kuhn–Tucker conditions are therefore satisfied. Every accumulation point is hence a stationary point of the ELBO.

The argument establishes convergence of the ELBO values and stationarity of every accumulation point.

## A.6 EM Iteration Visualization

We show the evolution of the ELBO under EM iterations in Figure 2. In general, the M-step performs a notably large update only at the very first iteration. The per-round improvement in the ELBO from the E-step and M-step decays roughly exponentially, and convergence to a local stationary point or local maximum is typically reached within 50 iterations.

## B. Details of Datasets and Training

NuminaMath. NuminaMath is a large-scale mathematical reasoning dataset comprising approximately 860K problem–solution pairs. It covers problems ranging from Chinese high-school mathematics to U.S. and international mathematical olympiad competitions, with data primarily collected from online examination papers and mathematics discussion forums. Each problem is accompanied by a chain-of-thought solution and a verifiable final answer. Following the dataprocessing protocols of GVM and CurES, we extract approximately 150K problems from the original corpus for RL training (Cui et al. 2025).

MATH. MATH is a competition-level mathematical reasoning dataset containing 12.5K problems, each accompanied by a complete step-by-step solution and a final answer. The dataset provides predefined dificulty annotations ranging from Level 1 to Level 5 and category labels covering seven subjects: prealgebra, algebra, number theory, counting and probability, geometry, intermediate algebra, and precalculus. We use these annotations only as external references rather than as training supervision. Specifically, we compare the complete method with variants that remove the clustering or graph-based components and examine how well the learned representations and structural relationships align with the provided category and dificulty labels. This analysis assesses whether the proposed components capture semantic similarity among problems and distinguish problems at diferent dificulty levels.

MATH-500. MATH-500 is a representative subset of 500 problems selected from the original MATH benchmark. It retains the diverse mathematical subjects and dificulty levels of MATH while enabling more eficient and standardized evaluation. Each problem is accompanied by a reference solution and a verifiable final answer. We use MATH-500 to evaluate the models’ general mathematical reasoning ability across a broad range of competition-level problems.

AIME 2024. Our AIME 2024 evaluation set combines the 15 problems from AIME I and the 15 problems from AIME II, yielding 30 problems in total. These problems cover areas such as algebra, geometry, number theory, counting, and probability and generally require multi-step reasoning and nontrivial mathematical insight. Each problem has a unique integer answer between 000 and 999, allowing reliable rulebased evaluation. We use AIME 2024 to assess reasoning performance on challenging competition-level mathematics problems. The general AIME format consists of 15 questions per examination, each requiring an integer answer from 000 to 999.

AIME 2025. Our AIME 2025 evaluation set similarly combines AIME I and AIME II, resulting in 30 problems. Each problem requires an integer answer between 000 and 999 and involves advanced high-school mathematics across multiple domains. Since these problems were released more recently than most commonly used mathematical reasoning benchmarks, AIME 2025 provides a challenging evaluation of the models’ reasoning and generalization capabilities.

OlympiadBench. OlympiadBench is a bilingual and multimodal scientific reasoning benchmark containing 8,476 problems collected from international olympiads, Chinese olympiads, and the Chinese college entrance examination. The final ACL 2024 version contains 6,142 mathematics problems and 2,334 physics problems, with every problem accompanied by an expert-annotated solution. The benchmark includes both open-ended problems and theoremproving problems and covers English and Chinese questions with textual or visual information. Following the evaluation protocol of GVM and CurES, we use the corresponding mathematics portion of OlympiadBench to evaluate reasoning performance on challenging olympiad-level problems.

## C. Significance Analysis of Main Results Question and Comparison Units

The main experiments report one Average@8 point estimate for each scheduler–base-model–benchmark combination. We use a two-sided exact sign test (Dixon and Mood 1946) to examine whether the observed improvements are directionally consistent across these heterogeneous experimental settings. This analysis directly evaluates whether integrating our method produces positive changes more frequently than would be expected under an equal-probability null hypothesis.

The analysis includes the three full integrations: GVM + Ours, PCL + Ours, and GRESO + Ours.

For integration method m, base model b, and benchmark d, define

$$
\Delta _ { m , b , d } = \mathrm { A c c } _ { m + \mathrm { O u r s } , b , d } - \mathrm { A c c } _ { m , b , d } .\tag{42}
$$

We record a gain if $\Delta _ { m , b , d } > 0$ , a loss if $\Delta _ { m , b , d } < 0 .$ , and a tie if $\Delta _ { m , b , d } = 0$ . Let W, $L ,$ and T denote the resulting numbers of gains, losses, and ties, respectively. Because ties provide no directional information, they are excluded from the sign-test sample size, yielding $n = W + L$

## Exact Test and Results

Under the null hypothesis that a non-tied comparison is equally likely to favor either method,

$$
H _ { 0 } : \operatorname* { P r } ( \Delta > 0 \mid \Delta \neq 0 ) = \frac { 1 } { 2 } , \qquad W \sim \operatorname { B i n o m i a l } ( n , \textstyle { \frac { 1 } { 2 } } ) .\tag{43}
$$

The two-sided exact p-value is

$$
p _ { \mathrm { s i g n } } = \operatorname* { m i n } \left\{ 1 , 2 \sum _ { r = 0 } ^ { \operatorname* { m i n } ( W , L ) } \binom { n } { r } 2 ^ { - n } \right\} .\tag{44}
$$

Counting directly from Table 1 gives the following results. For the PCL integration, $W = 7 , L = 1$ , and $n = 8$ . Its two-sided exact $p \cdot$ -value is

$$
{ \begin{array} { r l } & { p _ { \mathrm { s i g n } } = 2 \left[ { \binom { 8 } { 0 } } + { \binom { 8 } { 1 } } \right] 2 ^ { - 8 } } \\ & { \qquad = { \frac { 1 8 } { 2 5 6 } } = 0 . 0 7 0 3 1 2 5 . } \end{array} }\tag{45}
$$

<table><tr><td colspan="2">Full integration W LT</td><td> $\underline { { p _ { \mathrm { s i g n } } } }$ </td></tr><tr><td> $\mathbf { \overline { { G V M + O u r s } } }$ </td><td>7 0 1</td><td>0.0156</td></tr><tr><td> $\mathrm { P C L } + \mathrm { O u r s }$  GRESO + Ours</td><td>7 1 0 7 0 1</td><td>0.0703</td></tr><tr><td>Pooled</td><td>21 1 2 </td><td>0.0156  $\overline { { 1 . 1 0 \times 1 0 ^ { - 5 } } }$ </td></tr></table>

Table 4: Directional sign-test results for the full integrations. Ties are excluded from the exact test.

At the nominal significance level $\alpha = 0 . 1 0$ , all three full integrations exhibit statistically significant directional improvements. Specifically, GVM + Ours and GRESO + Ours achieve significance at the $\alpha = 0 . 0 5$ level, while PCL + Ours achieves significance at the $\alpha = 0 . 1 0$ level. These results demonstrate that the positive efect of our method is not confined to a particular scheduler, base model, or benchmark, but is consistently observed across the evaluated configurations.

For the pooled analysis, W = 21, L = 1, and $n = 2 2$ Equation (44) therefore gives

$$
{ \begin{array} { r l } & { p _ { \mathrm { s i g n } } = 2 \left[ { \binom { 2 2 } { 0 } } + { \binom { 2 2 } { 1 } } \right] 2 ^ { - 2 2 } } \\ & { \qquad = { \frac { 4 6 } { 4 , 1 9 4 , 3 0 4 } } = 1 . 0 9 6 7 \times 1 0 ^ { - 5 } . } \end{array} }\tag{46}
$$

The pooled result provides strong evidence against the null hypothesis that positive and negative changes are equally likely. Across the three full integrations, our method improves 21 of the 22 non-tied experimental comparisons, corresponding to a gain rate of 95.5%. The exact sign test therefore confirms a highly consistent positive improvement direction across the evaluated schedulers, base models, and benchmarks. This cross-setting consistency is particularly valuable because it shows that the benefit of our method generalizes across diferent integration strategies and experimental conditions rather than depending on a single favorable configuration.

## D. Cross-Domain Generalization to Code

<table><tr><td rowspan="2">Base Model</td><td rowspan="2">Methods</td><td colspan="2">Full Steps</td></tr><tr><td>MAE (↓)</td><td>r (↑)</td></tr><tr><td rowspan="3">Qwen2.5-Coder-7B</td><td>VIP</td><td>0.197</td><td>0.504</td></tr><tr><td>PCL</td><td>0.402</td><td>0.307</td></tr><tr><td>MoPPS</td><td>0.224</td><td>0.773</td></tr><tr><td>Instruct</td><td>Ours</td><td>0.176</td><td>0.845</td></tr></table>

Table 5: Dificulty-estimation performance on code generation using Qwen2.5-Coder-7B-Instruct. MAE is samplelevel, and r is the batch-level Pearson correlation.

## Setting

Many existing adaptive RL scheduling methods are developed and evaluated primarily for mathematical reasoning. Extending them to code generation may require domainspecific modifications to reward definitions, sampling strategies, or hyperparameters. Reimplementing these adaptations may result in unequal levels of optimization across baselines and compromise the fairness of the comparison. Therefore, rather than reproducing all scheduling baselines in the code domain, we examine whether our dificulty estimator can be directly transferred to code generation and track changes in sample dificulty during RL training.

We conduct RL training on LiveCodeBench (Jain et al. 2024) release\_v2, which contains 511 programming problems, using Qwen2.5-Coder-7B-Instruct. LiveCodeBench collects problems from LeetCode, AtCoder, and Codeforces and provides platform-derived dificulty ratings and executable test cases (Jain et al. 2024). A generated program is considered correct only if it passes all corresponding test cases.

Consistent with the main experiments, we generate eight rollouts for each selected sample and use the proportion of correct programs as its empirical success rate. We then replay the estimation process over the complete training trajectory. Before each training step, the estimator predicts sample success probabilities using only rollout feedback available from previous steps, and these predictions are compared with the subsequently observed outcomes.

## Results

As shown in Table 5, our method consistently tracks the evolving success probabilities of training samples throughout the RL trajectory of Qwen2.5-Coder-7B-Instruct. This result indicates that the proposed estimator does not depend on the exact-match reward commonly used in mathematical reasoning tasks. When the feedback is replaced by program correctness determined through test-case execution, the sample graph and historical rollout outcomes still provide efective information for estimating dynamic sample dificulty.

Overall, the experiment provides preliminary evidence that our estimator can be transferred to execution-based code generation without modifications specific to the code domain.

## Discussion

The practical need for our estimator may be weaker in code generation. Some competitive-programming benchmarks provide platform dificulty ratings and execution feedback, which may already sufice for curriculum learning or rollout allocation in certain settings. This may partly explain the stronger focus of related studies on mathematics. Thus, this experiment establishes transferability rather than a clear advantage in code generation.

## E. Hyper Parameter Analysis

## Guideline of K, $k _ { k n n }$ and $\beta$

We discuss three hyperparameters controlling our dificulty estimator. K is the number of latent dificulty states, determining modeling granularity; $k _ { \mathrm { n n } }$ is the neighborhood size for graph construction, controlling feedback propagation range; β is the Potts prior strength, balancing graph smoothing against adaptation to policy changes.

Firstly, we analyze the influence of hyperparameters β and K on dificulty estimation, as shown in Table 6, where MAE is sample-level and r is the batch-level Pearson correlation. When β increases from 2 to 5 or above, all metrics become nearly identical and slightly worse, indicating that an overstrong Potts prior makes the model rely too heavily on neighbor information and neglect the global dificulty structure. In contrast, a smaller K slightly improves the overall correlation r, suggesting that fewer latent states better capture the global dificulty trend, but the sample-level MAE increases, degrading per-sample accuracy. Considering this trade-of, we choose $\beta = 2$ and $K = 3 2 { \dot { 0 } }$ in the main experiments.

<table><tr><td rowspan="2"> $\beta$ </td><td rowspan="2">K</td><td rowspan="2">Early Steps r (↑)</td><td colspan="2">Full Steps</td></tr><tr><td>AE (↓) r (↑)</td><td>MAE (↓)</td></tr><tr><td>2</td><td>320</td><td>0.482</td><td>0.290</td><td>0.836 0.183</td></tr><tr><td>2</td><td>160</td><td>0.480</td><td>0.306</td><td>0.838 0.191</td></tr><tr><td>2</td><td>80</td><td>0.494</td><td>0.313</td><td>0.843 0.194</td></tr><tr><td>2</td><td>40</td><td>0.507</td><td>0.327</td><td>0.848 0.196</td></tr><tr><td>1</td><td>320</td><td>0.488</td><td>0.302</td><td>0.849 0.189</td></tr><tr><td>5</td><td>320</td><td>0.494</td><td>0.297</td><td>0.847 0.186</td></tr><tr><td>10</td><td>320</td><td>0.490</td><td>0.297</td><td>0.847 0.186</td></tr><tr><td>20</td><td>320</td><td>0.493</td><td>0.298</td><td>0.847 0.187</td></tr></table>

Table 6: Sensitivity analysis of β and K on Qwen-2.5-Math-1.5B. MAE is sample-level, and r is batch-level Pearson correlation.

The neighborhood size $k _ { \mathrm { n n } }$ determines the initial connectivity of the dificulty-aware sample graph, and thereby controls how far rollout feedback can propagate across samples. Experiments show that as long as $k _ { \mathrm { n n } }$ is not set extremely small (e.g., below 10), the estimation accuracy is barely affected: after reciprocal-neighbor and positive-similarity filtering, the graph itselfis already suficiently sparse, so a small number of neighbors sufices for efective feedback propagation. Meanwhile, the ablation study shows that removing graph sparsification leads to only a modest performance drop, indicating that excessively increasing the neighborhood size brings no additional benefit. From a computational perspective, increasing $k _ { \mathrm { n n } }$ raises the cost of message propagation on the graph—in each variational inference iteration, updating the latent-state distribution of each sample requires aggregating information from its neighbors, giving a theoretical complexity of $O ( N k _ { \mathrm { n n } } )$ . However, due to random memory access on sparse graphs and edge saturation caused by positive-similarity filtering, the actual wall-clock time does not grow linearly with $k _ { \mathrm { n n } } ^ { 3 }$

## Discussion of Embedding Models and Dificulty-Aware Instruction

Models Selection We compare several recent textembedding models that support task-aware or instruction-

<sup>3</sup>Theoretically, the complexity grows linearly with $k _ { \mathrm { n n } } .$ , but in practice, graph sparsity leads to random memory access, and positive-similarity filtering gradually saturates the actual edge count, so the time does not increase linearly even if $k _ { \mathrm { n n } }$ is increased. In our setting, the wall-clock time when the edge count saturates is about 1.61× that of the proposed method in the main text.

conditioned embedding, including Qwen3-Embedding-0.6B, Qwen3-Embedding-4B (Zhang et al. 2025), and BGE-M3 (Chen et al. 2024). When their standard dense representations are used to construct the sample graph, the resulting performance is broadly similar (see Table 7), suggesting that our framework is not sensitive to the exact choice among recent embedding models. Among the direct, uncompressed configurations, Qwen3-Embedding-0.6B performs on par with the larger Qwen3-Embedding-4B and BGE-M3 on Qwen-2.5-Math-1.5B (0.183 vs. 0.188 vs. 0.188 MAE), while on Llama-3.2-1B-Instruct it is slightly worse than Qwen3-Embedding-4B (0.131 vs. 0.130). Given that Qwen3- Embedding-0.6B is substantially more eficient, we adopt it as the default encoder as a trade-of between accuracy and computational cost.

Model scale and raw embedding dimensionality do not exhibit a monotonic relationship with downstream performance. In particular, compressing the Qwen3-Embedding-4B output to 1,024 dimensions using its Matryoshkacompatible representation (Kusupati et al. 2022) improves its performance relative to using the full-dimensional representation (0.179 vs. 0.188 MAE on Qwen; 0.121 vs. 0.130 on Llama, as shown in Table 7). Notably, the MRL-compressed Qwen3-Embedding-4B at 1,024 dimensions achieves the best overall results on both base models (0.179/0.844 for Qwen and 0.121/0.787 for Llama), suggesting that a larger encoder combined with dimensionality reduction can further improve graph quality if the additional computational cost is acceptable. Nevertheless, for the main experiments we prioritize low overhead and therefore use Qwen3-Embedding-0.6B as the default.

A plausible explanation for the benefit of MRL compression is the curse of dimensionality: in an excessively high-dimensional space, pairwise distances can become less discriminative, which may make nearest-neighbor selection noisier and weaken the local graph structure. Matryoshka Representation Learning (MRL) preserves information at nested dimensionalities, allowing the representation to be shortened without applying an arbitrary post-hoc projection. Our observation is consistent with this explanation, although it does not by itself establish dimensionality as the sole causal factor.

In contrast, replacing the recent instruction-aware encoders with earlier Sentence-BERT-family models (Reimers and Gurevych 2019), or with the encoder configuration adopted by VIP (Nguyen et al. 2026a), causes a substantial performance drop. These older representations were primarily optimized for general sentence-level similarity and appear less suitable for capturing the task-specific semantic, reasoning, and dificulty cues required by our graph. Based on these results, we recommend pairing our method with a recent task-aware embedding model. Qwen3-Embedding-0.6B provides a strong default eficiency–performance trade-of, while a larger MRL-compatible encoder truncated to a moderate dimension is a viable alternative.

Discussion of Dificulty-Aware Instruction Table 7 shows that removing the dificulty-aware instruction from Qwen3-Embedding-0.6B causes a substantial degradation in estimation quality. On Qwen-2.5-Math-1.5B, the MAE increases from 0.183 to 0.309 and the correlation drops from 0.836 to 0.583; on Llama-3.2-1B-Instruct, the MAE increases from 0.131 to 0.302 and the correlation drops from 0.776 to 0.486. This confirms that the instruction, which encourages the encoder to capture dificulty-related and reasoning-oriented features beyond pure surface semantics, contributes critically to the graph quality. Without it, the embeddings tend to connect samples that are semantically similar but not necessarily informative for dificulty transfer.

<table><tr><td rowspan="2">Base Model</td><td rowspan="2">Methods</td><td colspan="2">Full Steps</td></tr><tr><td>MAE (↓)</td><td>r (↑)</td></tr><tr><td rowspan="6">Qwen-2.5 Math-1.5B</td><td>Qwen3-Embedding-0.6B</td><td>0.183</td><td>0.836</td></tr><tr><td>-Difficulty-Aware Instruction</td><td>0.309</td><td>0.583</td></tr><tr><td>Qwen3-Embedding-4B</td><td>0.188</td><td>0.827</td></tr><tr><td>Qwen3-Embedding-4B MRL(1024 dimensions)</td><td>0.179</td><td>0.844</td></tr><tr><td>BGE-M3</td><td>0.188</td><td>0.823</td></tr><tr><td>MiniLM</td><td>0.326</td><td>0.407</td></tr><tr><td rowspan="6">Llama-3.2 1B-Instruct</td><td>Qwen3-Embedding-0.6B</td><td>0.131</td><td>0.776</td></tr><tr><td>-Difficulty-Aware Instruction</td><td>0.302</td><td>0.486</td></tr><tr><td>Qwen3-Embedding-4B</td><td>0.130</td><td>0.749</td></tr><tr><td>Qwen3-Embedding-4B MRL(1024 dimensions)</td><td>0.121</td><td>0.787</td></tr><tr><td>BGE-M3</td><td>0.139</td><td>0.783</td></tr><tr><td>MiniLM</td><td>0.304</td><td>0.407</td></tr></table>

Table 7: Full-trajectory dificulty-estimation performance of diferent embedding configurations. The row “-Dificulty-Aware Instruction” removes the instruction from the default encoder. MAE: sample-level mean absolute error (lower is better); r: batch-level Pearson correlation (higher is better).

The comparison between Qwen3-Embedding-4B and its MRL-truncated 1,024-dimension variant also suggests that reducing dimensionality, when done through a representation-learning-compatible method, can improve the discriminability of nearest-neighbor structure. BGE-M3 achieves comparable performance to the Qwen3 variants, while MiniLM performs poorly, which is consistent with the above observation that older generic encoders lack the taskawareness needed for constructing a dificulty-aware graph.

## Discussion of the Interpolation Weight γ

For samples with prior rollouts, we combine the graphbased prediction and the sample-level empirical success rate $\mathrm { a s } \widehat { p _ { i , t } } \stackrel { - } { = } \gamma \widehat { p } _ { i , t } ^ { \mathrm { m o d e l } } + ( 1 - \gamma ) \bar { \bar { r } } _ { i , < t }$ The parameter γ controls the balance between transferable graph-based evidence and direct sample-level history. A small γ makes the estimate dominated by $( \bar { r } _ { i , < t } .$ , which weakens graph propagation and can hurt dificulty ranking. A large γ relies more on the latent-state model and may ignore reliable direct observations, leading to larger estimation error.

We set $\gamma = 0 . 5$ as a general default. In our experiments, performance is relatively stable for $\gamma \in [ 0 . 5 , 0 . 8 ]$ . When $\gamma <$ 0.5, ranking quality tends to decrease; when $\gamma > 0 . 8$ , samplelevel estimation error usually increases. This suggests that a moderate interpolation weight is preferable.

A more principled choice would adapt γ according to uncertainty. The uncertainty of latent state (k) can be estimated from the variance of its Beta posterior:

$$
v _ { k , t - 1 } = { \frac { a _ { k , t - 1 } b _ { k , t - 1 } } { ( a _ { k , t - 1 } + b _ { k , t - 1 } ) ^ { 2 } ( a _ { k , t - 1 } + b _ { k , t - 1 } + 1 ) } } .
$$

Combined with assignment uncertainty, this gives

$$
\widehat { u } _ { i , t } = \sum _ { k = 1 } ^ { K } q _ { i k , t - 1 } \left[ v _ { k , t - 1 } + \left( \mu _ { k , t - 1 } - \widehat { p } _ { i , t } ^ { \mathrm { m o d e l } } \right) ^ { 2 } \right] .
$$

In principle, one could increase $\gamma$ when the model-based prediction is confident, and decrease it when the sample has suficient direct observations. However, such a rule also depends on rollout counts, sampling distributions, and the downstream scheduler. These factors vary across integrated frameworks and are hard to fix with limited experiments. We therefore use a fixed $\gamma = 0 . 5$

## F. Discussion of Spectral Clusters

## Association with Subject Categories

For each analyzed MATH problem i, let $z _ { i } \in \{ 1 , \ldots , K _ { + } \}$ denote its nonempty spectral-cluster assignment and let $c _ { i } \in$ $\{ 1 , \ldots , U \}$ denote its subject category. We form the $K _ { + } \times U$ contingency table

$$
O _ { k u } = \sum _ { i = 1 } ^ { n } \mathbb { I } ( z _ { i } = k , c _ { i } = u ) ,\tag{47}
$$

with row totals $O _ { k } .$ , column totals $O _ { \cdot u }$ , and total sample size n. Under the null hypothesis that cluster assignment and subject category are independent, the expected count in cell $( k , u )$ is

$$
E _ { k u } = \frac { O _ { k } . O _ { \cdot u } } { n } .\tag{48}
$$

We measure departure from independence using Pearson’s chi-squared statistic (Pearson 1900),

$$
\chi ^ { 2 } = \sum _ { k = 1 } ^ { K _ { + } } \sum _ { u = 1 } ^ { U } \frac { ( O _ { k u } - E _ { k u } ) ^ { 2 } } { E _ { k u } } ,\tag{49}
$$

whose asymptotic null distribution has $( K _ { + } - 1 ) ( U - 1 )$ degrees of freedom. Applying this calculation to the MATH training split gives $n = 7 , 5 0 0 , K _ { + } = 1 6 ,$ and $U = 7 ;$ thus, the contingency table contains $1 6 \times 7$ cells and the asymptotic reference distribution has 90 degrees of freedom. The resulting statistic is $\chi ^ { 2 } = 1 0 , 9 9 5 . 8 4$ , with asymptotic $p < 1 0 ^ { - 3 0 0 }$ , rejecting independence.

Because the chi-squared statistic increases with sample size, we additionally report Cramér’s V, a normalized efectsize measure for categorical association (Cramér 1946):

$$
V = \sqrt { \frac { \chi ^ { 2 } } { n \operatorname* { m i n } ( K _ { + } - 1 , U - 1 ) } } .\tag{50}
$$

The resulting $V = 0 . 4 9 4$ indicates a substantial association between the spectral clusters and the annotated mathematical subjects. This efect-size analysis complements the hypothesis test: the p-value establishes that the association is unlikely under independence, whereas V quantifies its magnitude.

## Dificulty Diferences Within Subject Categories

Subject category and cluster assignment are strongly associated, so a single pooled comparison could confound subject composition with dificulty. We therefore conduct seven separate Kruskal–Wallis tests, one within each MATH subject category (Kruskal and Wallis 1952). This stratification asks whether the annotated dificulty levels difer across spectral clusters among problems from the same subject.

For subject u, let $n _ { u }$ be its number of problems and $n _ { k u }$ the number assigned to nonempty cluster k. We rank the ordinal dificulty levels of all $n _ { u }$ problems within that subject, assigning midranks to ties. Let $R _ { k u }$ be the sum of ranks in cluster k. The uncorrected Kruskal–Wallis statistic is

$$
H _ { u } ^ { ( 0 ) } = \frac { 1 2 } { n _ { u } ( n _ { u } + 1 ) } \sum _ { k : n _ { k u } > 0 } \frac { R _ { k u } ^ { 2 } } { n _ { k u } } - 3 ( n _ { u } + 1 ) .\tag{51}
$$

Because MATH dificulty takes only five ordinal levels, ties are frequent. Let $t _ { u g }$ be the size of the g-th tied group within subject u. We apply the standard ties correction

$$
C _ { u } = 1 - \frac { \sum _ { g } ( t _ { u g } ^ { 3 } - t _ { u g } ) } { n _ { u } ^ { 3 } - n _ { u } } , \qquad H _ { u } = \frac { H _ { u } ^ { ( 0 ) } } { C _ { u } } .\tag{52}
$$

If $K _ { u }$ clusters are nonempty within subject u, the raw asymptotic p-value is computed from a chi-squared reference distribution with $K _ { u } - 1$ degrees of freedom:

$$
p _ { u } = \mathrm { P r } \left( \chi _ { K _ { u } - 1 } ^ { 2 } \geq H _ { u } \right) .\tag{53}
$$

The seven raw p-values form one family of hypotheses. We therefore apply the Benjamini–Hochberg procedure to control the false discovery rate (Benjamini and Hochberg 1995). After sorting the raw values as $p _ { ( 1 ) } \leq \dots \leq p _ { ( m ) }$ where $m = 7$ , the monotone adjusted values are

$$
\widetilde { p } _ { ( r ) } = \operatorname* { m i n } \left\{ 1 , \operatorname* { m i n } _ { j \geq r } \frac { m } { j } p _ { ( j ) } \right\} .\tag{54}
$$

<table><tr><td>Subject</td><td>Raw p</td><td>BH-adjusted p</td></tr><tr><td>Algebra</td><td>≈0</td><td>≈0</td></tr><tr><td>Prealgebra</td><td>≈0</td><td>≈0</td></tr><tr><td>Precalculus</td><td> $\approx 0$ </td><td> $1 . 0 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Intermediate algebra</td><td> $2 . 0 \times 1 0 ^ { - 6 }$ </td><td> $3 . 0 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Counting/probability</td><td> $8 . 0 \times 1 0 ^ { - 6 }$ </td><td> $1 . 1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Number theory</td><td> $3 . 1 9 \times 1 0 ^ { - 4 }$ </td><td> $3 . 7 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Geometry</td><td> $4 . 3 1 \times 1 0 ^ { - 3 }$ </td><td> $4 . 3 1 \times 1 0 ^ { - 3 }$ </td></tr></table>

Table 8: Within-subject Kruskal–Wallis tests of dificulty diferences across spectral clusters. Values displayed as zero are below the reporting precision; BH correction uses the unrounded raw values.

All seven subject-specific null hypotheses remain rejected after BH correction, with $\widetilde { p } _ { u } \le 4 . 3 \dot { 1 } \times 1 0 ^ { - 3 }$ . Therefore, cluster membership is associated with annotated dificulty even among problems from the same subject category. Together with the cluster–subject association, this result supports the interpretation that the spectral initialization captures both semantic and dificulty-related structure.

## G. Windowed Memory under Non-stationary Policies

## Windowed Forgetting Mechanism

The online inference procedure described in the main text accumulates rollout feedback across training steps by inheriting the state-level Beta posterior, i.e., $a _ { k , t } ^ { \mathrm { { \bar { h i s t } } } } \stackrel { - } { = } { a _ { k , t - 1 } }$ and $b _ { k , t } ^ { \mathrm { h i s t } } ~ = ~ b _ { k , t - 1 }$ . However, the rollout distribution is non-stationary because the policy parameters continuously change during RL optimization. Consequently, rollout observations collected many training steps earlier may no longer accurately characterize the success probability of the current policy. To reduce the influence of such outdated observations while remaining responsive to the current policy, we introduce a windowed forgetting mechanism. The windowed variant preserves the dificulty-aware sample graph W, the smoothed assignment prior $A ,$ the Potts graph prior, and the mean-field variational family introduced in the main method. It only changes how historical rollout evidence is organized: state-level success probabilities are estimated from a recent sliding window, whereas the complete rollout history of each previously observed sample is retained for sample-level prediction.

Windowed State-Level Statistics. Before predicting training step t, we define the preceding window of length $L \geq 1$ as

$$
\mathcal { T } _ { t } ^ { ( L ) } = \{ \tau : \operatorname* { m a x } ( 1 , t - L ) \leq \tau < t \} ,\tag{55}
$$

where L denotes the window length and is distinguished from the graph adjacency matrix W. For latent state k, the responsibility-weighted numbers of successful and unsuccessful rollouts within this window are

$$
S _ { k , t } ^ { ( L ) } = \sum _ { \tau \in \mathcal { T } _ { t } ^ { ( L ) } } \sum _ { i \in \mathcal { O } _ { \tau } } q _ { i k , \tau } s _ { i , \tau } ,\tag{56}
$$

and

$$
F _ { k , t } ^ { ( L ) } = \sum _ { \tau \in \mathcal { T } _ { t } ^ { ( L ) } } \sum _ { i \in \mathcal { O } _ { \tau } } q _ { i k , \tau } \left( n _ { i , \tau } - s _ { i , \tau } \right) .\tag{57}
$$

Rather than directly inheriting the complete Beta posterior from the previous step, the windowed variant reconstructs the state-level Beta parameters from a fixed prior and the rollout evidence within the current window:

$$
a _ { k , t } ^ { ( L ) } = a _ { 0 } + S _ { k , t } ^ { ( L ) } , \qquad b _ { k , t } ^ { ( L ) } = b _ { 0 } + F _ { k , t } ^ { ( L ) } .\tag{58}
$$

In our implementation, we set $a _ { 0 } = b _ { 0 } = 1$ , corresponding to the uniform prior Beta(1, 1). This is a standard weak prior for a Bernoulli success probability and does not favor any particular success rate before rollout evidence is observed. It also guarantees that the Beta parameters remain strictly positive. This property is particularly important under a finite window because some latent states may receive no rollout observations in the current window. When $S _ { k , t } ^ { ( L ) } = F _ { k , t } ^ { ( L ) } = 0 .$ the Beta(1, 1) prior still yields a valid posterior and keeps the posterior mean and the digamma terms used in the variational updates well-defined and numerically stable.

Ideally, the prior mean

$$
\mathbb { E } [ \theta _ { k , t } ] = \frac { a _ { 0 } } { a _ { 0 } + b _ { 0 } }\tag{59}
$$

should be close to the average success probability of the current policy. Such a prior would provide a more informative default estimate for latent states with few or no recent observations. However, obtaining a reliable estimate of the current policy’s average ability would require evaluating a suficiently large and representative subset of the training set at each stage, which would introduce substantial additional rollout cost. Estimating it only from the samples selected by the scheduler would also be potentially biased, because the selected samples are generally not representative of the full training distribution. We therefore adopt Beta(1, 1) as a practical trade-of among prior neutrality, numerical stability, and computational eficiency.

The posterior mean success probability of latent state k is then

$$
\mu _ { k , t } ^ { ( L ) } = \frac { a _ { k , t } ^ { ( L ) } } { a _ { k , t } ^ { ( L ) } + b _ { k , t } ^ { ( L ) } } .\tag{60}
$$

The model-based prediction for sample $x _ { i }$ retains the same mixture form as in the main method:

$$
\widehat { p } _ { i , t } ^ { \mathrm { m o d e l } } = \sum _ { k = 1 } ^ { K } q _ { i k , t } \mu _ { k , t } ^ { ( L ) } .\tag{61}
$$

When the window covers all preceding training steps, the windowed formulation reduces to the cumulative-history variant. With a finite $L ,$ rollout observations older than $L$ training steps no longer directly contribute to the current state-level success probabilities.

Windowed Variational Updates. The windowed variant retains the mean-field factorization

$$
q _ { t } ( \mathbf { z } _ { t } , \pmb { \theta } _ { t } ) \approx \prod _ { i = 1 } ^ { N } q _ { i } ( z _ { i , t } ) \prod _ { k = 1 } ^ { K } q ( \theta _ { k , t } ) ,\tag{62}
$$

where $q _ { i k , t } \equiv q _ { i } ( z _ { i , t } = k )$ and

$$
q ( \theta _ { k , t } ) = \mathrm { B e t a } \left( a _ { k , t } ^ { ( L ) } , b _ { k , t } ^ { ( L ) } \right) .\tag{63}
$$

The coordinate update for the latent-state assignments preserves the same three sources of information as the main method: $q _ { i k } ^ { ( r + 1 ) }$ = Softmax<sub>k</sub> $\begin{array} { r } { \left[ \log A _ { i k } + \beta \sum _ { j = 1 } ^ { N } W _ { i j } q _ { j k } ^ { ( r ) } + \mathbb { I } \left( i \in \mathcal { O } _ { t } ^ { ( L ) } \right) \ell _ { i k , t } ^ { ( L ) } \right] } \end{array}$ where

$$
\mathcal { O } _ { t } ^ { ( L ) } = \bigcup _ { \tau \in \mathcal { T } _ { t } ^ { ( L ) } } \mathcal { O } _ { \tau } .\tag{64}
$$

For each sample $i ,$ we aggregate its rollout outcomes within the current window as

$$
s _ { i , t } ^ { ( L ) } = \sum _ { \tau \in \mathcal { T } _ { t } ^ { ( L ) } } s _ { i , \tau } , \qquad n _ { i , t } ^ { ( L ) } = \sum _ { \tau \in \mathcal { T } _ { t } ^ { ( L ) } } n _ { i , \tau } ,\tag{65}
$$

and compute the corresponding expected log-likelihood as $\begin{array} { r l r } { \ell _ { i k , t } ^ { ( L ) } } & { { } = } & { s _ { i , t } ^ { ( L ) } \left\lceil \psi \left( a _ { k , t } ^ { ( L ) } \right) - \psi \left( a _ { k , t } ^ { ( L ) } + \bar { b } _ { k , t } ^ { ( L ) } \right) \right\rceil + } \end{array}$ $\left( n _ { i , t } ^ { ( L ) } - s _ { i , t } ^ { ( L ) } \right) \left\lceil \psi \left( b _ { k , t } ^ { ( L ) } \right) ^ { \sim } - \psi \left( a _ { k , t } ^ { ( L ) } + b _ { k , t } ^ { ( L ) } \right) \right\rceil$

After updating the assignment probabilities, the state-level Beta factors are recomputed using only the rollout observations within the current window:

$$
a _ { k , t } ^ { ( L ) } = 1 + \sum _ { \tau \in \mathcal { T } _ { t } ^ { ( L ) } } \sum _ { i \in \mathcal { O } _ { \tau } } q _ { i k } s _ { i , \tau } ,\tag{66}
$$

and

$$
b _ { k , t } ^ { ( L ) } = 1 + \sum _ { \tau \in \mathcal { T } _ { t } ^ { ( L ) } } \sum _ { i \in \mathcal { O } _ { \tau } } q _ { i k } \left( n _ { i , \tau } - s _ { i , \tau } \right) .\tag{67}
$$

We alternate the assignment and Beta-factor updates using the same stopping criterion as in the main method. Therefore, the windowed variant does not change the coordinate-ascent structure of the inference procedure; it only restricts the temporal support of the rollout evidence used in the likelihood and state-level Beta updates.

Persistent Assignment Warm Start. In the cumulative formulation, the complete estimator state $S _ { t } = \{ q _ { t } , a _ { t } , b _ { t } \}$ is passed between consecutive training steps. In the windowed variant, we retain only the previous assignment probabilities as the default initialization:

$$
q _ { i k , t } ^ { ( 0 ) } = q _ { i k , t - 1 } .\tag{68}
$$

The Beta parameters are not directly inherited from the previous step. Instead, they are reconstructed from the fixed Beta(1, 1) prior and the rollout observations within the current window. The persistent initialization of q preserves continuity in the graph-structured latent assignments and provides a warm start for the finite-iteration variational procedure. Importantly, the inherited responsibilities are used only as an optimization initialization. They are not treated as additional successes or failures and therefore do not alter the strict temporal window applied to the state-level suficient statistics. We additionally consider a reset variant that initializes q from the static soft assignment prior A at every step; unless otherwise specified, the persistent warm-start variant is used by default.

Persistent Sample-Level History. The sliding window is applied only to the shared state-level statistics. For each sample x<sub>i</sub>, we separately retain its complete rollout history before step t:

$$
S _ { i , t } ^ { \mathrm { a l l } } = \sum _ { \tau < t } s _ { i , \tau } , \qquad C _ { i , t } ^ { \mathrm { a l l } } = \sum _ { \tau < t } n _ { i , \tau } .\tag{69}
$$

For a previously observed sample, its historical empirical success rate is

$$
\overline { { r } } _ { i , < t } = \frac { S _ { i , t } ^ { \mathrm { a l l } } } { C _ { i , t } ^ { \mathrm { a l l } } } .\tag{70}
$$

We then retain the prediction rule from the main method:

$$
\widehat { p } _ { i , t } = \left. \begin{array} { l l } { \widehat { p } _ { i , t } ^ { \mathrm { m o d e l } } , } & { C _ { i , t } ^ { \mathrm { a l l } } = 0 , } \\ { \gamma \widehat { p } _ { i , t } ^ { \mathrm { m o d e l } } + ( 1 - \gamma ) \overline { { r } } _ { i , < t } , } & { C _ { i , t } ^ { \mathrm { a l l } } > 0 . } \end{array} \right.\tag{71}
$$

Consequently, the two components of the final prediction operate at complementary time scales. The model-based term uses recent rollout feedback to track the evolving success probabilities of the shared latent states, whereas the samplelevel empirical term preserves all available direct evidence for each previously observed sample. The windowed mechanism therefore removes stale evidence from transferable state-level statistics without discarding the long-term historical experience associated with individual samples.

## Result and Discussion

<table><tr><td rowspan="2">Base Model</td><td rowspan="2">Methods</td><td colspan="2">Full Steps</td></tr><tr><td>MAE (↓)</td><td>r (↑)</td></tr><tr><td>Qwen-2.5</td><td>Ours</td><td>0.183</td><td>0.836</td></tr><tr><td>Math-1.5B</td><td>Windows</td><td>0.127</td><td>0.712</td></tr><tr><td>Llama-3.2</td><td>Ours</td><td>0.131</td><td>0.776</td></tr><tr><td>1B-Instruct</td><td>Windows</td><td>0.119</td><td>0.747</td></tr></table>

Table 9: Sample-level dificulty estimation performance of the cumulative estimator (Ours) and the windowed variant (Windows) over all training steps. MAE measures the absolute prediction error, while batch-level r denotes the Pearson correlation between the predicted and reference success probabilities. Lower MAE and higher r indicate better performance.

Batch-level reference mean accuracy and predicted accuracy  
![](images/7e2777e9e7979abad58c460b02269fb0ba6febd6166f973dcb4c83ed8f8ad38a.jpg)  
Figure 3: Batch-level reference accuracy and predicted accuracy of the cumulative and windowed estimators on Qwen-2.5-Math-1.5B and Llama-3.2-1B-Instruct.

We evaluate the windowed variant by replaying the full training logs of Qwen-2.5-Math-1.5B and Llama-3.2-1B-Instruct, and compare it with the cumulative estimator used in our main experiments. As shown in Table 9, the windowed variant achieves lower sample-level MAE on both models: from 0.183 to 0.127 on Qwen-2.5-Math-1.5B, and from 0.131 to 0.119 on Llama-3.2-1B-Instruct. This indicates that windowed forgetting makes the predicted success probability of each individual sample numerically closer to its reference rollout outcome. Fig. 3 further visualizes the batch-level reference accuracy and predicted accuracy over the full training process.

However, Table 9 also shows that the correlation metric r decreases after introducing the windowed mechanism: from 0.836 to 0.712 on Qwen-2.5-Math-1.5B, and from 0.776 to 0.747 on Llama-3.2-1B-Instruct. This suggests a tradeof: although the windowed estimator improves point-wise numerical accuracy, it weakens the ability to preserve the relative dificulty structure among samples. A likely reason is that the finite window reduces the amount of historical evidence available for state-level estimation, making the estimates more sensitive to sparse recent observations and sampling bias. For our dificulty-aware scheduling setting, stable relative ordering is more important than point-wise calibration alone. Therefore, we retain the cumulative estimator as the default method in the main experiments. Whether to use windowed forgetting should depend on the downstream objective: it is beneficial when current-policy calibration is prioritized, but less suitable when stable sample ranking is critical.

## H. Related work of Graph-Based Learning and Probabilistic Inference

Graph-based clustering and semi-supervised learning assume that nearby points on a data manifold tend to have compatible latent structure. Spectral methods partition a similarity graph through its eigensystem (Ng, Jordan, and Weiss 2001; Von Luxburg 2007), while Gaussian-field propagation, local-and-global consistency, and manifold regularization difuse sparse supervision according to graph geometry (Zhu, Ghahramani, and Laferty 2003; Zhou et al. 2003; Belkin, Niyogi, and Sindhwani 2006). These methods motivate transferring evidence from observed prompts to related but unobserved ones. Probabilistic graphical models additionally represent relational dependence and uncertainty. Potts and Markov random-field priors encourage neighboring discrete variables to take compatible states (Potts 1952; Besag 1974; Geman and Geman 1984). Because graph coupling makes exact posterior inference dificult, variational methods optimize a tractable approximation, with mean-field factorizations yielding scalable coordinate updates (Zhang 1992; McGrory et al. 2009; Wainwright and Jordan 2008; Blei, Kucukelbir, and McAulife 2017). Stochastic and streaming variants update approximate posteriors as new observations arrive (Hofman et al. 2013; Broderick et al. 2013).

## References

Bae, S.; Hong, J.; Lee, M. Y.; Kim, H.; Nam, J.; and Kwak, D. 2026. Online dificulty filtering for reasoning oriented reinforcement learning. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), 700–719.

Balunovic, M.; Dekoninck, J.; Petrov, I.; Jovanović, N.; and Vechev, M. 2026. Matharena: Evaluating llms on uncontam-

inated math competitions. Advances in Neural Information Processing Systems, 38.

Belkin, M.; Niyogi, P.; and Sindhwani, V. 2006. Manifold Regularization: A Geometric Framework for Learning from Labeled and Unlabeled Examples. Journal of Machine Learning Research, 7(85): 2399–2434.

Bengio, Y.; Louradour, J.; Collobert, R.; and Weston, J. 2009. Curriculum Learning. In International Conference on Machine Learning, 41–48.

Benjamini, Y.; and Hochberg, Y. 1995. Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. Journal of the Royal Statistical Society: Series B (Methodological), 57(1): 289–300.

Besag, J. 1974. Spatial interaction and the statistical analysis of lattice systems. Journal of the Royal Statistical Society: Series B (Methodological), 36(2): 192–225.

Blei, D. M.; Kucukelbir, A.; and McAulife, J. D. 2017. Variational Inference: A Review for Statisticians. Journal of the American Statistical Association, 112(518): 859–877.

Broderick, T.; Boyd, N.; Wibisono, A.; Wilson, A.; and Jordan, M. 2013. Streaming Variational Bayes. In Burges, C.; Bottou, L.; Welling, M.; Ghahramani, Z.; and Weinberger, K., eds., Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc.

Chen, J.; Xiao, S.; Zhang, P.; Luo, K.; Lian, D.; and Liu, Z. 2024. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. arXiv:2402.03216.

Cramér, H. 1946. Mathematical Methods of Statistics. Princeton, NJ: Princeton University Press.

Cui, G.; Yuan, L.; Wang, Z.; Wang, H.; Zhang, Y.; Chen, J.; Li, W.; He, B.; Fan, Y.; Yu, T.; et al. 2025. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456.

Dekoninck, J.; Jovanović, N.; Gehrunger, T.; Rögnvaldsson, K.; Petrov, I.; Sun, C.; and Vechev, M. 2026. Beyond benchmarks: Matharena as an evaluation platform for mathematics with llms. arXiv preprint arXiv:2605.00674.

Dixon, W. J.; and Mood, A. M. 1946. The Statistical Sign Test. Journal of the American Statistical Association, 41(236): 557–566.

Fang, Y.; Lin, J.; Fu, X.; Qin, C.; Shi, H.; Hu, C.; Pan, L.; Zeng, K.; and Cai, X. 2026. How to allocate, how to learn? dynamic rollout allocation and advantage modulation for policy optimization. In Findings of the Association for Computational Linguistics: ACL 2026, 14727–14744.

Gao, Z.; Kim, J.; Sun, W.; Joachims, T.; Wang, S.; Pang, R. Y.; and Tan, L. 2026. Prompt Curriculum Learning for Eficient LLM Post-Training. In The Fourteenth International Conference on Learning Representations.

Geman, S.; and Geman, D. 1984. Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. IEEE Transactions on pattern analysis and machine intelligence, (6): 721–741.

Grattafiori, A.; Dubey, A.; Jauhri, A.; Pandey, A.; Kadian,A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.;

Vaughan, A.; Yang, A.; Fan, A.; Goyal, A.; Hartshorn, A.; Yang, A.; et al. 2024. The Llama 3 Herd of Models. arXiv:2407.21783.

Guo, D.; Yang, D.; Zhang, H.; Song, J.; Wang, P.; Zhu, Q.; Xu, R.; Zhang, R.; Ma, S.; Bi, X.; et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

He, C.; Luo, R.; Bai, Y.; Hu, S.; Thai, Z.; Shen, J.; Hu, J.; Han, X.; Huang, Y.; Zhang, Y.; et al. 2024. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 3828–3850.

Hendrycks, D.; Burns, C.; Kadavath, S.; Arora, A.; Basart, S.; Tang, E.; Song, D.; and Steinhardt, J. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Hofman, M. D.; Blei, D. M.; Wang, C.; and Paisley, J. 2013. Stochastic variational inference. Journal of machine learning research.

Jain, N.; Han, K.; Gu, A.; Li, W.-D.; Yan, F.; Zhang, T.; Wang, S.; Solar-Lezama, A.; Sen, K.; and Stoica, I. 2024. LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code. arXiv preprint arXiv:2403.07974.

Kong, D.; Guo, Q.; Xi, X.; Wang, W.; Wang, J.; Cai, X.; Zhang, S.; and Ye, W. 2026. Rethinking the sampling criteria in reinforcement learning for LLM reasoning: A competencedificulty alignment perspective. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 31438– 31446.

Kruskal, W. H.; and Wallis, W. A. 1952. Use of Ranks in One-Criterion Variance Analysis. Journal of the American Statistical Association, 47(260): 583–621.

Kusupati, A.; Bhatt, G.; Rege, A.; Wallingford, M.; Sinha, A.; Ramanujan, V.; Howard-Snyder, W.; Chen, K.; Kakade, S.; Jain, P.; and Farhadi, A. 2022. Matryoshka Representation Learning. In Advances in Neural Information Processing Systems, volume 35.

Lalor, J. P.; Wu, H.; and Yu, H. 2016. Building an Evaluation Scale using Item Response Theory. In Su, J.; Duh, K.; and Carreras, X., eds., Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, 648– 657. Austin, Texas: Association for Computational Linguistics.

Li, J.; Beeching, E.; Tunstall, L.; Lipkin, B.; Soletskyi, R.; Huang, S.; Rasul, K.; Yu, L.; Jiang, A. Q.; Shen, Z.; et al. 2024. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9): 9.

Lightman, H.; Kosaraju, V.; Burda, Y.; Edwards, H.; Baker, B.; Lee, T.; Leike, J.; Schulman, J.; Sutskever, I.; and Cobbe, K. 2024. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, 39578–39601.

McGrory, C. A.; Titterington, D. M.; Reeves, R.; and Pettitt, A. N. 2009. Variational Bayes for estimating the parameters of a hidden Potts model. Statistics and Computing, 19(3): 329–340.

Mindermann, S.; Brauner, J. M.; Razzak, M. T.; Sharma, M.; Kirsch, A.; Xu, W.; Höltgen, B.; Gomez, A. N.; Morisot, A.; Farquhar, S.; and Gal, Y. 2022. Prioritized Training on Points that are Learnable, Worth Learning, and not yet Learnt. In Chaudhuri, K.; Jegelka, S.; Song, L.; Szepesvari, C.; Niu, G.; and Sabato, S., eds., Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, 15630– 15649. PMLR.

Ng, A.; Jordan, M.; and Weiss, Y. 2001. On spectral clustering: Analysis and an algorithm. Advances in neural information processing systems, 14.

Nguyen, H. T.; Nguyen, B.; Ma, W.; Zhao, Y.; She, R.; and Nguyen, V. A. 2026a. Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards. In The Fourteenth International Conference on Learning Representations.

Nguyen, M.; Venkatesh, S.; Le, H.; et al. 2026b. SPaCe: Unlocking Sample-Eficient Large Language Models Training With Self-Pace Curriculum Learning. In Findings of the Association for Computational Linguistics: ACL 2026, 3480–3507.

Pearson, K. 1900. On the Criterion That a Given System of Deviations from the Probable in the Case of a Correlated System of Variables Is Such That It Can Be Reasonably Supposed to Have Arisen from Random Sampling. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 50(302): 157–175.

Potts, R. B. 1952. Some generalized order-disorder transformations. In Mathematical proceedings of the cambridge philosophical society, volume 48, 106–109. Cambridge University Press.

Qu, Y.; Wang, Q.; Mao, Y.; Hu, V. T.; Ommer, B.; and Ji, X. 2026. Can prompt dificulty be online predicted for accelerating rl finetuning of reasoning models? In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, 1240–1250.

Reimers, N.; and Gurevych, I. 2019. Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th Internationa Joint Conference on Natural Language Processing, 3982– 3992. Association for Computational Linguistics.

Rodriguez, P.; Barrow, J.; Hoyle, A.; Lalor, J. P.; Jia, R.; and Boyd-Graber, J. 2021. Evaluation Examples are not Equally Informative: How should that change NLP Leaderboards? In Zong, C.; Xia, F.; Li, W.; and Navigli, R., eds., Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 4486–4503. Online: Association for Computational Linguistics.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y.; Wu, Y.; et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Sheng, G.; Zhang, C.; Ye, Z.; Wu, X.; Zhang, W.; Zhang, R.; Peng, Y.; Lin, H.; and Wu, C. 2024. HybridFlow: A Flexible and Eficient RLHF Framework. arXiv preprint arXiv: 2409.19256.

Swayamdipta, S.; Schwartz, R.; Lourie, N.; Wang, Y.; Hajishirzi, H.; Smith, N. A.; and Choi, Y. 2020. Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics. In Webber, B.; Cohn, T.; He, Y.; and Liu, Y., eds., Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 9275–9293. Online: Association for Computational Linguistics.

Team, K.; Du, A.; Gao, B.; Xing, B.; Jiang, C.; Chen, C.; Li, C.; Xiao, C.; Du, C.; Liao, C.; et al. 2025. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599.

Von Luxburg, U. 2007. A tutorial on spectral clustering. Statistics and computing, 17(4): 395–416.

Wainwright, M. J.; and Jordan, M. I. 2008. Graphical models, exponential families, and variational inference. Foundations and Trends® in Machine Learning, 1(1-2): 1–305.

Wang, H.; Hao, Z.; Luo, J.; Wei, C.; Shu, Y.; Liu, L.; Cheaterlin; Dong, H.; and Chen, J. 2026. Scheduling Your LLM Reinforcement Learning with Reasoning Trees. In The Fourteenth International Conference on Learning Representations.

Wu, M.; Zhang, Z.; Dong, Q.; Xi, Z.; Zhao, J.; Jin, S.; Fan, X.; Zhou, Y.; Lv, H.; Zhang, M.; et al. 2026. Reasoning or memorization? unreliable results of reinforcement learning due to data contamination. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 33944– 33952.

Xiong, W.; Ye, C.; Liao, B.; Dong, H.; Xu, X.; Monz, C.; Bian, J.; Jiang, N.; and Zhang, T. 2025. Reinforce-ada: An adaptive sampling framework for reinforce-style llm training. arXiv e-prints, arXiv–2510.

Xu, H.; Chen, S.; Qiu, R.; Yan, Y.; Luo, C.; Cheng, M.; He, J.; and Tong, H. 2026. Prune as You Generate: Online Rollout Pruning for Faster and Better RLVR. arXiv:2603.24840.

Yang, A.; Zhang, B.; Hui, B.; Gao, B.; Yu, B.; Li, C.; Liu, D.; Tu, J.; Zhou, J.; Lin, J.; Lu, K.; Xue, M.; Lin, R.; Liu, T.; Ren, X.; and Zhang, Z. 2024. Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement. arXiv:2409.12122.

Yao, J.; Hao, Y.; Zhang, H.; Dong, H.; Xiong, W.; Jiang, N.; and Zhang, T. 2026. Optimizing chain-of-thought reasoners via gradient variance minimization in rejection sampling and rl. Advances in Neural Information Processing Systems, 38: 163245–163284.

Yu, Q.; Zhang, Z.; Zhu, R.; Yuan, Y.; Zuo, X.; Yue, Y.; Dai, W.; Fan, T.; Liu, G.; Liu, L.; et al. 2026. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38: 113222–113244.

Zeng, Y.; Sun, Z.; Ji, B.; Min, E.; Cai, H.; Wang, S.; Yin, D.; Zhang, H.; Chen, X.; and Wang, J. 2026. CurES: From Gradient Analysis to Eficient Curriculum Learning for Reasoning LLMs. In The Fourteenth International Conference on Learning Representations.

Zhang, J. 1992. The mean field theory in EM procedures for Markov random fields. IEEE Transactions on signal processing, 40(10): 2570–2583.

Zhang, Y.; Li, M.; Long, D.; Zhang, X.; Lin, H.; Yang, B.; Xie, P.; Yang, A.; Liu, D.; Lin, J.; Huang, F.; and Zhou, J. 2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. arXiv:2506.05176.

Zheng, C.; Liu, S.; Li, M.; Chen, X.-H.; Yu, B.; Gao, C.; Dang, K.; Liu, Y.; Men, R.; Yang, A.; et al. 2025. Group sequence policy optimization. arXiv preprint arXiv:2507.18071.

Zheng, H.; Zhou, Y.; Bartoldson, B.; Kailkhura, B.; Lai, F.; Zhao, J.; and Chen, B. 2026. Act only when it pays: Eficient reinforcement learning for llm reasoning via selective rollouts. Advances in Neural Information Processing Systems, 38: 124321–124346.

Zhou, D.; Bousquet, O.; Lal, T.; Weston, J.; and Schölkopf, B. 2003. Learning with Local and Global Consistency. In Thrun, S.; Saul, L.; and Schölkopf, B., eds., Advances in Neural Information Processing Systems, volume 16. MIT Press.

Zhu, X.; Ghahramani, Z.; and Laferty, J. D. 2003. Semisupervised learning using gaussian fields and harmonic functions. In Proceedings of the 20th International conference on Machine learning (ICML-03), 912–919.