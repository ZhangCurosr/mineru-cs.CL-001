# BAYESPROMPT: HUMAN READABLE PROMPTS THAT MAKE SENSE

Franky Kevin Nando Tezoh, Ali Hussaini Umar, Alessandro Laio, & Guido Sanguinetti <sup>∗</sup>   
Department of Physics,   
Scuola Internazionale Superiore di Studi Avanzati (SISSA),   
Trieste, Italy,   
{fnandote,aumar,laio,gsanguin}@sissa.it

Riccardo Rende

Center for Computational Quantum Physics, Flatiron Institute, 162 5th Avenue, New York, NY 10010, {rrende}@flatironinstitute.org

## ABSTRACT

Reconstructing prompts that can elicit a desired answer or behaviour in an LLM is an open and important research topic. Optimisation methods which aim at minimising the perplexity of a given answer, however, consistently yield socalled pseudoprompts, unintelligible strings of tokens which can lack human interpretability. We argue that this is a consequence of the ill-posedness of the prompt optimisation task. By reframing the task as a Bayesian posterior inference over prompts, we propose an efficient algorithm to sample prompts which are both efficient (in terms of perplexity) and human readable. We compare our approach with state of the art alternatives showing on a real data set a marked improvement over a range of metrics.

## 1 INTRODUCTION

In-context learning (ICL) (Dong et al., 2022) has recently become a major focus in machine learning. Researchers found that Large Language Models (LLMs) can solve diverse tasks like translating languages, analyzing sentiments, or solving math problems simply by conditioning on a few exemplars within a prompt. They do this without requiring any update to the model’s internal parameters. Much research has been dedicated to trying to explain what gives rise to the surprising in-context learning behaviour of LLMs (Rai et al., 2024; Singh et al., 2024; Vulic et al., 2020; Jin et al., 2024;´ Ju et al., 2024; Elhage et al., 2021; Olsson et al., 2022); regardless of its origin, the ICL phenomenon has also been immediately recognised for its practical importance.

A key ingredient for ICL is designing the correct prompt for a given task. Prompt optimisation approaches (Sinha et al., 2021) aim to discover high confidence (low perplexity) prompts to leverage ICL for practical purposes, using both continuous soft prompts (Lester et al., 2021; Liu et al., 2024) and discrete hard prompts (Wen et al., 2023) to construct more effective inputs.

Many studies have addressed the prompt-optimisation problem, revealing a striking gap between human intuition and model behavior: optimal prompts are often short non-sensical sequences of tokens that nonetheless reliably steer a model towards a target output. Such pseudoprompts were first identified through universal adversarial triggers (Wallace et al., 2019); this phenomenon has since been shown to generalize, as models conditioned on unnatural, human-illegible sequences can match or exceed the performance of natural prompts (Zou et al., 2023; Kervadec et al., 2023; Cherepanova & Zou, 2024), consistent with evidence that models may rely more on distributional statistics than on human syntactic structure (Sinha et al., 2021). This suggests that the space of effective prompts is fa larger, and far less human readable, than the space of prompts a person would ever write. Subsequent studies on Unnatural Language Processing broadened this observation, challenging the assumption that LLMs rely on human-centric linguistic structures: (Zou et al., 2023; Kervadec et al., 2023; Cherepanova & Zou, 2024) demonstrated that models prompted with unnatural sequences which are semantically and syntactically incoherent to humans still produce predictable, meaningful outputs.

Contribution: In this paper, we aim to illustrate the statistical origins of pseudoprompts and to propose an efficient algorithm that produces highly efficient prompts (in terms of resulting in the correct output with low perplexity) which are also human interpretable. Our motivation is both in terms of obtaining more readable, and hence presumably trustable, prompts, and to understand the operational modalities of LLMs when run to solve an inverse problem.

Our main contributions are:

• we reformulate the prompt optimisation problem in terms of Bayesian inference, showing that a neglect of the prior term gives rise to the pseudoprompt phenomenon;

• we propose an efficient algorithm, based on Markov-chain Monte Carlo, to obtain human readable prompts;

• We illustrate how reverse language modelling can be used to provide an effective initialisation to both Bayesian and optimisation methods that greatly improves prompt reconstruction quality.

We validate both qualitatively and quantitatively the new framework on a range of question and answer tasks, showing that it obtains low perplexity prompts with high readability, significantly improving on existing techniques.

## 2 RELATED WORK

Learning a prompt from a given answer can be framed as a reverse engineering problem: Large Language Models (LLMs) are trained to predict an answer given a question, and recovering the prompt requires inverting the direction. Prompt recovering can be formulated either in discrete token space or continuous embedding space. Among the earliest works on automatic prompt learning in continuous space, (Lester et al., 2021; Li & Liang, 2021) showed that learned soft prompts yield higher confidence in the target answer than natural, human-interpretable prompts. However, (Lester et al., 2021) note that once decoded into discrete tokens, these learned prompts are gibberish, and incomprehensible to humans. To make such prompts interpretable upon decoding, (Wen et al., 2023) alternate between projecting the current embedding sequence onto the vocabulary and computing the gradient over the projected embeddings to update the learned representations. Still, the resulting prompts lack fluency.

In the discrete setting, gradient based discrete token search was first introduced by (Ebrahimi et al., 2018), who use the gradient with respect to one-hot token encoding to identify the token replacement that most increases the classification loss. This strategy was later adapted by (Zou et al., 2023) to the setting of eliciting a specific target generation from an LLM. Building on this, (Melamed et al., 2023) showed that for a given ground-truth question, there exists a corresponding evil twin that elicits the same behavior as the ground-truth, and further showed that regularizing the objective function with the fluency term produces prompts that are fluent but lack similarity with the ground-truth question. Later, (Rakotonirina et al., 2025) and (Melamed et al., 2025) study the structure of these evil twins in more depth, finding that they contain a mixed language-like and non-linguistic tokens.

MCMC sampling, a key ingredient in our approach, has been applied to text generation more broadly, though not to prompt learning itself. (Goyal et al., 2021) show that the masked conditionals of Masked Language Models (MLMs) do not correspond to a valid joint distribution, and instead use them as proposal distribution within a Metropolis-Hastings sampler to draw samples from the model’s implicit energy. Reverse language models, another key ingredient, have been explored for purposes other than sampling: (Yin et al., 2026) train a purely reverse language model from scratch and use it to construct a posterior based reward function, reranking forward generated outputs by how well the reconstruct the input. In our approach we use the reverse model as warm-start initialiser for the MCMC, and not for post hoc reranking the prompts. To our knowledge, no prior work formulates prompt learning itself as a sampling problem.

In summary, existing literature presents a strict trade-off: optimised prompts are either highly effective but completely uninterpretable, or fluent but semantically disconnected from the target task. Critically, no prior method achieves both high task confidence and human fluency; the two properties are always in direct tension with one another. This is the gap our framework addresses.

## 3 BAYESIAN REFORMULATION OF THE PROBLEM

Large language models (LLMs) learn the conditional distribution of the subsequent tokens given a preceding context. Let $( { \bf a } , { \bf q } ) \sim R$ be a pair of answer and question (each defined by a sequence of tokens).

Given a pretrained model one could learn the input/question tokens that minimize the negative loglikelihood of the answer a:

$$
\operatorname* { m i n } _ { \pmb { q } \in \mathscr { V } ^ { n } } \mathcal { L } ( \pmb { q } ) , \quad \mathrm { w h e r e } \quad \mathcal { L } ( \pmb { q } ) = - \log P ( \pmb { a } | \pmb { q } ) ,\tag{1}
$$

where denotes the model’s vocabulary and n is the sequence length of the question. $P ( { \pmb a } | { \pmb q } )$ can be explicitly estimated by a LLM trained on next token prediction as $\begin{array} { r l } { \prod _ { i = 1 } ^ { m } { P } ( a _ { i } | \pmb { a } _ { < i } , \dot { \pmb { q } } ) \stackrel { - } { = } } & { { } } \end{array}$ $\Pi _ { i = 1 } ^ { m }$ softmax $\left( f ( \pmb q , \pmb { a } _ { < i } ) \right) _ { a _ { i } }$ , where $z _ { i } = f ( \pmb { q } , \pmb { a } _ { < i } ) \in \mathbb { R } ^ { | \nu | }$ is the logit vector the LLM outputs at position $i ,$ m is the length of the answer $^ { a . }$ In this equation, the function $f$ is defined by the neural networks, and therefore depends on a (large) set of parameters, which in our derivation are considered as given and held fixed.

However, this approach to learning the questions leads to prompts which are often ungrammatical sequences of seemingly random tokens. This stems from the ill-posed nature of equation 1. To ensure that equation 1 is well defined, we must incorporate a prior distribution over the questions $\begin{array} { r } { P ( \pmb q ) = \prod _ { i = 1 } ^ { n } \dot { P } \left( q _ { i } | \pmb q _ { < i } \right) } \end{array}$ . Consequently, our objective becomes:

$$
\operatorname* { m i n } _ { \mathbf { q } \in \mathcal { V } ^ { n } } \mathcal { L } ( \mathbf { q } ) , \quad \mathrm { w h e r e } \quad \mathcal { L } ( \mathbf { q } ) = - \log P ( \mathbf { a } | \mathbf { q } ) - \log P ( \mathbf { q } ) .\tag{2}
$$

By incorporating the prior distribution over the inputs $P ( \pmb q )$ , we encourage the optimisation framework to prioritize the generation of fluent prompts. To solve the optimisation problem in equation 2, we use two algorithms described in the literature, greedy coordinate gradient (GCG) and gradient descend (GD), and the approach which we propose, Markov Chain Monte Carlo (MCMC).

Greedy coordinate Gradient (GCG) (Zou et al., 2023): Gradient information is used to get potential token replacements for a given position by evaluating $- \nabla _ { \mathbf { } e _ { q _ { i } } } \mathcal { L } ( \pmb q )$ with respect to one-hot vector $e _ { q _ { i } }$ of the token at the position i; then we select the top-k tokens which maximize the gradient of $- \mathscr { L } ( \pmb { q } ) , \mathscr { X } _ { i } : = \mathrm { T o p } - k \left( - \nabla _ { e _ { q _ { i } } } \mathscr { L } ( \pmb { q } ) \right)$ . We construct B proposals sequence tokens $\{ \tilde { \pmb q } ^ { ( b ) } \} _ { b = 1 } ^ { B } ,$ by selecting a position i uniformly at random and replacing the original token with a candidate $\boldsymbol { \tilde { q } } _ { i } ^ { ( b ) } \sim \mathrm { U n i f o r m } ( \mathcal { X } _ { i } )$ . Finally, the best proposal is selected greedily based on the actual loss value. This process is iterated for a fixed number of steps or until convergence.

Gradient Descent (GD): It involves applying gradient descent on the sequence embedding $\mathbf { q } _ { e m b } \in \mathbb { R } ^ { n \times d }$ to minimize the loss $\mathcal { L } ( q _ { e m b } )$ , where d is the embedding dimension. After optimising the continuous embedding space, the final sequence is obtained by mapping the learned embedding back to the nearest discrete tokens in the vocabulary tokens. A natural extension of this approach, known as Hard prompts made $\operatorname { E a Z y } ,$ in abbreviation PEZ (Wen et al., 2023) interleaves this nearest-neighbor embedding assignment with the gradient updates: at each iteration, the current embeddings are first replaced by their closest vocabulary embeddings via cosine similarity, and the gradient is then computed at those vocabulary-aligned embeddings before the optimiser update is applied. This ensures that the gradient signal is always evaluated at an embedding that belongs to the vocabulary embedding matrix, yielding a more principled optimisation than standard GD.

## 3.1 REVERSE LANGUAGE MODEL

Standard autoregressive language models are trained to sequentially predict each token in a sequence given all the previous observed tokens. Formally, given a sequence of n tokens $\pmb { q } = ( q _ { 1 } , q _ { 2 } , \dots , q _ { n } )$ the joint probability of the sequence admits the following factorization:

$$
{ \cal P } ( { \pmb q } ) = \prod _ { i = 1 } ^ { n } { \cal P } ( q _ { i } | { \pmb q } _ { < i } ) ,\tag{3}
$$

where $\pmb q _ { < i } = ( q _ { 1 } , q _ { 2 } , \dots , q _ { i - 1 } )$ denotes the subsequence of all tokens preceding position i. Under this formulation, the model processes tokens strictly from left to right, and each conditional distribution $P ( q _ { i } | \mathbf { q } _ { < i } )$ is parameterized by a neural network, typically a Transformer architecture (Vaswani et al., 2017). We refer to this conventional formulation as theforward model.

A natural question arises: what if the sequence dependency is reversed? Rather than conditioning each token to its left context, one can construct a model that processes the sequence from right to left. Concretely, given the original token sequence q, we define its reversal as:

$$
\pmb q _ { \mathrm { r e v } } = \left( q _ { n } , q _ { n - 1 } , q _ { n - 2 } , \ldots , q _ { 1 } \right) ,\tag{4}
$$

obtaining by permuting the tokens indices in reverse order. A reversed or backward (Peters et al., 2018; Yin et al., 2026) model is then trained autoregressively on this reversed sequence, learning the following factorization:

$$
P ( \pmb q _ { \mathrm { r e v } } ) = \prod _ { i = 1 } ^ { n } P \left( q _ { n + 1 - i } | \pmb q _ { \mathrm { r e v , < } i } \right) .\tag{5}
$$

where ${ \pmb q } _ { \mathrm { r e v } , < i } = ( q _ { n } , q _ { n - 1 } , \ldots , q _ { n + 2 - i } )$ denotes the suffix context acting as the casual history for the target token $q _ { n + 1 - i } .$

Importantly, the reversed model is not a post-hoc transformation of the forward model; it is an independently trained model whose training data consists of token sequences presented in reversed order. Consequently, the backward model learns a distinct set of attention patterns and internal representations shaped by right-to-left causal dynamics.

## 3.2 METROPOLIS-HASTINGS ALGORITHM.

Our approach aims at obtaining samples from the posterior distribution $P ( \mathbf { q } | \mathbf { a } )$ . We do so using the classical Metropolis-Hastings (MH) algorithm, treating log $P ( \mathbf { a } , \mathbf { q } )$ term from equation 2 as target distribution. MH allows exploring the prompt space $\mathcal { V } ^ { n }$ in a way that respects the prior $\mathbb { P } ( \mathbf { q } )$ , ensuring natural language structure while still gravitating toward regions that maximize the likelihood of the answer a conditioned on the question q. At each time step, we draw a new sample $q ^ { \prime }$ from a proposal distribution $Q \left( { q ^ { \prime } | q } \right)$ , whose form we specify below, and we either accept or reject it with an acceptance probability equal to:

$$
\alpha \left( { \pmb q } ^ { \prime } , { \pmb q } \right) = \mathrm { m i n } \left( 1 , \frac { \pi ( { \pmb q } ^ { \prime } ) Q ( { \pmb q } | { \pmb q } ^ { \prime } ) } { \pi ( { \pmb q } ) Q ( { \pmb q } ^ { \prime } | { \pmb q } ) } \right) ,\tag{6}
$$

where $\pi \left( \mathbf { q } \right) = P ( \mathbf { a } , \mathbf { q } )$

The distribution of the sequence of sample generated $\pmb q _ { n }$ converges to the target distribution $\pi$ for irreducible and aperiodic Markov chains as n goes to infinity. While the convergence to the stationary distribution of the algorithm is satisfied for any valid proposal and initial state $\mathbf { \delta q } _ { 0 } .$ , practical deployment of MH requires an effective initialisation and a proposal distribution that promotes good mixing.

Initialisation: We initialise the process by sampling a question from a simple reverse model, obtained by finetunining the forward model on the reverse task. This avoids the high computational costs of evolving the MCMC from an initial state which is atypical for the distribution that we want to sample. This warm-start initialisation ensures the chain begins with a linguistically fluent question, even if the initial confidence on the answer is relatively low.

Proposal Distribution for the MH sampler: Transitioning from state $\pmb q$ to state $q ^ { \prime }$ involves a set of discrete edit operations, specifically replacement, insertion, and deletion. While replacement modifies one or more tokens within the existing sequence, insertion and deletion operations dynamically adjust the sequence length.

Actions: Let $\pmb q = ( q _ { 1 } , q _ { 2 } , \dots , q _ { i - 1 } , q _ { i } , q _ { i + 1 } , \dots , q _ { n } )$ be the current sequence of question tokens. To generate a proposal state $q ^ { \prime }$ , we select a token index $i \in \{ 1 , \ldots , n \}$ uniformly at random and replace $q _ { i }$ with a new token $q _ { i } ^ { \prime } ,$ sampling from the probability distribution generated by either the left $( \pmb q _ { < i } )$ or the right context $( \pmb q _ { > i } )$ . The proposal probability $Q ( \pmb { q } ^ { \prime } | \pmb { q } )$ is then defined by the selection of either the forward or the reverse model:

$$
Q ( \pmb { q } ^ { \prime } | \pmb { q } ) = \frac { 1 } { n } \times \left\{ \begin{array} { l l } { P ( q _ { i } ^ { \prime } | \pmb { q } _ { < i } ) } & { \mathrm { i f ~ u s i n g ~ t h e ~ f o r w a r d ~ m o d e l } } \\ { P _ { \mathrm { r e v } } ( q _ { i } ^ { \prime } | \pmb { q } _ { > i } , \pmb { a } ) } & { \mathrm { i f ~ u s i n g ~ t h e ~ r e v e r s e ~ m o d e l } } \end{array} \right.\tag{7}
$$

This formulation ensures that the local edit is informed by either the preceding or succeeding lexical context.

To perform an insertion (Miao et al., 2019), we select an index $i \in \{ 1 , \ldots , n + 1 \}$ uniformly at random to determine the position of a new token $q ^ { * }$ . This results in an expanded sequence of length $n + 1 , \pmb { q } = ( q _ { 1 } , q _ { 2 } , \dots , q ^ { * } , q _ { i } , q _ { i + 1 } , \dots , q _ { n } )$ . The proposal probability is defined by the selection of the position and the sampling of the new token:

$$
Q ( \pmb { q } ^ { \prime } | \pmb { q } ) = \frac { 1 } { n + 1 } \times \left\{ P ( \pmb { q } ^ { * } | \pmb { q } _ { 1 : i } ) \qquad \mathrm { i f ~ u s i n g ~ t h e ~ f o r w a r d ~ m o d e l } \right.\tag{8}
$$

Lastly, to perform deletion, we select an index $i \in \{ 1 , \ldots , n \}$ uniformly at random and remove the corresponding tokens $q _ { i }$ . This operation reduces the sequence length from n to $n - 1$ , resulting in the state $\pmb q ^ { \prime } = ( q _ { 1 } , q _ { 2 } , \dots , q _ { i - 1 } , q _ { i + 1 } , \dots , q _ { n } )$ . Since this operation involves only the selection of the position and no token sampling,

The proposal probability is defined simply as:

$$
Q ( q ^ { \prime } | q ) = { \frac { 1 } { n } } .\tag{9}
$$

This ensures that any token within the sequence has an equal probability of being targeted for deletion during the Markov chain transition.

## 4 IMPLEMENTATION DETAILS

Dataset and Model: We evaluate our methodology using the NQ-OPEN dataset, an open domain benchmark structured into natural language question-answer pairs, using the standard train/test split. This configuration enables the model to establish direct semantic mappings between queries and their respective targets. We fine-tune Llama-3.2-1B-Instruct model on the NQ-OPEN TRAIN split in a supervise manner to optimize its generative performance (Shani et al., 2025).

GCG and GD for NQ-OPEN TEST: Given the forward model Llama-3.2-1B-Instruct fine-tuned on NQ-OPEN dataset (Huang et al., 2025), we perform prompt learning on equation 2 for each answer in NQ-OPEN TEST, using optimisation algorithms such as greedy coordinate gradient for discrete token selection and gradient descent for learning in continuous space described.

Reverse-model: We finetune Llama-3.2-1B-Instruct model with Low-Rank Adaptation (LoRA) (Hu et al., 2022) on NQ-OPEN TRAIN split sequences whose token order is fully reversed, so the model learns to generate a question, in reverse, when prompted with a reversed answer (see section 3.1).

MCMC for NQ-OPEN TEST: Following the fine-tuning of both the forward and reverse Llama-3.2-1B-Instruct models on NQ-OPEN dataset, we employed the Metropolis Hastings algorithm to explore the posterior distribution of the input given the answer $P ( \pmb q | \overset { \cdot } { \pmb q } ) \propto P \left( \pmb q | \overset { \cdot } { \pmb q } \right) P \overset { \cdot } { \left( \pmb q \right) }$ . This MCMC approach allows us to generate a diverse set of question samples conditioned on a given answer. For each MCMC trajectory under a fixed answer, we retain exclusively the final state of the Markov chain as the optimised question. The transition proposal distribution relies on token level mutations with action probabilities set to $p _ { \mathrm { r e p l a c e } } = 0 . 6 , p _ { \mathrm { d e l e t e } } = 0 . 2$ , and $p _ { \mathrm { i n s e r t } } = 0 . 2$ (ensuring $\sum p = 1 )$ . Specially, a replacement move modifies at most two tokens simultaneously, whereas insertion and deletion moves operate on a single token per step. We provided a detailed statistical analysis of the MCMC sampling process to illustrate its convergence and efficiency in the prompt learning task in Appendix B.

## 5 RESULTS AND DISCUSSION

This section reports the primary evaluation metrics for the learned prompts, comparing the optimisation-based baselines (GCG, GCG-Reg and GD-PEZ) against our proposed MCMC approach, which uses the forward model to generate proposals. We analyze two quantities: the fluency $P ( \pmb q )$ and the confidence, defined as the conditional probability of the answer given the question $P ( \pmb { a } | \pmb { q } )$ . Beyond comparing MCMC to the baselines directly, we also examine how much each baseline benefits from reverse-model warm-start initialisation relative to random initialisation. The resulting performance gap between the two schemes directly quantifies the importance of a good initialisation for the prompt inversion problem, a non trivial consideration given the vast, discrete, and highly non-convex nature of the token search space.

## 5.1 DISTRIBUTIONAL-LEVEL ANALYSIS OF THE METHODS

![](images/5ffdbdd6b3f770ada9a68c59fdcc96302029b9a2ef309f556a37d4d28289c9d4.jpg)  
(a)

![](images/92b67beb1596fcc2ed31268f5fbfaa4bedb7d40b36daeb0c9c77892fc0fb0394.jpg)  
(b)  
Figure 1: Comparison of the distribution of probability between the ground truth (black), the optimisation-based methods (GCG, GCG-Reg, GD-PEZ) under random initialisation and the MCMC-sampled approach. Random initialisation results are computed over 371 question-answer pairs from the NQ-OPEN dataset, using a prompt length of 10 tokens. Left: Distribution of the negative log-conditional probability  log $\bar { P } ( { \pmb a } | { \pmb q } )$ , measuring how well the generated question recovers the target answer. Right: Distribution of the negative log-fluency, log $P ( \pmb q )$ , measuring the linguistic naturalness of the generated prompt.

As shown in Figure 1a, GCG-Reg’s and GD-PEZ’s confidence distribution under random initialisation both closely tracks the ground-truth (GT) distribution, while MCMC’s confidence is modestly lower than GT, with a distribution notably more concentrated (peaked) around its mode than the others. In contrast, GCG under random initialisation exhibits markedly higher confidence than GT, with a pronounced heavy tail toward much higher confidence values, indicating that a substantial fraction of its recovered prompts are far more confident in the answer than GT or any other method, consistent with unregularized GCG over-optimizing for confidence in the answer. Figure 1b shows that MCMC achieves the highest fluency among all methods, with a distribution closest to the ground truth, while GCG, GCG-Reg, and GD-PEZ under random initialisation all remain noticeably less fluent. Taken together, these results confirm that MCMC is the only method to jointly achieve near GT confidence and the best fluency among the methods compared. GCG’s excess confidence comes at the cost of the marked fluency gap seen in Figure 1b. We refer to prompts that achieve confidence in the answer at least as high as GT at the cost of near-zero fluency as pseudoprompts, a designation that captures this pathological trade-off between confidence and fluency more precisely and quantitatively than a purely qualitative description will allow. Hence, GCG, GCG-Reg, and GD-PEZ under random initialisation all produce pseudoprompts, whereas MCMC’s confidence remains slightly below GT while achieving the best fluency among all methods compared, placing it outside this pathological trade-off.

## 5.2 THE EFFECT OF INITIALISATION ON OPTIMISATION-BASED METHODS

The effect of the initialisation on optimisation-based methods (GCG, GCG-Reg and GD-PEZ) is depicted by Figure 2. Warm-start initialisation does not meaningfully change GCG’s confidence pattern: it remains substantially more confident than the GT under either initialisation, consistent with the heavy-tail behavior noted earlier. GD-PEZ, by contrast, shows confidence close to GT un der both initialisation schemes, with warm-start and random-init curves nearly overlapping. GCG Reg shows the opposite shift: while its confidence closely tracked GT under random initialisation, warm-start pushes it to become noticeably less confident than GT. Turning to fluency, warm-start initialisation yields a substantial improvement for all three methods relative to random initialisation, most dramatically for GCG-Reg, whose fluency distribution moves from being almost entirely separated from GT under random initialisation to a substantially overlapping with the GT under warmstart, though a distinguishable gap remains. GD-PEZ and GCG also improve under warm-start but remain far less fluent than GT. These results show the effect of initialisation is asymmetric across confidence and fluency: warm-start consistently and substantially improves fluency across all three methods, whereas its effect on confidence is method-specific, leaving GCG and GD-PEZ unchanged, and pushing GCG-Reg past GT in the opposite direction. Comparing all three optimisation-based baselines directly to MCMC, only GCG-Reg under warm-start approaches MCMC’s performance in both confidence and fluency: its gap from the GT on both metrics is comparable in size to MCMC’s, whereas GCG and GD-PEZ remain far more separated from GT on at least one of the two metrics even under warm-start. MCMC nonetheless remains closer to GT than GCG-Reg on both confidence and fluency, but GCG-Reg is the only baseline for which warm-start narrows the gap enough to be broadly comparable to MCMC, rather than dominated by it.

## 5.3 QUALITATIVE ANALYSIS

Figure 3 provides qualitative examples of the prompts learned by optimisation-based methods (under random initialisation) and the sampling-based MCMC framework. As observed, the candidate prompts optimised via unregularized GCG, GCG-Reg and GD-PEZ consist primarily of gibberish, incorporating unnecessary formatting symbols alongside ungrammatical, mixed-language text (Rakotonirina et al., 2025; Melamed et al., 2025) that lacks grammatical, syntactic and semantic coherence. In contrast, the candidate prompts sampled via MCMC demonstrate clear human readability and semantic interpretability, preserving structured linguistic features from the original ground-truth reference questions. Extensive examples of the learned prompts under random initialisation and the sampled MCMC prompts are provided in Appendix C, Figure 8, and qualitative results on the optimisation-based methods under warm-start initialisation, alongside our MCMC approach are also depicted in Appendix C, Figure 7

To automatically assess the optimised/sampled question-answer pairs, Figure 4 presents two metrics scored using an LLM-judge evaluation workflow modeled on LMSYS’s (Large Model Systems Organization) Chatbot Arena (prompts detailed in the appendix D): the plausibility of the questionanswer pair, and the grammar correctness of the question alone. Across both metrics, our MCMC approach for prompt learning comes out ahead of the baseline methods (GCG, GCG-REG and GD-PEZ under random initialisation), with MCMC showing the largest margin on plausibility (329 wins vs. 6 for GCG-Reg, 13 for GCG and 23 for GD-PEZ), and a similarly clear separation on grammar correctness, where MCMC’s scores concentrate sharply near 0.9 -1.0, while GCG, GCG-Reg, and GD-PEZ all show broad distributions peaking in the 0.4-0.5 range, with GCG-Reg shifted slightly further right than GCG and GD-PEZ.

![](images/a3e203710f671ec98552364e814c44b4c49a78655574d61bcaafd3d8b3925712.jpg)  
Figure 2: Effect of initialisation on optimisation-based methods (GCG, GCG-Reg, GD-PEZ), compared against ground truth (black dashed) and MCMC (red), evaluated over 371 question-answer pairs fron NQ-OPEN dataset. Each row shows one method under warm-start (solid) and random (dotted) initialisation, with the same ground truth and the MCMC distributions repeated across rows for reference. Left column $\mathbf { ( a , c , e ) } { \mathrm { : } }$ : Distributionod the negative log-conditional probability − <sup>log</sup> $P ( { \pmb a } | { \pmb q } )$ , measuring confidence in the recovered answer. Right column $_ { ( \mathbf { a } , \mathbf { c } , \mathbf { e } ) : }$ Distribution of the negative log-fluency log $P ( \pmb q )$ , measuring the linguistic naturalness of the recovered prompt.

## 6 CONCLUSION

In this work, we recast prompt optimisation as a problem of Bayesian inference. We show that incorporating the prior over the question is essential to learn prompts that remain fluent rather than degenerating into uninterpretable token sequences. In contrast to prompt optimisation techniques such as unregularized GCG and GD-PEZ, our approach, combined with the warm-start initialisation from a reverse model, learns prompts that achieve a genuine balance between answer confidence and fluency. While regularized GCG (GCG-Reg) with the prior distribution over the question and warmstart initialisation also improves this balance, our MCMC-based approach achieves slightly better confidence and fluency jointly. Moreover, and very importantly, our approach is the only one which, for a given target answer, allows sampling a probability distribution of the questions, consistently with probabilistic formulation of language models (see eq. 3).

![](images/a3f96c5b6bac6a3b9e7e8f7dea5be97874a29c63a328e7adfff670bc0f52d70e.jpg)

<table><tr><td>Methods</td><td>Prompts</td><td>Target-Answers</td></tr><tr><td>Ground truth</td><td>who has the maximum number of wimbledon singles title</td><td>Roger Federer</td></tr><tr><td>GCG</td><td>lengULK-&gt;iggs(&quot;)_previousinkerieur nhé cz</td><td>Lengulk</td></tr><tr><td>GCG-Reg</td><td>puə Federawre yakkenpreuddel</td><td>Federer</td></tr><tr><td>MCMC</td><td>who won the world&#x27;s best men&#x27;s player award in 2018 tennis wikipedia page</td><td>Roger Federer</td></tr><tr><td>GD-PEZ</td><td>pouringères发布 Mario.awtextra Rachel RonaldoMari),( Justin</td><td>Tiger Woods</td></tr><tr><td>Ground truth</td><td>who won the 2018 formula 1 french grand pix tournament</td><td>Sebastian Vettel</td></tr><tr><td>GCG</td><td>Bin whose_copy accused Plantae.VERSION roli Batman Shots&quot;&quot;))Trading driver</td><td>Sebastian Vettel</td></tr><tr><td>GCG-Reg</td><td>how didevite coach game of mous instruments of change</td><td>the sitar</td></tr><tr><td>MCMC</td><td>who the most professional racing driver is in the current century</td><td>Sebastian Vettel</td></tr><tr><td>GD-PEZ</td><td>Shaunned Temple championships tyres gearbox SebastianVáclav Adams</td><td>Václav Adams</td></tr><tr><td>Ground truth</td><td>what were the 2 sides in ww1 called</td><td>the Triple Entente; Allied Powers</td></tr><tr><td>GCG</td><td>RED pellet  Fr gro Berm p.\nplanes opposing</td><td>United States</td></tr><tr><td>GCG-Reg</td><td>13 c Germany allied )); BufReads Rank NetCoord</td><td>13th</td></tr><tr><td>MCMC GD-PEZ</td><td>was the and fought in ww1 together too?</td><td>France</td></tr><tr><td></td><td>ting War zbNational&lt;y wonu IKE approximately entity</td><td>United States of America</td></tr><tr><td>Ground truth</td><td>who is the youngest formula 1 world champion</td><td>Sebastian Vettel</td></tr><tr><td>GCG</td><td>Cast Audiabda \n\n\n Franzı tutin</td><td>Klaus Maria Brandauer</td></tr><tr><td>GCG-Reg</td><td>dtHat wum Runas Torumewo</td><td>Tora Tora Tora! The Battle of Midway</td></tr><tr><td>MCMC</td><td>what is the most racing drivers worldwide today</td><td>Lewis Hamilton</td></tr><tr><td>GD-PEZ</td><td>守 playing 解 annonce Hisitiwe turboWould Ferrari</td><td>Ferrari</td></tr></table>

Figure 3: Qualitative analysis of prompts learned by the optimisation-based methods and the sampling-based MCMC framework. For each target answer, the ground truth question is shown alongside the corresponding question recovered by GCG, GCG-Reg, and GD-PEZ under random initialisation, and by our MCMC based sampler (final sample and forward proposal).  
Figure 4: Left: Histogram showing the total distribution of winning instances for question-answer plausibility across optimisation-based methods under random initialisation (GCG, GCG-Reg, GD-PEZ) and sampling framework (reporting the forward variant proposal for our MCMC approach ). Right: Grammar correctness score distribution for the same evaluated methods.

Several directions are worth exploring. For the sake of computational efficiency all the analysis presented in this work has been performed on Llama-3.2-1B, a relatively small model. All the scheme can be ported with no modification to much larger models. A second direction is replacing Metropolis-Hastings with alternative sampling schemes, such as Rosenbluth Sequential Monte

Carlo. A final intriguing direction is to extend prompt inversion to world models such as joint embedding predictive architecture (JEPA) (Assran et al., 2023). Unlike autoregressive models, JEPA learns by predicting abstract latent representations of its target input. It would be interesting to define a notion of prompt inversion in a purely representational space, where the target is no longer a token sequence but a latent state.

## 7 ACKNOWLEDGEMENTS

The authors would like to thank Arsene Gibbs Nwemadji Tiako and Nafack Baurice for insightful discussions and suggestions.

## REFERENCES

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15619–15629, 2023.

Valeriia Cherepanova and James Zou. Talking nonsense: Probing large language models’ understanding of adversarial gibberish inputs. arXiv preprint arXiv:2404.17120, 2024.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2022.

Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing Dou. Hotflip: White-box adversarial examples for text classification. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 31–36, 2018.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1(1):12, 2021.

Kartik Goyal, Chris Dyer, and Taylor Berg-Kirkpatrick. Exposing the implicit energy networks behind masked language models via metropolis–hastings. arXiv preprint arXiv:2106.02736, 2021.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.

Hai Huang, Yann LeCun, and Randall Balestriero. Llm-jepa: Large language models meet joint embedding predictive architectures. arXiv preprint arXiv:2509.14252, 2025.

Mingyu Jin, Qinkai Yu, Jingyuan Huang, Qingcheng Zeng, Zhenting Wang, Wenyue Hua, Haiyan Zhao, Kai Mei, Yanda Meng, Kaize Ding, et al. Exploring concept depth: How large language models acquire knowledge at different layers? arXiv preprint arXiv:2404.07066, 2024.

Tianjie Ju, Weiwei Sun, Wei Du, Xinwei Yuan, Zhaochun Ren, and Gongshen Liu. How large language models encode context knowledge? a layer-wise probing study. arXiv preprint arXiv:2402.16061, 2024.

Corentin Kervadec, Francesca Franzon, and Marco Baroni. Unnatural language processing: How do language models handle machine-generated prompts? arXiv preprint arXiv:2310.15829, 2023.

Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021.

Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 4582–4597, 2021.

Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. Gpt understands, too. AI open, 5:208–215, 2024.

Rimon Melamed, Lucas H McCabe, Tanay Wakhare, Yejin Kim, H Howie Huang, and Enric Boix-Adsera. Prompts have evil twins. arXiv preprint arXiv:2311.07064, 2023.

Rimon Melamed, Lucas Hurley McCabe, and H Howie Huang. Demystifying optimized prompts in language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 2983–2999, 2025.

Ning Miao, Hao Zhou, Lili Mou, Rui Yan, and Lei Li. Cgmh: Constrained sentence generation by metropolis-hastings sampling. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 6834–6842, 2019.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022.

ME Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer, and PG Allen. Deep contextualized word representations. 2227–2237. New Orleans, Louisiana: Association for Computational Linguistics, 2018.

Daking Rai, Yilun Zhou, Shi Feng, Abulhair Saparov, and Ziyu Yao. A practical review of mechanistic interpretability for transformer-based language models. arXiv preprint arXiv:2407.02646, 2024.

Nathanaël Carraz Rakotonirina, Corentin Kervadec, Francesca Franzon, and Marco Baroni. Evil twins are not that evil: Qualitative insights into machine-generated prompts. In Proceedings of the 8th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pp. 48– 68, 2025.

Chen Shani, Liron Soffer, Dan Jurafsky, Yann LeCun, and Ravid Shwartz-Ziv. From tokens to thoughts: How llms and humans trade compression for meaning. arXiv preprint arXiv:2505.17117, 2025.

Chandan Singh, Jeevana Priya Inala, Michel Galley, Rich Caruana, and Jianfeng Gao. Rethinking interpretability in the era of large language models. arXiv preprint arXiv:2402.01761, 2024.

Koustuv Sinha, Robin Jia, Dieuwke Hupkes, Joelle Pineau, Adina Williams, and Douwe Kiela. Masked language modeling and the distributional hypothesis: Order word matters pre-training for little. arXiv preprint arXiv:2104.06644, 2021.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Ivan Vulic, Edoardo Maria Ponti, Robert Litschko, Goran Glavaš, and Anna Korhonen. Probing´ pretrained language models for lexical semantics. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 7222–7240, 2020.

Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, and Sameer Singh. Universal adversarial triggers for attacking and analyzing nlp. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pp. 2153–2162, 2019.

Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery. Advances in Neural Information Processing Systems, 36:51008–51025, 2023.

Xunjian Yin, Sitao Cheng, Yuxi Xie, Xinyu Hu, Li Lin, Xinyi Wang, Liangming Pan, William Yang Wang, and Xiaojun Wan. Ledom: Reverse language model. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 21306– 21326, 2026.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.

![](images/f13f1d3cc28ddf91fe8dfdcd3d79ad2ea7b8362568900e3d9f7c8620c9804452.jpg)

![](images/fdeba426ff4ae5ff86b7222e61d8a4bedd8e9d4b09bfd1943b629b7f6c96f3f2.jpg)

Figure 5: Comparison of the distribution of probability between the ground truth (blue), the sampling and optimisation frameworks evaluated over 371 question-answer pairs under warm initialisation. Left: Distribution of the negative log-conditional probability log $P ( \mathbf { a } | \mathbf { q } )$ . Right: Distribution of the negative log-fluency, log $P ( \mathbf { q } )$  
![](images/2e95e282e80f9f19b34a796836f2984fa6f21bc5552d6c85a909ef534ab7c167.jpg)

![](images/5b033fc10202d4191d9fa3d805345004bad07cc29659970d2db7210d65eec5f9.jpg)  
Figure 6: Left: Mean over $3 7 1$ questions of the log $P ( \pmb { a } , \pmb { q } )$ as function of the number of steps for the forward and backward/reverse proposals under warm-start initialisation. Right: Mean acceptance rate as function of the number of steps for the same configurations.

## A COMPARATIVE GRAPHS

Figures 5 present a comparative graph between the ground-truth, the sampling and the optimisation frameworks across all methods, under warm-start initialisation. These results show that MCMC is the only method that consistently yields prompts combining strong fluency with high confidence in the target answer. Meanwhile, if the priority is high confidence in the answer relative to the ground truth, GCG is the preferable choice.

## B CONVERGENCE OF THE METROPOLIS-HASTINGS

Figure 6 (Left) shows the empirical mean of the joint negative log-probabilities  log $P ( \pmb { a } , \pmb { q } )$ as function of the sampling step (t), computed as:

$$
\mathbb { E } _ { ( \mathbf { a } , \mathbf { q } ) \sim R } \left[ - \log P ( \mathbf { a } , \mathbf { q } ) \right] \approx - \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \log P ( { a } _ { i } , { q } _ { i } ) .
$$

We compare the forward and backward proposal strategies, both under warm-start initialisation. The warm started forward and backward chains converge to the same region of the target distribution, demonstrating that the Metropolis-Hastings sampler is well-mixed and the choice of the proposal does not materially affect the outcome. Figure 6 (right), reports the mean MH acceptance rate over the sampling steps (t). Both the forward and backward warm started chains exhibit low, stable acceptance rate $( \approx 0 . 0 2 )$ as $t \to \infty$ , indicating that, the chains are initialised directly within high density regions of the target distribution, leading to local, precise update.

<table><tr><td>Methods</td><td>Prompts</td><td>Target-Answer</td></tr><tr><td>Ground truth</td><td>who has the maximum number of wimbledon singles title</td><td>Roger Federer</td></tr><tr><td>GCG</td><td>illob!!)\rn aoeeksenpyxigmatic tennisimen</td><td>Roger Federer</td></tr><tr><td>GCG-Reg</td><td>who won the world&#x27;s defeating singles player in tennis?</td><td>Roger Federer</td></tr><tr><td>MCMC GD-PEZ</td><td>who won the world&#x27;s best men&#x27;s player award in 2018 tennis wikipedia page</td><td>Roger Federer</td></tr><tr><td></td><td>who won the world&#x27;s men singles player in tennis?</td><td>Roger Federer</td></tr><tr><td>Ground truth</td><td>who won the 2018 formula 1 french grand pix tournament</td><td>Sebastian Vettel</td></tr><tr><td>GCG</td><td>Ich GraphQL&#x27;deki scoreboard racing driver...FileNotFoundExceptioneastwstring</td><td>Sebastian Vettel</td></tr><tr><td>GCG-Reg</td><td>who unknown sponsors top racing driver in the world.</td><td>Red bull Racing</td></tr><tr><td>MCMC</td><td>who the most professional racing driver is in the current century</td><td>Sebastian Vettel</td></tr><tr><td>GD-PEZ</td><td>who the most professional racing driver is in the world?</td><td>Denny Hulme</td></tr><tr><td>Ground truth</td><td>what were the 2 sides in ww1 called</td><td>the Triple Entente; Allied Powers</td></tr><tr><td>GCG</td><td>Initially DarketCodeiles allied accordingtεpa191 gyr</td><td>Triple Entente</td></tr><tr><td>GCG-Reg</td><td>who firstust have allied in world warServletResponse</td><td>Germany</td></tr><tr><td>MCMC</td><td>was the and fought in ww1 together too?</td><td>France</td></tr><tr><td>GD-PEZ</td><td>Whoever merged European agricultural.exp意思 scientific military pollutants</td><td>the Treaty of Paris (1856) and the Berlin Conference (1884–85) of the 19th century</td></tr><tr><td>Ground truth</td><td>who is the youngest formula 1 world champion</td><td>Sebastian Vettel</td></tr><tr><td>GCG</td><td>whoafka most professionalül driver Nicoilloworth认为</td><td>Sebastian Vettel</td></tr><tr><td>GCG-Reg</td><td>who won mostBest racing driver in the world?</td><td>Sebastian Vettel</td></tr><tr><td>MCMC</td><td>what is the most racing drivers worldwide today</td><td>Lewis Hamilton</td></tr><tr><td>GD-PEZ</td><td>who the most professional racing driver is in the world?</td><td>Denny Hulme</td></tr></table>

Figure 7: Qualitative analysis of prompts learned by the optimisation-based methods and the sampling-based MCMC framework. For each target answer, the ground truth question is shown alongside the corresponding question recovered by GCG, GCG-Reg, and GD-PEZ under warm-start initialisation, and by our MCMC based sampler (final sample and forward proposal).

## C QUALITATIVE ANALYSIS RESULTS

In section 5.1, we present representative examples of prompts recovered by each method, the examples shown are not exhaustive, and we refer the reader to Figures 7 and 8 for additional results.

## D PLAUSIBILITY AND GRAMMAR SCORES PROMPTS

To generate the scores reported in Figure 4, we queried the Chatbot Arena platform (LMSYS) using the following prompt:

## LMSYS Judge Prompt for Plausibility:

You are an impartial evaluator. You are given: Four candidate questions generated by different methods, a target answer. Your task is to rank the questions according to how likely each question is to naturally elicit the target answer from a knowledgeable assistant. Evaluate the questions as if you were a careful human reviewer by considering question quality (naturalness, non English characters), semantic relevance pointing to the target (highest weight) and lexical overlapping with the target answer. Return a csv file with the winning percentage for GCG, GCG-REG, GD-PEZ, MCMC and make sure it sums to 100.

## LMSYS Judge Prompt for Grammar:

Evaluate only the grammatical quality of the following four questions (MCMC, GCGC, GCG-Reg, GD-PEZ). Ignore semantics and usefulness. Using a heuristic based on grammar, spelling, punctuation, word order, repeated words, malformed phrases, and corrupted tokens, assign each question a score between 0.0 (malformed) and 1.0 (perfect English). Return a csv file of the scores for each method.

<table><tr><td>Methods</td><td>Prompts</td><td>Target-Answer</td></tr><tr><td>Ground truth</td><td>where is the baylor vanderbilt game being played</td><td>Houston, Texas</td></tr><tr><td>GCG</td><td>&#x27;uckets cole學院 relocated 开 struuë</td><td>Houston, Texas</td></tr><tr><td>GCG-Reg</td><td>)晨:: columbia Sour candy tattuso now .</td><td>Candy</td></tr><tr><td>MCMC</td><td>the center of gravity crowd is going to take place at the park where?</td><td>the park&#x27;s central grassy area</td></tr><tr><td>GD-PEZ</td><td>neuron(cmd inches Linda&quot;,\nHouston_province_font IXauen</td><td>Linda&#x27;s neuron is located in the hypothalamus,...</td></tr><tr><td>Ground truth</td><td>when did michigan last beat ohio state in columbus</td><td>2000; 108 2011; 2018,</td></tr><tr><td>GCG</td><td>Did Catchånaf社会 novo sober到了uidocke</td><td>2017年12月8日</td></tr><tr><td>GCG-Reg</td><td>disyu possessed the most maEsuleeb performance tool</td><td>MaEsuleeb</td></tr><tr><td>MCMC</td><td>when did rabbit hole episode of friends come out?</td><td>September 25, 1998</td></tr><tr><td>GD-PEZ</td><td>kok&quot;N doors for&quot;,&quot; persone Ali Kend mand:</td><td>Kolkata</td></tr><tr><td>Ground truth</td><td>when did texas become a part of the us</td><td>December 29, 1845; 1845; In 1845</td></tr><tr><td>GCG</td><td>FIRST MONEYistributed Palmer motenohoharsgivingIn elapsed</td><td>1861</td></tr><tr><td>GCG-Reg</td><td>wcréationhopefully Khalafawaap and Koa News</td><td>Khalafawaap and Koa News</td></tr><tr><td>MCMC</td><td>when did the t book first be around</td><td>around 1960</td></tr><tr><td>GD-PEZ</td><td>Merr commemorulton infra/deho.auth desert In179</td><td>Merr commemorulton infra/deho.auth desert In1792</td></tr><tr><td>Ground truth</td><td>where does metal gear solid 3 take place</td><td>Soviet Union rainforest; a 1960s Soviet jungle setting</td></tr><tr><td>GCG</td><td>Kontersonadio Jaydetecti Cap前的 ecosystem Imagine </td><td>the ocean</td></tr><tr><td>GCG-Reg</td><td>hin#if breathed volcano Conditions Environment Wildlife Toonyidalus</td><td>mountainous region</td></tr><tr><td>MCMC GD-PEZ</td><td>what of what world was set in what show?</td><td>the 1950s</td></tr><tr><td></td><td>bi kann Soviet%% Blue cott TreeMap).\n\n Soviet jungle</td><td>Soviet Union</td></tr><tr><td>Ground truth</td><td>when did the petrified forest became a national park</td><td>1962</td></tr><tr><td>GCG</td><td>ahrn++ \u3000 \u3000 \u3000 \u3000 \u3000ton/she Lou Ellison Release</td><td>1961</td></tr><tr><td>GCG-Reg MCMC</td><td>PLAIN Vilainium Coropeter built here</td><td>1920s</td></tr><tr><td>GD-PEZ</td><td>when did the first superamas song come out Constirand\tfunc tahun-J Bip colomb200</td><td>1987 2008-09</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>Ground truth how many series of parks and recreation are there</td><td>7; seven; seven seasons</td></tr><tr><td>GCG GCG-Reg</td><td>Season awful kingdomsQRS.Ch kouyiska)); mount</td><td>Season 2</td></tr><tr><td>MCMC</td><td>sup Haz one Haz seasons of twalville?</td><td>two seasons</td></tr><tr><td>GD-PEZ</td><td>how many weeks is kestral first season kids season</td><td>8.5 episode</td></tr><tr><td></td><td>Victoriaicleseason bay Duel count simply timings calendaráng</td><td>1,2,3,4</td></tr><tr><td></td><td>Ground truth when were tomatoes brought to spain how did they end up in italy</td><td>1540s</td></tr><tr><td>GCG</td><td>utow-calCS tightened(sillo Truquetzza.ONE))}\n YEAR</td><td>1500s</td></tr><tr><td>GCG-Reg</td><td>rum distribution methodRev emergropgeoia Beronian diffusion method</td><td>diffusion of gas through a porous medium</td></tr><tr><td>MCMC GD-PEZ</td><td>who has won the most state choppers controversies since 1996,...</td><td>The Republican Party</td></tr><tr><td></td><td>Byzricques Lucodiacus dat rose secrecy])-&gt;</td><td>the rose of Byzantium</td></tr><tr><td>Ground truth</td><td>when did lebron start playing in the nba</td><td>2003</td></tr><tr><td>GCG</td><td>just -iadmal icXML,tagpop released&#x27;</td><td>2007</td></tr><tr><td>GCG-Reg</td><td>nuja thibowRelease date Invoice numb feature</td><td>2018</td></tr><tr><td>MCMC</td><td>who first season the challenge aired on dvd</td><td>2009-10 season</td></tr><tr><td>GD-PEZ</td><td>DE inclusive HRESULTuli đu&#x27;c Tayaroseconds teamoment</td><td>2017-18 season</td></tr><tr><td colspan="2">Ground truth the old testament contains how many books of history</td><td>Judges; Joshua; Kings; Samuel</td></tr><tr><td>GCG</td><td>LevYoApplicationContextFormattingWednesday.a Civilwhat scripturesued</td><td>Psalms</td></tr><tr><td>GCG-Reg</td><td>ch bub interest high &amp;oduleOf the Bible</td><td>Biblical figures who were involved in high-profile controversies</td></tr><tr><td>MCMC</td><td>how many school books are there in the bible</td><td>&quot;over 1,000 books&quot;</td></tr><tr><td>GD-PEZ</td><td>northwest Joseph Theemiah(b]]) MOS kid Babylon myths</td><td>Joseph Theemiah(b]]) MOS kid Babylon myths</td></tr></table>

Figure 8: Qualitative analysis of prompts learned by the optimisation-based methods and the sampling-based MCMC framework. For each target answer, the ground truth question is shown alongside the corresponding question recovered by GCG, GCG-Reg, and GD-PEZ under random initialisation, and by our MCMC based sampler (final sample and forward proposal).