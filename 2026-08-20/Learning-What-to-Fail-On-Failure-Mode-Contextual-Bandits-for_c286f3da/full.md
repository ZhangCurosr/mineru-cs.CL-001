# Learning What to Fail On: Failure-Mode Contextual Bandits for Adversarial Data Curation

<table><tr><td>Roie Kazoom¹*</td><td>roieka@post.bgu.ac.il</td></tr><tr><td>Ofir Cohen2*</td><td>cofir@post.bgu.ac.il</td></tr><tr><td>Rami Puzis²</td><td>puzis@bgu.ac.il</td></tr><tr><td>Asaf Shabtai²</td><td>shabtaia@bgu.ac.il</td></tr><tr><td>Ofer Hadar¹</td><td>hadar@bgu.ac.il</td></tr></table>

<sup>1</sup>Electrical and Computers Engineering, Ben Gurion University of The Negev

<sup>2</sup>Software and Information Systems Engineering, Ben Gurion University of The Negev

Reviewed on OpenReview: https: // openreview. net/ forum? id= DSpKdN6whZ

![](images/0b225a1d4a8980b40c4a6720a57c17b27f054b7d5c5b6043ba2414600650c885.jpg)  
Figure 1: Overview of the proposed failure-mode contextual bandit curation framework. Given an input-labe pair, retrieval-augmented prompting generates candidate outputs. The target model filters candidates by retaining only incorrect predictions, which are then automatically validated by an LLM judge ensemble and clustered into recurring failure modes. A contextual-bandit policy selects failure modes under an adversaria budget, and selected examples are mixed with original data to retrain the target model. Validation feedback provides reward $G _ { t }$ for updating the policy $\pi _ { \theta }$ and critic $R _ { \phi } .$ , enabling adaptive selection of high-impact failure modes across rounds.

## Abstract

We introduce a failure-aware adversarial retrieval-augmented framework for improving robustness in natural language understanding. Rather than selecting synthetic examples with a fixed reward threshold, our method formulates adversarial data curation as a failure-mode contextual bandit problem. Candidate examples are generated with retrieval-augmented prompting, filtered by the current target model, automatically validated by an LLM judge ensemble, and clustered into recurring failure modes. A stochastic policy then selects which failure modes to sample for retraining, and is updated using validation-based reward that balances robustness gains, forgetting, and data cost. This makes the data curator itself the learning agent, enabling adaptive selection of the most useful model failures across training rounds. On standard benchmarks, our approach improves RoBERTa-base accuracy from

88.48% to 92.60% on SNLI, from 75.04% to 80.95% on ANLI, and from 54.67% to 71.99% on MultiNLI, while consistently outperforming prior adversarial augmentation methods. We further demonstrate transfer to FEVER fact verification, achieving up to 79.86% FEVER score and 82.45% accuracy with RoBERTa-large. Finally, we provide a theoretical interpretation showing that, under stated assumptions, failure-mode sampling can reduce shortcutaligned gradient contributions while inducing bounded distributional drift. By combining retrieval, automated validation, contextual-bandit failure selection, and controlled adversarial retraining, our framework enables scalable robustness improvement without additional human annotation.

## 1 Introduction

Natural Language Inference (NLI), the task of determining whether a hypothesis is entailed by, contradicted by, or neutral with respect to a given premise, is a central component of many natural language understanding problems, including question answering, summarization, dialogue systems, and fact verification. More broadly, many supervised NLP tasks sufer from similar robustness limitations when confronted with adversarial or out-of-domain examples. Despite rapid progress, even state-of-the-art models remain brittle, often relying on spurious lexical cues or failing under simple syntactic and semantic variations (Glockner et al., 2018; Carmona et al., 2018). Benchmarks such as SNLI and MultiNLI (Bowman et al., 2015; Williams et al., 2018), as well as adversarial datasets (Nie et al., 2019), have driven robustness improvements but incur high annotation costs and still leave many failure modes uncovered. More recently, large-scale synthetic datasets such as GNLI (Hosseini et al., 2024) have been generated automatically, but their largely untargeted nature can dilute the adversarial patterns most useful for improving a particular target model. Existing adversarial data generation pipelines typically rely on static filtering, heuristic selection rules, or one-shot validation. As a result, they do not explicitly learn which types of failures should be prioritized as the target model evolves. This limitation is especially important because model failures are not equally useful: some expose persistent decision shortcuts, while others are noisy, redundant, or too easy to produce meaningful robustness gains. Motivated by this gap, we formulate adversarial data curation as a failure-mode contextual bandit problem. The learning agent is not the target classifier itself, but the data curator that decides which types of validated model failures should be sampled for retraining. Our framework first retrieves label-balanced few-shot contexts using both semantic embeddings (BGE M3 (Chen et al., 2024)) and lexical matching (BM25 (Robertson & Zaragoza, 2009)). These contexts are assembled into LLM prompts to generate challenging candidate hypotheses. Each candidate is evaluated by the current target model, and only examples that induce misclassification are passed to an automated LLM judge ensemble for label validation. The validated failures are then embedded and clustered into recurring failure modes, such as lexical shortcut failures, negation errors, entity mismatch errors, numerical reasoning failures, or contradiction confusion. A stochastic contextual-bandit policy then observes a state vector for each failure mode, including cluster size, target-model loss, uncertainty, classification margin, label distribution, retrieval score, judge agreement, novelty, and previous reward statistics. The policy selects which failure modes to sample under a fixed adver sarial budget. After retraining the target model on a controlled mixture of original and selected adversaria examples, the policy receives a validation-based reward that balances robustness improvement, forgetting on the clean distribution, and data cost. A lightweight critic estimates the expected utility of each failure mode, but selection is governed by the learned policy rather than a fixed reward threshold. This design provides an explicit policy, action space, reward signal, and policy-optimization procedure for adaptive adversaria data curation Kazoom et al. (2024a). In human-free adversarial fine-tuning and transfer evaluations on NLI benchmarks, our approach improves RoBERTa-base accuracy from 88.48% to 92.60% on SNLI, from 75.04% to 80.95% on ANLI, and from 54.67% to 71.99% on MultiNLI, while consistently outperforming prior adversarial augmentation methods. Beyond NLI, we further demonstrate transfer to the FEVER fact verification benchmark. Using RoBERTa-large, our method achieves up to 79.86% FEVER score and 82.45% label ac curacy, outperforming strong retrieval-augmented and synthetic-data baselines. These results indicate that failure-mode bandit curation can improve robustness across task formulations and supervision regimes while using automatically generated and automatically validated data. Our contributions are:

Framework. We propose a failure-mode contextual bandit framework for adversarial data curation. The framework requires no additional human annotation and learns which validated model-failure modes should be sampled for retraining.

Methodology. We introduce an adaptive curation pipeline that combines label-balanced retrieval, LLMbased candidate generation, target-model failure filtering, automated judge validation, unsupervised failuremode clustering, and contextual-bandit selection under an adversarial data budget.

Empirical Evaluation. We provide empirical evidence that failure-mode bandit curation improves robustness and data eficiency compared with static adversarial augmentation, reward-threshold filtering, retrievalonly selection, and untargeted synthetic-data baselines.

Theoretical Interpretation. We provide an analytical interpretation showing that, under stated assumptions, failure-mode sampling can reduce shortcut-aligned gradient contributions while preserving core-feature contributions. We further show that mixture-based updates induce bounded distributional drift and that bounded utility noise causes bounded distortion in the induced sampling policy.

## 2 Background and Related Work

Improving the robustness of NLI models remains a central challenge in natural language understanding (Glockner et al., 2018; Carmona et al., 2018). Benchmarks such as SNLI (Bowman et al., 2015) and MultiNLI (Williams et al., 2018) have enabled large-scale supervised training, while ANLI (Nie et al., 2019) introduced a human-and-model-in-the-loop protocol for collecting harder adversarial examples. However, these datasets require substantial annotation efort and still leave many systematic failure modes uncovered. More recently, automated and synthetic data-generation approaches have reduced the need for human annotation. For example, GNLI (Hosseini et al., 2024) shows that large-scale generated NLI data can rival human-curated data, and Kazoom et al. (2025c) proposed a training-free retrieval-augmented framework for adversaria detection and filtering. Related work on counterfactual and paraphrase generation has also been used to enrich training distributions (Li et al., 2023; Klemen & Robnik-Šikonja, 2021; Kazoom et al., 2024b). Despite these advances, most existing approaches rely on static generation, heuristic filtering, or one-shot validation, and therefore do not explicitly learn which types of model failures should be prioritized as the target model evolves.

Adversarial and Synthetic Example Generation. Automated adversarial pipelines aim to expose and correct model weaknesses without manual curation. Minervini & Riedel (2018); Kazoom et al. (2025a) generate logical-constraint-violating examples, improving robustness on SNLI and MultiNLI. Nie et al. (2020) use a model-in-the-loop setup to surface challenging examples and improve out-of-domain transfer. Iyyer et al. (2018); Kazoom et al. (2025b) introduce syntactically controlled transformations for paraphrase-based attacks. Recent LLM-based pipelines further demonstrate that generated hypotheses, especially when combined with automated validation, can provide useful adversarial supervision. However, these methods typically select examples using fixed rules, confidence thresholds, heuristic filters, or limited human feedback. In contrast, our work treats adversarial data curation as an adaptive learning problem: the system identifies recurring target-model failures, clusters them into failure modes, and learns which modes are most useful for retraining Kazoom et al. (2026).

Retrieval for Few-Shot Prompting. Few-shot retrieval is important for reliable LLM-based generation because the retrieved context controls both the label distribution and the semantic structure of generated examples. Dense retrieval models such as BGE (Chen et al., 2024) provide semantic similarity, while lexical methods such as BM25 (Robertson & Zaragoza, 2009) capture surface-level overlap and exact lexical cues. Hybrid retrieval can therefore provide complementary context for generating diverse and label-consistent hypotheses. In our framework, retrieval is not the main contribution by itself. Instead, it serves as the first stage of a larger closed-loop curation pipeline: retrieved examples guide generation, generated candidates expose target-model failures, and the downstream bandit policy decides which validated failure modes should be sampled for retraining.

Reinforcement Learning, Bandits, and Data Selection. Learning to select training examples has been studied in reinforcement-learning and curriculum-learning settings. Fan et al. (2017) propose a Neural Data

![](images/a850fedec36b619d2494a06fb84102d4f3f7f69d061228a9d70b7639e526a097.jpg)  
Figure 2: Example of the progressive construction of our failure-aware policy. From left to right: using only an LLM for hypothesis generation without retrieval or feedback; adding few-shot retrieval to condition generation on similar failures; and the full reinforcement-guided policy, which combines retrieval, multimodel validation, and reward-based selection to identify and reinject high-impact adversarial updates.

Filter that learns to select useful training samples. Reinforcement-guided curricula have also been explored in structured prediction tasks such as neural machine translation, where policies learn to sequence or weight examples for improved training (Zhao et al., 2020). More recent work formulates data selection for model finetuning as a sequential decision problem, where an agent chooses subsets of data to optimize validation rewards (Jha et al., 2025). Related methods such as LearnAlign (Li et al., 2025) and RL-Selector (Yang et al., 2025) use reward feedback to select informative examples or reduce redundancy. Our method difers from these approaches in both the unit of selection and the source of supervision. Rather than selecting arbitrary training examples, we first construct a validated pool of adversarial failures induced by the current target model. These failures are then clustered into semantic failure modes, and a contextual-bandit policy selects which modes to sample under a fixed adversarial budget. The reward is computed after retraining, using validation improvement, forgetting penalty, and data cost. This provides an explicit state, action, reward, and policy update while avoiding the instability and expense of per-example utility estimation.

## 3 Methodology

Let $\mathcal { D } = \{ ( p _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ denote the NLI training set, where each premise $p _ { i }$ is paired with a label $y _ { i } \in \mathcal { V }$ and $\mathcal { V } =$ {entail, neutral, contradict}. We denote by $M ^ { ( t ) }$ the target model after t rounds of failure-aware adversarial data curation, with $M ^ { ( 0 ) }$ trained on D. The goal is to construct a compact adversarial training set that improves robustness without relying on large volumes of untargeted synthetic examples. We formulate this process as failure-mode contextual bandit curation. At each iteration, the system retrieves few-shot contexts, generates candidate adversarial examples, filters candidates that induce errors in the current target model, and validates them using an automated judge ensemble. The validated failures are then grouped into semantic failure modes. Instead of selecting individual examples by a fixed reward threshold, a trainable stochastic policy $\pi _ { \theta }$ observes each failure mode and decides which modes should be sampled for retraining. After the target model is updated, the policy receives a validation-based reward and is optimized using a policygradient objective. Thus, the learning agent is the data curator, whose role is to learn which types of model failures are most useful for improving future robustness. This formulation makes the reinforcement-learning component explicit. The state is a vector of statistics describing a failure mode, the action is whether to sample from that failure mode, the reward is the downstream validation gain after retraining, and the policy is updated to maximize expected validation improvement while penalizing forgetting and computationa cost. Each iteration consists of six steps: label-balanced retrieval, LLM-based candidate generation, failure filtering using the current target model, automated validation, clustering validated candidates into failure modes, and contextual-bandit selection of failure modes for retraining.

Retrieval. For each premise $p ,$ we construct a label-balanced few-shot context: $\begin{array} { r } { \mathcal { C } _ { p } = \bigcup _ { y ^ { \prime } \in \mathcal { y } } \mathcal { C } _ { p , y ^ { \prime } } } \end{array}$ , where $\mathcal { C } _ { p , y ^ { \prime } }$ contains k examples with label $y ^ { \prime } .$ . Let $\mathcal { D } _ { y ^ { \prime } }$ denote the subset of training examples with label $y ^ { \prime }$ . Labelbalanced retrieval prevents the prompt from being dominated by a single class and provides the generator with controlled examples from all NLI relations.

Semantic Retrieval. Let $E _ { \mathrm { e m b } }$ denote the embedding model. Each premise x is embedded as: $e _ { x } =$   
$E _ { \mathrm { e m b } } ( x ) \in \mathbb { R } ^ { d }$ . For a query premise $p ,$ we compute: $e _ { p } = E _ { \mathrm { e m b } } ( p )$ . For each label $y ^ { \prime } { \mathrm { . } }$ semantic neighbors   
are selected by: $\begin{array} { r } { \mathcal { C } _ { p , y ^ { \prime } } ^ { \mathrm { s e m } } = \arg \operatorname* { m a x } _ { S \subseteq \mathcal { D } _ { y ^ { \prime } } } \sum _ { x \in S } \cos ( e _ { p } , e _ { x } ) } \end{array}$ . This corresponds to selecting the top-k nearest |S|=k   
neighbors in embedding space within each label group.

Lexical Retrieval. We index all premises using BM25 with parameters $( k _ { 1 } = 1 . 5 , b = 0 . 7 5 )$ and define the lexical relevance score: s<sub>BM25</sub> $\begin{array} { r } { ( p , x ) = \sum _ { w \in p } \mathrm { I D F } ( w ) \cdot \frac { \mathrm { t f } ( w , x ) ( k _ { 1 } + 1 ) } { \mathrm { t f } ( w , x ) + k _ { 1 } \left( 1 - b + b \frac { | x | } { \mathrm { a v g d } 1 } \right) } } \end{array}$ . For each label $y ^ { \prime } .$ , lexical neighbors are selected as: $\mathcal { C } _ { p , y ^ { \prime } } ^ { \mathrm { l e x } } = \arg \operatorname* { m a x } _ { S \subseteq \mathcal { D } _ { y ^ { \prime } } } \sum _ { x \in S } s _ { \mathrm { B M 2 5 } } ( p , x )$

Hybrid Retrieval. To integrate semantic and lexical signals, for each candidate x and query $p ,$ we compute normalized scores: $\begin{array} { r } { \tilde { s } _ { \mathrm { s e m } } ( p , x ) = \frac { \cos \big ( E _ { \mathrm { e m b } } ( p ) , E _ { \mathrm { e m b } } ( x ) \big ) - \mu _ { \mathrm { s e m } } } { \sigma _ { \mathrm { r e m } } } } \end{array}$ , and: $\begin{array} { r } { \tilde { s } _ { \mathrm { l e x } } ( p , x ) = \frac { s _ { \mathrm { B M 2 5 } } ( p , x ) - \mu _ { \mathrm { l e x } } } { \sigma _ { 1 \dots } } } \end{array}$ . The hybrid relevance score is defined as: $s _ { \mathrm { c o m b } } ( p , x ) = \stackrel { \sim } { \alpha } \tilde { s } _ { \mathrm { s e m } } ( p , x ) + ( 1 - \alpha ) \tilde { s } _ { \mathrm { l e x } } ( p , x )$ , where $\alpha \in [ 0 , 1 ]$ controls the interpolation between semantic and lexical similarity. For each label $y ^ { \prime } { \mathrm { . } }$ , we select: $\mathcal { C } _ { p , y ^ { \prime } } ^ { \mathrm { c o m b } } =$ arg max $\begin{array} { r l } & { s \subseteq _ { \overline { { \varphi } } _ { y ^ { \prime } } } \sum _ { x \in S } s _ { \mathrm { c o m b } } ( p , x ) } \\ & { \ | S | = k } \end{array}$ . The final context is: $\begin{array} { r } { \mathcal { C } _ { p } = \bigcup _ { y ^ { \prime } \in \mathcal { y } } \mathcal { C } _ { p , y ^ { \prime } } ^ { m } , } \end{array}$ where $m \in \{ \mathrm { s e m }$ , lex, comb} denotes the retrieval mode. The complete retrieval is summarized in Algorithm 1.

Algorithm 1 Balanced Few-Shot Context Retrieval   
Input: Premise $p ,$ label-partitioned dataset $\{ \mathcal { D } _ { y } \} _ { y \in \mathcal { Y } }$   
Parameter: examples per label $k ,$ , retrieval mode m ∈ {sem, lex, comb}   
Output: Few-shot context $\mathcal { C } _ { p }$   
1: $\mathcal { C } _ { p } \gets \emptyset$   
2: Compute retrieval scores $s _ { m } ( p , x )$ for all $x \in \mathcal { D }$   
3: for each label $y ^ { \prime } \in \mathcal { V }$ do   
4: $\mathcal { C } _ { p , y ^ { \prime } } \gets$ arg ma $\boldsymbol { \mathrm { x } } _ { x \in \mathcal { D } _ { y ^ { \prime } } } ^ { k } \boldsymbol { s } _ { m } ( p , x )$   
5: $\mathcal { C } _ { p }  \mathcal { C } _ { p } \cup \mathcal { C } _ { p , y ^ { \prime } }$   
6: end for   
7: return $\mathcal { C } _ { p }$

Task-Specific Candidate Generation. Given an input x, its label $y ,$ and the retrieved context ${ \mathcal { C } } _ { x }$ , we employ a large language model to sample task-specific candidate outputs from:

$$
o \sim P _ { \mathrm { L L M } } ( o \mid x , \mathcal { C } _ { x } , y ) .\tag{1}
$$

This stochastic generation process produces a candidate set ${ \mathcal { O } } _ { x } ^ { ( t ) }$ for each input at iteration t, where o denotes a task-dependent output, such as a hypothesis for NLI or an evidence claim for fact verification.

Failure-Based Filtering. Each generated candidate $o \in \mathcal { O } _ { x } ^ { ( t ) }$ is evaluated by the current target model $M ^ { ( t ) }$ . Let the predicted label be:

$$
\hat { y } _ { o } = \arg \operatorname* { m a x } _ { y ^ { \prime } \in \mathcal { V } } M ^ { ( t ) } ( y ^ { \prime } \mid x , o ) .\tag{2}
$$

Candidates that are correctly classified are discarded, and only misclassified instances are retained:

$$
\mathcal { O } _ { x } ^ { \mathrm { f a i l } } = \left\{ o \in \mathcal { O } _ { x } ^ { ( t ) } \mid \hat { y } _ { o } \neq y \right\} .\tag{3}
$$

This step focuses the remaining pipeline on examples that expose the current model’s weaknesses.

Automated Validation. Let the candidate adversarial triples be:

$$
{ \mathcal { Q } } ^ { ( t ) } = \left\{ ( x , o , y ) ~ | ~ o \in { \mathcal { O } } _ { x } ^ { \mathrm { f a i l } } \right\} .\tag{4}
$$

Each triple is evaluated by an ensemble of automated judge models. Let the predicted label of judge j be:

$$
v _ { j } ( x , o ) = M _ { j } ( x , o ) , \qquad j \in \{ 1 , 2 , 3 \} .\tag{5}
$$

A candidate is retained only if all judges agree with the original label: $\textstyle \sum _ { j = 1 } ^ { 3 } \mathbb { I } \left[ v _ { j } ( x , o ) = y \right] = 3$ . The validated failure pool is therefore: $\begin{array} { r } { \mathcal { V } ^ { ( t ) } \ = \ \Big \{ ( x , o , y ) \in \mathcal { Q } ^ { ( t ) } \ | \sum _ { j = 1 } ^ { 3 } \mathbb { I } \left[ v _ { j } ( x , o ) = y \right] = 3 \Big \} } \end{array}$ . This unanimity constraint reduces label noise and prevents the policy from learning from corrupted reward signals.

Failure-Mode Construction. Rather than selecting individual failures independently, we group validated failures into failure modes. For each validated triple $q _ { i } = ( x _ { i } , o _ { i } , y _ { i } ) \in \mathcal { V } ^ { ( t ) }$ , we compute an embedding:

$$
g _ { i } = E _ { \mathrm { f a i l } } \left( \left[ x _ { i } ; o _ { i } ; y _ { i } \right] \right) ,\tag{6}
$$

where $E _ { \mathrm { f a i l } }$ may be the same text embedding model used for retrieval or a separate sentence encoder. We then cluster the validated failure pool:

$$
\left\{ \mathcal { F } _ { 1 } ^ { ( t ) } , \ldots , \mathcal { F } _ { K _ { t } } ^ { ( t ) } \right\} = \operatorname { C l u s t e r } \left( \left\{ g _ { i } \right\} _ { q _ { i } \in \mathcal { V } ^ { ( t ) } } \right) .\tag{7}
$$

Each cluster $\mathcal { F } _ { k } ^ { ( t ) }$ represents a failure mode, such as lexical shortcut failures, negation errors, entity mismatch errors, numerical reasoning failures, or contradiction confusion. The clustering is unsupervised and does not require human failure labels.

Bandit State. For each failure mode $\mathcal { F } _ { k } ^ { ( t ) }$ , we construct a state vector $z _ { t , k }$ containing normalized statistics of that cluster:

$$
\begin{array} { r } { z _ { t , k } = \left[ \log ( | \mathcal { F } _ { k } ^ { ( t ) } | + 1 ) , \bar { \ell } _ { t , k } , \bar { H } _ { t , k } , \bar { \mu } _ { t , k } , \mathrm { h i s t } _ { t , k } ( y ) , \bar { s } _ { t , k } ^ { \mathrm { r e t r } } , \bar { a } _ { t , k } ^ { \mathrm { i u d g e } } , \nu _ { t , k } , \bar { G } _ { t - 1 , k } \right] . } \end{array}\tag{8}
$$

Here, $\bar { \ell } _ { t , k }$ is the mean target-model loss on the cluster, $\bar { H } _ { t , k }$ is the mean predictive entropy, $\bar { \mu } _ { t , k }$ is the mean classification margin, hist ${ \bf \Phi } _ { ; , k } ( y )$ is the label distribution, $\bar { s } _ { t , k } ^ { \mathrm { r e t r } }$ is the average retrieval score, $\bar { a } _ { t , k } ^ { \mathrm { j u d g e } }$ is the average judge agreement, $\nu _ { t , k }$ is a novelty score measuring distance from previously selected failure modes, and $\bar { G } _ { t - 1 , k }$ is the previous moving-average reward associated with similar clusters. This state summarizes both the dificulty and the diversity of each failure mode.

Contextual-Bandit Policy. The policy $\pi _ { \theta }$ is a trainable stochastic selector over failure modes. For each cluster state $z _ { t , k }$ , the policy outputs a Bernoulli distribution:

$$
\pi _ { \boldsymbol { \theta } } ( a _ { t , k } = 1 \mid z _ { t , k } ) = \sigma \left( f _ { \boldsymbol { \theta } } ( z _ { t , k } ) \right) ,\tag{9}
$$

where $a _ { t , k } \in \{ 0 , 1 \}$ is the action for cluster k, $a _ { t , k } = 1$ means selecting that failure mode for retraining, and $f _ { \theta }$ is a small neural network. The complete action at iteration t is:

$$
a _ { t } = ( a _ { t , 1 } , \dots , a _ { t , K _ { t } } ) .\tag{10}
$$

To encourage exploration, actions are sampled from π<sub>θ</sub> during training rather than selected deterministically.   
At evaluation time, the policy may use greedy selection according to the learned probabilities.

Given an adversarial budget $B _ { \mathrm { a d v } }$ , the selected adversarial set is:

$$
\mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) } = \bigcup _ { k : a _ { t , k } = 1 } \mathrm { S a m p l e } \left( \mathcal { F } _ { k } ^ { ( t ) } , n _ { t , k } \right) ,\tag{11}
$$

where:

$$
n _ { t , k } = \left\lfloor B _ { \mathrm { a d v } } \frac { a _ { t , k } | \mathcal { F } _ { k } ^ { ( t ) } | } { \sum _ { \ell = 1 } ^ { K _ { t } } a _ { t , \ell } | \mathcal { F } _ { \ell } ^ { ( t ) } | } \right\rfloor .\tag{12}
$$

If no cluster is selected, the cluster with the highest policy probability is selected as a fallback. This avoids empty updates.

Target Model Retraining. The target model is updated using supervised learning on a mixture of original and selected adversarial examples:

$$
\boldsymbol { M } ^ { ( t + 1 ) } \gets \operatorname { T r a i n } \left( \boldsymbol { M } ^ { ( t ) } , \mathcal { D } _ { \operatorname* { m i x } } ^ { ( t ) } \right) .\tag{13}
$$

The construction of $\mathcal { D } _ { \operatorname* { m i x } } ^ { ( t ) }$ is described in Section 3.2. This update changes the environment observed by the curator, since future failure pools depend on the updated target model $M ^ { ( t + 1 ) }$

Validation-Based Reward. After retraining, the policy receives a scalar reward based only on validation performance. Let $\mathcal { D } _ { \mathrm { v a l } } ^ { \mathrm { r o b } }$ denote the robustness validation set and $\mathcal { D } _ { \mathrm { v a l } } ^ { \mathrm { c l e a n } }$ denote the clean validation set. We define:

$$
G _ { t } = \Delta _ { \mathrm { r o b } } ^ { ( t ) } - \beta _ { f } \Delta _ { \mathrm { f o r g e t } } ^ { ( t ) } - \beta _ { c } \Delta _ { \mathrm { c o s t } } ^ { ( t ) } .\tag{14}
$$

The robustness gain is:

$$
\Delta _ { \mathrm { r o b } } ^ { ( t ) } = \mathrm { P e r f } \left( M ^ { ( t + 1 ) } , \mathcal { D } _ { \mathrm { v a l } } ^ { \mathrm { r o b } } \right) - \mathrm { P e r f } \left( M ^ { ( t ) } , \mathcal { D } _ { \mathrm { v a l } } ^ { \mathrm { r o b } } \right) .\tag{15}
$$

The forgetting penalty is:

$$
\Delta _ { \mathrm { f o r g e t } } ^ { ( t ) } = \operatorname* { m a x } \left( 0 , \mathrm { P e r f } \left( M ^ { ( t ) } , \mathcal { D } _ { \mathrm { v a l } } ^ { \mathrm { c l e a n } } \right) - \mathrm { P e r f } \left( M ^ { ( t + 1 ) } , \mathcal { D } _ { \mathrm { v a l } } ^ { \mathrm { c l e a n } } \right) \right) .\tag{16}
$$

The cost penalty is:

$$
\Delta _ { \mathrm { c o s t } } ^ { ( t ) } = \frac { | \mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) } | } { B _ { \mathrm { a d v } } } .\tag{17}
$$

The first term rewards robustness gains, the second penalizes degradation on the original distribution, and the third penalizes excessive adversarial data usage. The coeficients $\beta _ { f }$ and $\beta _ { c }$ control the trade-of between robustness, retention, and eficiency.

Policy Optimization. The curator policy is optimized to maximize expected validation reward:

$$
J ( \theta ) = \mathbb { E } _ { a _ { t } \sim \pi _ { \theta } } \left[ G _ { t } \right] .\tag{18}
$$

We update $\pi _ { \theta }$ with a Reinforce-style policy-gradient objective:

$$
\mathcal { L } _ { \pi } ( \theta ) = - ( G _ { t } - b _ { t } ) \sum _ { k = 1 } ^ { K _ { t } } \log \pi _ { \theta } ( a _ { t , k } \mid z _ { t , k } ) - \beta _ { H } \sum _ { k = 1 } ^ { K _ { t } } \mathcal { H } \left( \pi _ { \theta } ( \cdot \mid z _ { t , k } ) \right) ,\tag{19}
$$

where $b _ { t }$ is a moving-average baseline and $\mathcal { H } ( \cdot )$ is an entropy regularizer that encourages exploration. The baseline is updated as:

$$
b _ { t } = \rho b _ { t - 1 } + ( 1 - \rho ) G _ { t } .\tag{20}
$$

We also maintain a critic $R _ { \phi }$ that predicts the expected return of a failure mode:

$$
R _ { \phi } ( \boldsymbol { z } _ { t , k } ) \approx \mathbb { E } \left[ G _ { t } \ | \ \boldsymbol { z } _ { t , k } \right] .\tag{21}
$$

The critic is trained by minimizing:

$$
\mathcal { L } _ { R } ( \phi ) = \sum _ { k : a _ { t , k } = 1 } \left( R _ { \phi } ( z _ { t , k } ) - G _ { t } \right) ^ { 2 } .\tag{22}
$$

Unlike a threshold-based filtering rule, $R _ { \phi }$ is not used as a direct keep-or-discard mechanism. Instead, it serves as a learned utility estimator and variance-reduction signal for policy learning. The actual data-selection behavior is governed by the trainable policy $\pi _ { \theta } .$ . The complete procedure is summarized in Algorithm 2.

Algorithm 2 Failure-Mode Contextual Bandit Curation   
Input: Training set D, validation sets D<sup>rob</sup><sub>val</sub> , val , $\mathcal { D } _ { \mathrm { v a l } } ^ { \mathrm { c l e a n } }$ , iterations T, retrieval mode m, budget $B _ { \mathrm { a d v } }$   
Output: Enhanced model $M ^ { ( T ) }$   
1: Train $M ^ { ( 0 ) }$ on $\mathcal { D }$   
2: Initialize policy $\pi _ { \theta } ,$ critic $R _ { \phi } .$ , and baseline $b _ { 0 }$   
3: for $t = 0$ to $T - 1$ do   
4: V<sup>(t)</sup> ← GENERATE\_AND\_VALIDATE(M<sup>(t)</sup>,D,m)   
5: $\{ \mathcal { F } _ { k } ^ { ( t ) } \} _ { k = 1 } ^ { K _ { t } }$ ← CLUSTER\_FAILURES(V<sup>(t)</sup>)   
6: for each failure mode $\mathcal { F } _ { k } ^ { ( t ) }$ do   
7: construct state $z _ { t , k }$ and sample $a _ { t , k } \sim \pi _ { \theta } ( \cdot \mid z _ { t , k } )$   
8: end for   
9: $\mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) }$ ← BUDGETED\_SAMPLE( $\{ \mathcal { F } _ { k } ^ { ( t ) } : a _ { t , k } = 1 \} , B _ { \mathrm { a d v } } )$   
10: $\mathcal { D } _ { \operatorname* { m i x } } ^ { ( t ) }  \mathrm { M I X } ( \mathcal { D } , \mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) } )$   
11: $M ^ { ( t + 1 ) } \gets \mathrm { T r a i n } ( \bar { M ^ { ( t ) } } , \mathcal { D } _ { \mathrm { m i x } } ^ { ( t ) } )$   
12: compute reward $G _ { t }$ on $\mathcal { D } _ { \mathrm { v a l } } ^ { \mathrm { r o b } }$ and $\mathcal { D } _ { \mathrm { v a l } } ^ { \mathrm { c l e a } }$ n   
13: update π<sub>θ</sub>, $R _ { \phi } ,$ and $b _ { t }$ using $G _ { t }$   
14: end for   
15: return $M ^ { ( T ) }$

Critic Architecture and Reward Supervision. The reward model $R _ { \phi }$ is implemented as a lightweight MLP critic operating on failure-mode states rather than individual examples. Its input is the cluster-level state vector $z _ { t , k }$ defined above, which contains the cluster size, mean target-model loss, entropy, classification margin, label distribution, retrieval score, judge agreement, novelty score, and previous reward statistics. The critic outputs a scalar utility estimate:

$$
R _ { \phi } ( z _ { t , k } ) \in \mathbb { R } ,\tag{23}
$$

which approximates the expected validation reward obtained by selecting failure mode $\mathcal { F } _ { k } ^ { ( t ) }$

Importantly, we do not compute a separate utility value $\Delta ( h )$ for each individual candidat ${ \mathrm { , e , } }$ since doing so would require prohibitively expensive per-example retraining. Instead, reward supervision is defined at the failure-mode level. After the policy selects a subset of failure modes, the target model is retrained once on the resulting adversarial mixture, and the scalar validation reward $G _ { t }$ is computed from robustness improvement, forgetting penalty, and data cost. This same observed return is assigned to all selected failure modes in that round and used to train the critic:

$$
\mathcal { L } _ { R } ( \phi ) = \sum _ { k : a _ { t , k } = 1 } \left( R _ { \phi } ( z _ { t , k } ) - G _ { t } \right) ^ { 2 } .\tag{24}
$$

The critic is updated once after each retraining round, together with the policy update. Since selection is performed by the stochastic policy $\pi _ { \boldsymbol { \theta } } \big ( a _ { t , k } \mid z _ { t , k } \big )$ , the method does not require a fixed reward threshold τ. This removes the threshold-tuning step used in reward-filtering pipelines and replaces it with validationdriven policy optimization.

## 3.1 Hyperparameter Tuning for Retrieval

Standard training on $P$ may reinforce spurious correlations that fail under distribution shift. Our framework instead concentrates updates on validated failure modes, where such shortcuts are more likely to break and task-relevant reasoning is required. Lemma 3.1 formalizes this intuition by showing that, under stated assumptions, failure-focused sampling reduces shortcut-aligned gradient contributions while increasing semantically meaningful ones. To initialize the hybrid retrieval score, we tune the semantic weight α on 1,000 SNLI training examples using BGE M3. For calibration only, we retrieve a candidate pool of

9 examples per label and treat each pair $( p , x )$ as relevant if $\mathrm { l a b e l } ( x ) = \mathrm { l a b e l } ( p )$ The hybrid score is: $s _ { \mathrm { c o m b } } ( p , x ) = \alpha \tilde { s } _ { \mathrm { s e m } } ( p , x ) + ( 1 - \alpha ) \tilde { s } _ { \mathrm { l e x } } ( p , x )$ . We search $\alpha \in \{ 0 , 0 . 0 1 , \ldots , 1 . 0 \}$ and select the value with the highest ROC AUC over positive and negative pairs. The best value is $\alpha ^ { * } = 0 . 8 3$ , achieving an AUC of 0.93 in Figure 4. We fix $\alpha = 0 . 8 3$ for all downstream experiments. In the main six-shot setting, we use $k = 2$ examples per label. This tuning only initializes retrieval; later adaptation is performed by the contextual-bandit failure-mode policy.

![](images/fb73196a678feb30bc584665589fd3a46d615bf0f91631d74da17eb889171adf.jpg)  
Figure 3: ROC AUC as a function of the semantic-lexical weighting parameter α.

![](images/c4e53c97807d213e4273d21820d803b98cc755692c32abc8b624cae83aef8d16.jpg)  
Figure 4: ROC curve at optimal $\alpha = 0 . 8 3$

## 3.2 Avoiding Forgetting

Training only on adversarial examples can induce a non-stationary training distribution and lead to catastrophic forgetting, where performance on the original data distribution deteriorates. In our setting, this risk is especially important because the policy is explicitly encouraged to focus on dificult failure modes. To stabilize training, we mix original and selected adversarial examples during retraining. Let $\mathcal { D } _ { \mathrm { o r i g } }$ denote the original training set and $\mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) }$ denote the adversarial examples selected by the policy at iteration t. We define the original-to-adversarial mixing ratio:

$$
\lambda _ { \mathrm { m i x } } = \frac { | \mathcal { D } _ { \mathrm { o r i g } } | } { | \mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) } | } \in \left\{ 0 , 1 , \frac { 1 } { 2 } , \frac { 1 } { 3 } , \frac { 1 } { 4 } \right\} .\tag{25}
$$

Here, $\lambda _ { \operatorname* { m i x } } = 0$ corresponds to training only on selected adversarial examples, while $\begin{array} { r } { \lambda _ { \operatorname* { m i x } } = \frac { 1 } { 4 } } \end{array}$ denotes one original example per four adversarial examples. For $\lambda _ { \operatorname* { m i x } } > 0$ , we construct:

$$
\mathcal { D } _ { \operatorname* { m i x } } ^ { ( t ) } ( \lambda _ { \operatorname* { m i x } } ) = \mathcal { D } _ { \operatorname { o r i g } } \cup \operatorname { S a m p l e } \left( \mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) } , \big \lfloor \lambda _ { \operatorname* { m i x } } ^ { - 1 } \big \lvert \mathcal { D } _ { \operatorname { o r i g } } \big \rvert \big \rfloor \right) .\tag{26}
$$

For $\lambda _ { \operatorname* { m i x } } = 0$ , we set: $\mathcal { D } _ { \mathrm { m i x } } ^ { ( t ) } ( 0 ) = \mathcal { D } _ { \mathrm { s e l } } ^ { ( t ) }$ . For each retrieval mode $m \in$ {sem, lex, comb}, the target model is optimized for T iterations and evaluated as:

$$
A _ { m } \big ( \lambda _ { \mathrm { m i x } } \big ) = \mathrm { P e r f } \left( M ^ { ( T ) } \mid \mathcal { D } _ { \mathrm { m i x } } ^ { ( t ) } \big ( \lambda _ { \mathrm { m i x } } \big ) \right) ,\tag{27}
$$

where $\mathrm { P e r f } ( \cdot )$ denotes the task-specific evaluation metric.

Figure 5 reports $A _ { \mathrm { s e m } } ( \lambda _ { \mathrm { m i x } } ) , A _ { \mathrm { l e x } } ( \lambda _ { \mathrm { m i x } } )$ , and $A _ { \mathrm { { c o m b } } } ( \lambda _ { \operatorname* { m i x } } )$ as functions of the mixing ratio. All three curves improve substantially when moderate original-data mixing is introduced. The hybrid retrieval strategy achieves the strongest performance near $\begin{array} { r } { \lambda _ { \operatorname* { m i x } } = \frac { 1 } { 4 } } \end{array}$ , indicating that semantic and lexical retrieval are most efective when policy-selected adversarial failures are balanced with suficient original-distribution coverage. This setting provides the best trade-of between robustness improvement and forgetting prevention.

We therefore use $\textstyle \lambda _ { \mathrm { m i x } } ^ { * } = { \frac { 1 } { 4 } }$ in the main experiments. This controlled mixing mitigates catastrophic forgetting while preserving the benefit of failure-aware adversarial training.

![](images/5f17a1a42c7f2ae74561b5cb7d62b0f0ae1fde3811d49aa232232190d4a05fc1.jpg)  
Figure 5: Task performance $A _ { m } ( \lambda _ { \operatorname* { m i x } } )$ versus the mixing ratio of selected adversarial examples to original training data.

Theoretical interpretation. Our method changes the efective training distribution by mixing the original data distribution P with a policy-induced distribution over validated failure modes $\check { Q } _ { t } ^ { \pi }$

$$
P _ { t } ^ { \lambda } = ( 1 - \lambda ) P + \lambda \hat { Q } _ { t } ^ { \pi } .\tag{28}
$$

This view explains why failure-mode curation can reduce reliance on spurious shortcuts: selected failures are examples where the current model’s decision rule breaks, so training on them increases the relative contribution of task-relevant gradients. Under the assumptions stated in Appendix B, Lemma B.1 shows that failure-mode sampling reduces shortcut-aligned gradient contributions while preserving core-feature contributions. Propositions B.2 and B.2 further show that the mixture update induces bounded distributional drift and that bounded reward noise causes bounded distortion in the induced sampling policy.

## 4 Evaluation and Results

We evaluate the proposed failure-mode contextual bandit curation pipeline on standard benchmarks for natural language inference and fact verification. All experiments use automatically generated and automatically validated adversarial examples, without additional human annotation.

Target NLI Model. For NLI experiments, the target model is RoBERTa-base-SNLI (125M parameters) (HuggingFace, 2022), a RoBERTa-base model fine-tuned on SNLI. This model serves as the classifier whose failures are mined, clustered into failure modes, and used for adaptive adversarial retraining.

Generation LLM. Adversarial hypotheses are generated using LLaMA-4-Scout-17B-16E-Instruct (Meta AI, 2025). For each input, the generator is conditioned on a label-balanced retrieved context constructed using semantic retrieval, lexical retrieval, or hybrid BGE+BM25 retrieval.

Validation LLMs. Each generated candidate is automatically validated by an ensemble of three instructiontuned judge models: Gemma-3-27B-IT (Google Research, 2025), Phi-4 (Microsoft Research, 2025), and Qwen3-32B (Qwen Team, 2025). A candidate is retained only when all judges agree with the intended gold label. This validation stage is used to reduce label noise before failure-mode clustering and policy selection.

Bandit Policy and Critic. The data curator is implemented as a contextual-bandit policy $\pi _ { \theta }$ over validated failure modes. Each failure mode is represented by a cluster-level state vector containing statistics such as cluster size, target-model loss, entropy, classification margin, label distribution, retrieval score, judge agreement, novelty, and previous reward. The critic $R _ { \phi }$ is a lightweight MLP that predicts the expected validation reward of each selected failure mode. The policy and critic are updated after each retraining round using validation-based feedback.

Datasets. We report NLI results on SNLI (Bowman et al., 2015), the original human-annotated inference dataset; ANLI (Nie et al., 2019), which contains adversarially constructed examples collected through human-and-model interaction; and MultiNLI (Williams et al., 2018), a multi-genre corpus for evaluating cross-domain. To assess transfer beyond NLI, we also evaluate on the FEVER fact verification benchmark.

To contextualize the gains, we compare against GNLI (Hosseini et al., 2024), a synthetic NLI corpus of approximately 685K LLM-generated examples. Fine-tuning RoBERTa-base on GNLI alone reaches 89.42% on SNLI, 77.07% on ANLI, and 57.61% on MultiNLI. Our pipeline generates approximately 30K adversarial candidates per retrieval strategy, applies target-model failure filtering and automated LLM validation, and retains 6637 BGE-based and 5991 BM25-based candidates for failure-mode clustering and policy-guided sampling. With controlled adversarial mixing, our method improves RoBERTa-base from 88.48% to 92.60% on SNLI, from 75.04% to 80.95% on ANLI, and from 54.67% to 71.99% on MultiNLI, as shown in Table 1. The table also shows that unfiltered adversarial data improves performance, but automated validation and failure-aware selection provide additional gains, indicating that robustness benefits come from prioritizing useful failure modes rather than simply adding more synthetic data.

Table 1: Accuracy (%) on each test set under adversarial mixing. Method names list only BGE, BM25, and BGE+BM25, denoting their respective generation methods. “Reward-Guided” indicates unanimous LLM validation.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">RoBERTa Base</td><td rowspan="2">Additional Data</td><td colspan="9"></td></tr><tr><td>Paraphrasing</td><td>GNLI</td><td>Method</td><td>Guided No</td><td>r = 0 90.98%</td><td>r = 1 91.17%</td><td>r = 2 91.51%</td><td>r = 3 91.54%</td><td>r = 4 91.55%</td></tr><tr><td rowspan="8">SNLI</td><td rowspan="8"></td><td rowspan="8">89.42%</td><td rowspan="8"></td><td></td><td>BGE</td><td>Yes No</td><td>90.10%</td><td>91.21%</td><td>92.13%</td><td>92.12%</td><td>92.13%</td></tr><tr><td></td><td></td><td></td><td>90.03% 91.02%</td><td>91.14%</td><td></td><td>91.18%</td><td>91.19%</td></tr><tr><td></td><td>BM25 BM25</td><td></td><td>90.09%</td><td>91.20%</td><td>92.00%</td><td>92.11%</td><td>92.12%</td></tr><tr><td></td><td></td><td>Yes</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>BGE+BM25</td><td>No</td><td>90.11%</td><td>91.19%</td><td>91.35%</td><td>91.61%</td><td>91.68%</td></tr><tr><td></td><td>BGE+BM25</td><td>Yes</td><td>90.54%</td><td>90.78%</td><td>92.33%</td><td>92.41%</td><td>92.60%</td></tr><tr><td></td><td>T5-Small</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>T5-Large</td><td></td><td>-</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="8">Adversarial</td><td rowspan="8"></td><td rowspan="8">77.07%</td><td rowspan="8"></td><td></td><td></td><td></td><td>79.07%</td><td>79.72%</td><td>79.52%</td><td>79.92%</td><td>79.47%</td></tr><tr><td></td><td>BGE</td><td>No</td><td>78.72%</td><td>79.12%</td><td>80.02%</td><td>79.72%</td><td>80.27%</td></tr><tr><td></td><td>BGE</td><td>Yes</td><td>78.07%</td><td>78.52%</td><td>78.72%</td><td>78.82%</td><td>78.88%</td></tr><tr><td></td><td>BM25 BM25</td><td>No Yes</td><td></td><td>78.62%</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>77.97%</td><td></td><td>78.92%</td><td>79.07%</td><td>79.12%</td></tr><tr><td></td><td>BGE+BM25</td><td>No</td><td>78.11%</td><td>79.18%</td><td>78.51%</td><td>78.99%</td><td>78.91%</td></tr><tr><td></td><td>BGE+BM25 T5-Small</td><td>Yes</td><td>79.12%</td><td>80.43%</td><td>80.67%</td><td>80.89%</td><td>80.95%</td></tr><tr><td>33.00% 45.72%</td><td>T5-Large</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="7">MultiNLI</td><td rowspan="7"></td><td rowspan="7">57.61%</td><td rowspan="7">50.01%</td><td></td><td>BGE</td><td>- No</td><td>69.54%</td><td>69.32%</td><td>70.22%</td><td>69.72%</td><td>71.08%</td></tr><tr><td></td><td>BGE</td><td>Yes</td><td>69.15%</td><td>69.62%</td><td>70.22%</td><td>69.97%</td><td>71.15%</td></tr><tr><td></td><td>BM25</td><td>No</td><td>68.34%</td><td>68.72%</td><td>68.92%</td><td>69.22%</td><td></td><td>69.54%</td></tr><tr><td></td><td></td><td>Yes</td><td>68.57%</td><td>68.82%</td><td>69.12%</td><td>69.47%</td><td></td></tr><tr><td></td><td>BM25</td><td>No</td><td>68.05%</td><td>68.15%</td><td>68.81%</td><td>69.11%</td><td>69.74% 70.02%</td></tr><tr><td></td><td>BGE+BM25</td><td></td><td>69.21%</td><td>69.37%</td><td>70.59%</td><td>69.81%</td><td>71.99%</td></tr><tr><td></td><td>BGE+BM25</td><td>Yes</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td>82.18% 90.61%</td><td>T5-Small T5-Large</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>91.77%</td><td>T5-XXL</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

We further evaluate transfer beyond NLI on FEVER. As shown in Table 2, our method improves across model scales. RoBERTa-base reaches 76.58% FEVER score and 79.42% label accuracy, while RoBERTalarge achieves 79.86% FEVER score and 82.45% accuracy. Lightweight models such as SmolLM2-360M and Qwen3-0.6B also benefit, suggesting that failure-aware curation transfers beyond NLI.

Table 2: Comparison on the FEVER benchmark. We report FEVER score and label accuracy.
<table><tr><td>Method</td><td>Backbone</td><td>Dev FEVER</td><td>Dev Acc.</td><td>Test FEVER</td><td>Test Acc.</td></tr><tr><td>GEAR Zhou et al. (2019)</td><td>BERT-base</td><td>70.69</td><td>74.84</td><td>67.19</td><td>71.60</td></tr><tr><td>KGAT Liu et al. (2020)</td><td>RoBERTa-large</td><td>76.11</td><td>78.29</td><td>70.38</td><td>74.07</td></tr><tr><td>WgtSum Tymoshenko &amp; Moschitti (2021)</td><td>RoBERTa-large</td><td>79.02</td><td>81.30</td><td>73.44</td><td>77.18</td></tr><tr><td>BEVERS DeHaven &amp; Scott (2023)</td><td>DeBERTa-v2-XL</td><td></td><td></td><td>77.70</td><td>80.24</td></tr><tr><td>Ours</td><td>RoBERTa-base</td><td>81.24</td><td>83.17</td><td>76.58</td><td>79.42</td></tr><tr><td>Ours</td><td>DeBERTa-v3</td><td>82.91</td><td>84.76</td><td>78.12</td><td>81.03</td></tr><tr><td>Ours</td><td>RoBERTa-large</td><td>84.37</td><td>86.12</td><td>79.86</td><td>82.45</td></tr><tr><td>Ours</td><td>SmolLM2-360M</td><td>79.68</td><td>81.54</td><td>74.92</td><td>77.31</td></tr><tr><td>Ours</td><td>Qwen3-0.6B</td><td>80.97</td><td>82.83</td><td>75.89</td><td>78.64</td></tr></table>

Table 3 and Figure 6 show that increasing the retrieved few-shot context improves performance across SNLI, ANLI, and MultiNLI. In particular, validated BGE reaches 92.15% on SNLI, 80.26% on ANLI, and 71.15%

on MultiNLI in the 9-shot setting, while BM25 shows similar but slightly weaker trends. The 6-shot setting already matches or exceeds the strongest adversarial mixing results, highlighting the importance of retrieval quality and validated failure selection.

Table 3: Few-shot accuracy (%) of our generation methods on each set. Columns indicate the number of few-shot examples.
<table><tr><td>Dataset</td><td>Method</td><td>Filtered?</td><td>0-shot</td><td>3-shot</td><td>6-shot</td><td>9-shot</td></tr><tr><td rowspan="6">SNLI</td><td>BGE</td><td>No</td><td>87.51%</td><td>90.05%</td><td>91.55%</td><td>91.56%</td></tr><tr><td>BGE</td><td>Yes</td><td>88.18%</td><td>90.69%</td><td>92.13%</td><td>92.15%</td></tr><tr><td>BM25</td><td>No</td><td>87.51%</td><td>89.69%</td><td>91.19%</td><td>91.22%</td></tr><tr><td>BM25</td><td>Yes</td><td>88.18%</td><td>90.67%</td><td>92.12%</td><td>92.11%</td></tr><tr><td>BGE+BM25</td><td>No</td><td>87.51%</td><td>89.98%</td><td>91.68%</td><td>91.51%</td></tr><tr><td>BGE+BM25</td><td>Yes</td><td>88.18%</td><td>90.71%</td><td>92.60%</td><td>92.51%</td></tr><tr><td rowspan="6">Adversarial NLI</td><td>BGE</td><td>No</td><td>75.81%</td><td>77.72%</td><td>79.47%</td><td>79.47%</td></tr><tr><td>BGE</td><td>Yes</td><td>76.27%</td><td>78.76%</td><td>80.27%</td><td>80.26%</td></tr><tr><td>BM25</td><td>No</td><td>75.81%</td><td>77.37%</td><td>78.87%</td><td>78.88%</td></tr><tr><td>BM25</td><td>Yes</td><td>76.27%</td><td>77.60%</td><td>79.12%</td><td>79.10%</td></tr><tr><td>BGE+BM25</td><td>No</td><td>75.81%</td><td>77.71%</td><td>78.81%</td><td>78.91%</td></tr><tr><td>BGE+BM25</td><td>Yes</td><td>76.27%</td><td>77.81%</td><td>78.95%</td><td>80.95%</td></tr><tr><td rowspan="6">MultiNLI</td><td>BGE</td><td>No</td><td>67.18%</td><td>69.25%</td><td>71.07%</td><td>71.08%</td></tr><tr><td>BGE</td><td>Yes</td><td>67.87%</td><td>69.02%</td><td>71.12%</td><td>71.15%</td></tr><tr><td>BM25</td><td>No</td><td>67.18%</td><td>68.07%</td><td>69.57%</td><td>69.54%</td></tr><tr><td>BM25</td><td>Yes</td><td>67.87%</td><td>68.22%</td><td>69.72%</td><td>69.74%</td></tr><tr><td>BGE+BM25</td><td>No</td><td>67.18%</td><td>69.42%</td><td>69.99%</td><td>70.02%</td></tr><tr><td>BGE+BM25</td><td>Yes</td><td>67.87%</td><td>71.00%</td><td>71.12%</td><td>71.99%</td></tr></table>

![](images/8aa165d0819dcdf7841be868c04b46ac03a41b27c649c9392e296c0a254e2c66.jpg)  
Figure 6: Few-shot accuracy of generation methods by dataset.

## 5 Conclusion and Future Work

We presented a failure-mode contextual bandit framework for adversarial data curation. Instead of selecting synthetic examples with a fixed reward threshold, our method clusters validated model errors into recurring failure modes and learns which modes should be sampled for retraining. This turns the data curator into an adaptive policy that receives validation-based feedback and balances robustness gains, forgetting, and data cost. Across NLI benchmarks and FEVER, the framework improves robustness while using substantially less data than large untargeted synthetic corpora. The results show that prioritizing validated, model-specific failure modes is more efective than simply adding more generated examples. Future work will explore richer failure-mode representations, uncertainty-aware policy updates, adaptive generation budgets, and online curation settings where failures are generated and selected continuously. We also plan to extend the framework to multilingual, domain-specific, and broader robustness tasks, and to combine it with complementary methods such as contrastive learning, adversarial regularization, and representation-level alignment.

## References

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíček, Agustín Piqueres Lajarín, Vaibhav Srivastav, et al. Smollm2: When smol goes big–data-centric training of a small language model. arXiv preprint arXiv:2502.02737, 2025.

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, EMNLP 2015, pp. 632–642, Lisbon, Portugal, 2015. Association for Computational Linguistics. doi: 10.18653/v1/D15-1075. URL https://aclanthology.org/D15-1075. pdf.

Vicente Iván Sánchez Carmona, Jef Mitchell, and Sebastian Riedel. Behavior analysis of nli models: Uncovering the influence of three factors on robustness. arXiv preprint, abs/1805.04212, 2018. URL https://arxiv.org/abs/1805.04212.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multilingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint, abs/2402.03216, 2024. URL https://arxiv.org/abs/2402.03216.

Mitchell DeHaven and Stephen Scott. Bevers: A general, simple, and performant framework for automatic fact verification. In Proceedings of the Sixth Workshop on Fact Extraction and VERification (FEVER). Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023.fever-1.6/.

Yang Fan, Fei Tian, Tao Qin, Jiang Bian, and Tie-Yan Liu. Learning what data to learn. arXiv preprint, abs/1702.08635, 2017. URL https://arxiv.org/abs/1702.08635.

Max Glockner, Vered Shwartz, and Yoav Goldberg. Breaking nli systems with sentences that require simple lexical inferences. arXiv preprint, abs/1805.02266, 2018. URL https://arxiv.org/abs/1805.02266.

Google Research. Gemma-3-27b-it. Hugging Face model repository, 2025. https://huggingface.co/ google/gemma-3-27b-it.

Mohammad Javad Hosseini, Andrey Petrov, Alex Fabrikant, and Annie Louis. A synthetic data approach for domain generalization of NLI models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2212–2226, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.120. URL https://aclanthology.org/ 2024.acl-long.120/.

HuggingFace. pepa/roberta-base-snli, 2022. URL https://huggingface.co/pepa/roberta-base-snli. Accessed: October 12, 2024.

Mohit Iyyer, John Wieting, Kevin Gimpel, and Luke Zettlemoyer. Adversarial example generation with syntactically controlled paraphrase networks. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 1875–1885, New Orleans, Louisiana, USA, 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1170. URL https://aclanthology.org/N18-1170.pdf.

Mojan Javaheripi, Sébastien Bubeck, Marah Abdin, Jyoti Aneja, Sebastien Bubeck, Caio César Teodoro Mendes, Weizhu Chen, Allie Del Giorno, Ronen Eldan, Sivakanth Gopi, et al. Phi-2: The surprising power of small language models. Microsoft Research Blog, 1(3):3, 2023.

Animesh Jha, Harshit Gupta, and Ananjan Nandi. Rl-guided data selection for language model finetuning. In NeurIPS 2025 Workshop on Reliable Machine Learning from Unreliable Data, Vancouver, Canada, 2025. Neural Information Processing Systems Foundation. URL https://arxiv.org/abs/2509.25850.

Roie Kazoom, Raz Birman, and Ofer Hadar. Enhancing object detection robustness: Detecting and restoring confidence in the presence of adversarial patch attacks. arXiv preprint arXiv:2403.12988, 2024a.

Roie Kazoom, Raz Birman, and Ofer Hadar. Improving the robustness of object detection and classification ai models against adversarial patch attacks. arXiv e-prints, pp. arXiv–2403, 2024b.

Roie Kazoom, Raz Birman, and Ofer Hadar. From adversity to advantage: Difusion models for improved detection under attack. In International Symposium on Cyber Security, Cryptology, and Machine Learning, pp. 104–121. Springer, 2025a.

Roie Kazoom, Ofir Cohen, Rami Puzis, Asaf Shabtai, and Ofer Hadar. Vault: Vigilant adversarial updates via llm-driven retrieval-augmented generation for nli. arXiv preprint arXiv:2508.00965, 2025b.

Roie Kazoom, Raz Lapid, Moshe Sipper, and Ofer Hadar. Don’t Lag, RAG: Training-free adversarial detection using rag. arXiv preprint, abs/2504.04858, 2025c. URL https://arxiv.org/abs/2504.04858.

Roie Kazoom, Alon Goldberg, Hodaya Cohen, and Ofer Hadar. Seeing isn’t believing: Context-aware adversarial patch synthesis via conditional gan. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 202–211, 2026.

Matej Klemen and Marko Robnik-Šikonja. Extracting and filtering paraphrases by bridging natural language inference and paraphrasing. arXiv preprint, abs/2111.07119, 2021. URL https://arxiv.org/abs/2111. 07119.

Shipeng Li, Shikun Li, Zhiqin Yang, Xinghua Zhang, Gaode Chen, Xiaobo Xia, Hengyu Liu, and Zhe Peng. Learnalign: Reasoning data selection for reinforcement learning in large language models based on improved gradient alignment. arXiv preprint arXiv:2506.11480, 2025.

Yongqi Li, Mayi Xu, Xin Miao, Shen Zhou, and Tieyun Qian. Prompting large language models for counterfactual generation: An empirical study. arXiv preprint, abs/2305.14791, 2023. URL https: //arxiv.org/abs/2305.14791.

Zhenghao Liu, Chenyan Xiong, Maosong Sun, and Zhiyuan Liu. Fine-grained fact verification with kernel graph attention network. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2020. URL https://aclanthology.org/2020. acl-main.655/.

Meta AI. Llama-4-scout-17b-16e-instruct. Hugging Face model repository, 2025. https://huggingface. co/meta-llama/Llama-4-Scout-17B-16E-Instruct.

Microsoft Research. Phi-4. Hugging Face model repository, 2025. https://huggingface.co/microsoft/ phi-4.

Pasquale Minervini and Sebastian Riedel. Adversarially regularising neural NLI models to integrate logical background knowledge. In Proceedings of the 22nd Conference on Computational Natural Language Learning (CoNLL), pp. 65–74. Association for Computational Linguistics, 2018.

Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. Adversarial NLI: A new benchmark for natural language understanding. arXiv preprint arXiv:1910.14599, 2019.

Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. Adversarial NLI: A new benchmark for natural language understanding. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 4885–4901. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.acl-main.441.

Qwen Team. Qwen3-32b. Hugging Face model repository, 2025. https://huggingface.co/Qwen/ Qwen3-32B.

Stephen E. Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333–389, 2009. doi: 10.1561/1500000019.

Kateryna Tymoshenko and Alessandro Moschitti. Strong and light baseline models for fact-checking joint inference. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pp. 4824– 4830, Online, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-acl.426. URL https://aclanthology.org/2021.findings-acl.426/.

Adina Williams, Nikita Nangia, and Samuel Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 1112–1122, 2018.

An Yang et al. Qwen2.5: A suite of foundation models. arXiv preprint arXiv:2412.15115, 2412.15115, 2024. URL https://arxiv.org/abs/2412.15115.

Suorong Yang, Peijia Li, Furao Shen, and Jian Zhao. Rl-selector: Reinforcement learning-guided data selection via redundancy assessment. arXiv preprint arXiv:2506.21037, 2025.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In Proceedings of the 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, 2020. OpenReview.net. URL https://openreview.net/forum?id= SkeHuCVFDr.

Mingjun Zhao, Haijiang Wu, Di Niu, and Xiaoli Wang. Reinforced curriculum learning on pre-trained neural machine translation models. arXiv preprint, abs/2004.05757, 2020. URL https://arxiv.org/abs/2004. 05757.

Jie Zhou, Xu Han, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, and Maosong Sun. Gear: Graphbased evidence aggregating and reasoning for fact verification. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2019. URL https://aclanthology.org/P19-1085/.

## A Appendix

## B Theoretical Interpretation and Proofs

## B.1 Bias Reduction via Failure-Mode Curation

Let X be the input space, $Y \in \mathcal { V }$ the label space, and P the underlying data distribution. Let $f _ { \theta _ { t } } : \mathcal { X }  \Delta ( \mathcal { Y } )$ denote the target model at iteration $t ,$ and let $\ell ( \theta ; x , y )$ be the per-example loss. We define the failure indicator as:

$$
e _ { t } ( x , y ) = \mathbb { I } \left[ \underset { c \in \mathcal { V } } { \arg \operatorname* { m a x } } f _ { \theta _ { t } } ( x ) _ { c } \neq y \right] .\tag{29}
$$

The corresponding failure probability is:

$$
\varepsilon _ { t } = \mathbb { E } _ { ( x , y ) \sim P } \left[ e _ { t } ( x , y ) \right] .\tag{30}
$$

When $\varepsilon _ { t } > 0$ , the failure-conditioned distribution is:

$$
F _ { t } ( x , y ) = P ( x , y \mid e _ { t } ( x , y ) = 1 ) = \frac { P ( x , y ) e _ { t } ( x , y ) } { \varepsilon _ { t } } .\tag{31}
$$

In practice, the method does not sample directly from $F _ { t }$ . Instead, it generates candidate adversarial examples, filters candidates that fool $M ^ { ( t ) }$ , validates them with automated judges, clusters the validated failures into failure modes, and samples from these modes using the contextual-bandit policy $\pi _ { \theta }$ . Let $\widehat { F } _ { t } ^ { \pi }$ denote the policy-induced distribution over selected validated failure examples. The efective training distribution is:

$$
P _ { t } ^ { \lambda } = ( 1 - \lambda ) P + \lambda \widehat { F } _ { t } ^ { \pi } , \qquad \lambda \in ( 0 , 1 ) .\tag{32}
$$

The target model is then updated by stochastic gradient descent on:

$$
\mathbb { E } _ { ( x , y ) \sim P _ { t } ^ { \lambda } } \left[ \ell ( \theta ; x , y ) \right] .\tag{33}
$$

Spurious bias model. We use an abstract decomposition $\boldsymbol { x } = \left( c , s \right)$ , where c denotes task-relevant core information and s denotes a spurious feature that is correlated with the label under $P$ but unreliable under distribution shift. This decomposition is not assumed to be explicitly available to the model; it is used only for analysis. Let $g _ { c } ( x , y )$ and $g _ { s } ( x , y )$ denote the gradient components along unit directions $u _ { c }$ and $u _ { s } \colon$

$$
\begin{array} { r } { g _ { c } ( x , y ) = \left. \nabla _ { \theta } \ell ( \theta ; x , y ) , u _ { c } \right. , \qquad g _ { s } ( x , y ) = \left. \nabla _ { \theta } \ell ( \theta ; x , y ) , u _ { s } \right. . } \end{array}\tag{34}
$$

We assume that standard training on $P$ may reinforce the spurious direction, while failure regions reduce reliance on this shortcut. Specifically, assume there exist constants $\mu _ { s } > 0$ and $\mu _ { c } \geq 0$ such that:

$$
\begin{array} { r } { \mathbb { E } _ { P } \left[ g _ { s } ( x , y ) \right] \ge \mu _ { s } , } \end{array}\tag{35}
$$

and:

$$
\mathbb { E } _ { F _ { t } } \left[ g _ { s } ( x , y ) \right] \le 0 , \qquad \mathbb { E } _ { F _ { t } } \left[ g _ { c } ( x , y ) \right] \ge \mu _ { c } .\tag{36}
$$

This captures the intended failure-focused behavior: validated failures are examples where shortcut-based prediction is less reliable and core task-relevant evidence is more important.

Policy approximation. The contextual-bandit policy does not need to recover $F _ { t }$ exactly. It is suficient that the policy-induced selected distribution $\widehat { F } _ { t } ^ { \pi }$ approximates $F _ { t }$ with bounded error along the relevant gradient directions:

$$
\left. \mathbb { E } _ { \widehat { F } _ { t } ^ { \pi } } \left[ g _ { s } \right] - \mathbb { E } _ { F _ { t } } \left[ g _ { s } \right] \right. \leq \delta _ { s } , \qquad \left. \mathbb { E } _ { \widehat { F } _ { t } ^ { \pi } } \left[ g _ { c } \right] - \mathbb { E } _ { F _ { t } } \left[ g _ { c } \right] \right. \leq \delta _ { c } ,\tag{37}
$$

for $\delta _ { s } , \delta _ { c } \geq 0$

Lemma A.1. Failure-mode sampling reduces shortcut-aligned gradients. Under the above assumptions, the mixture distribution $P _ { t } ^ { \lambda }$ satisfies:

$$
\mathbb { E } _ { P _ { t } ^ { \lambda } } \left[ g _ { s } ( x , y ) \right] \le ( 1 - \lambda ) \mathbb { E } _ { P } \left[ g _ { s } \right] + \lambda \delta _ { s } ,\tag{38}
$$

and:

$$
\begin{array} { r } { \mathbb { E } _ { P _ { t } ^ { \lambda } } \left[ g _ { c } ( x , y ) \right] \ge ( 1 - \lambda ) \mathbb { E } _ { P } \left[ g _ { c } \right] + \lambda ( \mu _ { c } - \delta _ { c } ) . } \end{array}\tag{39}
$$

Consequently, if $\delta _ { s } < \mathbb { E } _ { P } [ g _ { s } ]$ , then failure-mode sampling reduces the shortcut-aligned gradient contribution relative to training only on $P$ by at least:

$$
\lambda \left( \mathbb { E } _ { P } [ g _ { s } ] - \delta _ { s } \right) .\tag{40}
$$

Proof. $\mathrm { B y }$ linearity of expectation under the mixture distribution:

$$
\mathbb { E } _ { P _ { t } ^ { \lambda } } \left[ g _ { s } \right] = ( 1 - \lambda ) \mathbb { E } _ { P } \left[ g _ { s } \right] + \lambda \mathbb { E } _ { \widehat { F } _ { t } ^ { \pi } } \left[ g _ { s } \right] .\tag{41}
$$

Using the policy approximation bound and the failure-region assumption:

$$
\mathbb { E } _ { \widehat { F } _ { t } ^ { \pi } } \left[ g _ { s } \right] \leq \mathbb { E } _ { F _ { t } } \left[ g _ { s } \right] + \delta _ { s } \leq \delta _ { s } .\tag{42}
$$

Substituting this into the mixture expression gives:

$$
\mathbb { E } _ { P _ { t } ^ { \lambda } } \left[ g _ { s } \right] \le ( 1 - \lambda ) \mathbb { E } _ { P } \left[ g _ { s } \right] + \lambda \delta _ { s } .\tag{43}
$$

The result for the core component follows similarly. From the approximation assumption:

$$
\mathbb { E } _ { \widehat { F } _ { t } ^ { \pi } } \left[ g _ { c } \right] \geq \mathbb { E } _ { F _ { t } } \left[ g _ { c } \right] - \delta _ { c } \geq \mu _ { c } - \delta _ { c } .\tag{44}
$$

Therefore:

$$
\mathbb { E } _ { P _ { t } ^ { \lambda } } \left[ g _ { c } \right] \ge ( 1 - \lambda ) \mathbb { E } _ { P } \left[ g _ { c } \right] + \lambda ( \mu _ { c } - \delta _ { c } ) .\tag{45}
$$

Finally, comparing the shortcut bound to $\mathbb { E } _ { P } [ g _ { s } ]$ gives:

$$
\begin{array} { r } { \mathbb { E } _ { P } [ g _ { s } ] - \mathbb { E } _ { P _ { t } ^ { \lambda } } [ g _ { s } ] \ge \lambda \left( \mathbb { E } _ { P } [ g _ { s } ] - \delta _ { s } \right) , } \end{array}\tag{46}
$$

which is positive whenever $\delta _ { s } < \mathbb { E } _ { P } [ g _ { s } ]$ . This completes the proof.

## B.2 Boundedness of Failure-Aware Bandit Updates

The contextual-bandit policy changes the efective training distribution by selecting diferent failure modes across iterations. To avoid uncontrolled distributional drift, the target model is trained on a mixture of original and selected adversarial examples. Let $\mathcal { P } _ { t }$ denote the efective training distribution at iteration t, and let $\widehat { F } _ { t } ^ { \pi }$ denote the distribution induced by the failure-mode policy. We analyze the abstract update:

$$
\mathcal { P } _ { t + 1 } = ( 1 - \eta ) \mathcal { P } _ { t } + \eta \widehat { F } _ { t } ^ { \pi } , \qquad \eta \in ( 0 , 1 ) .\tag{47}
$$

Proposition A.2. Per-step distributional drift is bounded. For any iteration $t ,$ the per-step change in the efective training distribution satisfies:

$$
\left. \mathcal { P } _ { t + 1 } - \mathcal { P } _ { t } \right. _ { 1 } = \eta \left. \widehat { F } _ { t } ^ { \pi } - \mathcal { P } _ { t } \right. _ { 1 } \leq 2 \eta .\tag{48}
$$

Moreover, for any t, the distance from the initial distribution is bounded by:

$$
\left\| \mathcal { P } _ { t } - \mathcal { P } _ { 0 } \right\| _ { 1 } \leq 2 .\tag{49}
$$

Proof. From the update rule:

$$
\mathcal { P } _ { t + 1 } - \mathcal { P } _ { t } = \eta \left( \widehat { F } _ { t } ^ { \pi } - \mathcal { P } _ { t } \right) .\tag{50}
$$

Taking the $\ell _ { 1 }$ norm gives:

$$
\left\| \mathcal { P } _ { t + 1 } - \mathcal { P } _ { t } \right\| _ { 1 } = \eta \left\| \widehat { F } _ { t } ^ { \pi } - \mathcal { P } _ { t } \right\| _ { 1 } .\tag{51}
$$

Because the $\ell _ { 1 }$ distance between any two probability distributions is at most 2:

$$
\| \mathcal { P } _ { t + 1 } - \mathcal { P } _ { t } \| _ { 1 } \leq 2 \eta .\tag{52}
$$

The second statement follows from the same fact, since both $\mathcal { P } _ { t }$ and $\mathcal { P } _ { 0 }$ are probability distributions:

$$
\left\| \mathcal { P } _ { t } - \mathcal { P } _ { 0 } \right\| _ { 1 } \leq 2 .\tag{53}
$$

This establishes that each update is locally controlled by the mixing coeficient $\eta .$

Reward-noise setting. The policy is updated using validation feedback, which may be noisy due to finite validation sets and stochastic retraining. Let $u _ { k }$ denote the ideal utility of failure mode $k ,$ and let the observed utility be:

$$
\tilde { u } _ { k } = u _ { k } + \xi _ { k } , \qquad | \xi _ { k } | \leq \varepsilon .\tag{54}
$$

For analysis, consider the normalized policy-induced allocation distribution over failure modes:

$$
\pi _ { u } ( k ) = \frac { \exp ( u _ { k } ) } { \sum _ { j } \exp ( u _ { j } ) } , \qquad \pi _ { \tilde { u } } ( k ) = \frac { \exp ( \tilde { u } _ { k } ) } { \sum _ { j } \exp ( \tilde { u } _ { j } ) } .\tag{55}
$$

This log-linear allocation is a standard smooth relaxation of selecting failure modes according to estimated utility.

Proposition A.3. Bounded reward noise induces bounded sampling distortion. Assume $| \xi _ { k } | \le \varepsilon$ for all failure modes k. Then, for any set of failure modes A with $\pi _ { u } ( A ) > 0$

$$
e ^ { - 2 \varepsilon } \leq \frac { \pi _ { \tilde { u } } ( A ) } { \pi _ { u } ( A ) } \leq e ^ { 2 \varepsilon } .\tag{56}
$$

Proof. For each failure mode k:

$$
\begin{array} { r } { e ^ { - \varepsilon } \exp ( u _ { k } ) \leq \exp ( \tilde { u } _ { k } ) \leq e ^ { \varepsilon } \exp ( u _ { k } ) . } \end{array}\tag{57}
$$

Summing over all modes gives:

$$
e ^ { - \varepsilon } \sum _ { j } \exp ( u _ { j } ) \leq \sum _ { j } \exp ( { \tilde { u } } _ { j } ) \leq e ^ { \varepsilon } \sum _ { j } \exp ( u _ { j } ) .\tag{58}
$$

Combining the numerator and denominator bounds yields, for each k:

$$
e ^ { - 2 \varepsilon } \leq \frac { \pi _ { \tilde { u } } ( k ) } { \pi _ { u } ( k ) } \leq e ^ { 2 \varepsilon } .\tag{59}
$$

Summing over all $k \in A$ preserves the same multiplicative bound:

$$
e ^ { - 2 \varepsilon } \leq \frac { \sum _ { k \in A } \pi _ { \tilde { u } } ( k ) } { \sum _ { k \in A } \pi _ { u } ( k ) } \leq e ^ { 2 \varepsilon } .\tag{60}
$$

Therefore:

$$
e ^ { - 2 \varepsilon } \leq \frac { \pi _ { \tilde { u } } ( A ) } { \pi _ { u } ( A ) } \leq e ^ { 2 \varepsilon } .\tag{61}
$$

Together, Proposition B.2 and Proposition B.2 show that the failure-aware bandit update operates in a controlled regime: the mixture coeficient bounds the per-step distributional shift, and bounded noise in validation-based utility estimates induces only bounded distortion in the policy-induced failure-mode allocation.

## B.3 Backbone and Training Strategy Ablation

We analyze the impact of backbone architecture and training strategy using the results reported in Table 4. This experiment evaluates five representative models spanning a wide range of parameter scales, from lightweight decoder-only architectures (SmolLM2-360M and Qwen3-0.6B) to large encoder-based models (RoBERTa-base, DeBERTa-v3, and RoBERTa-large). For each backbone, we compare three settings: (i) evaluation without fine-tuning (No FT), (ii) fine-tuning with paraphrase-based data augmentation (Paraphrasing), and (iii) our reinforcement-guided failure-driven training framework (Ours). All models are trained under identical conditions using fixed 6-shot prompting and a constant original-to-generated ratio of 1:4 $( r = 4 )$

Several consistent trends emerge across all datasets. First, models evaluated without fine-tuning exhibit the lowest performance in every configuration, confirming that direct transfer without adaptation is insuficient for robust NLI. Paraphrase-based augmentation yields moderate improvements over No FT, indicating that generic linguistic variation helps alleviate some distributional mismatch. However, these gains remain limited, particularly on challenging benchmarks such as Adversarial NLI, where paraphrasing fails to systematically target the model’s dominant failure modes.

In contrast, our reinforcement-guided approach consistently achieves the strongest performance for all backbone architectures and datasets. On SNLI, our method improves RoBERTa-base from 90.72% under paraphrasing to 92.60%, and yields comparable gains for smaller models such as SmolLM2-360M (+1.81 points) and Qwen3-0.6B (+1.91 points). Similar patterns are observed on Adversarial NLI, where our framework outperforms paraphrasing by 2.83 points for RoBERTa-base and by more than 2 points for all other models. On MultiNLI, which exhibits substantial genre diversity, reinforcement-guided training produces consistent improvements ranging from 2.36 to 3.47 points over paraphrasing.

The results further demonstrate that the benefits of failure-driven policy learning are preserved across model scales. While larger backbones such as RoBERTa-large achieve higher absolute accuracy, smaller and medium-sized models also benefit substantially from targeted adversarial mining. This indicates that the proposed framework does not rely on excess model capacity, but instead improves generalization by reshaping the training distribution toward informative failure regions.

Importantly, the consistent gap between paraphrasing and our method highlights the limitations of heuristic data augmentation. Paraphrase-based approaches introduce surface-level variation but do not adaptively concentrate on systematic errors. In contrast, our method leverages reinforcement learning to prioritize samples that expose decision boundary weaknesses, resulting in more eficient and targeted learning.

Overall, the results in Table 4 demonstrate that reinforcement-guided adversarial training yields robust and scalable improvements across diverse architectures and training regimes, confirming the generality of the proposed approach.

Efect of Generator and Verifier Scale. Our framework relies on large language models for adversarial generation and validation, and its performance may depend on their representational capacity. To analyze this dependence, future work will systematically vary the scale of both the generator and verifier models, ranging from lightweight open-source LLMs to large proprietary systems. Such controlled experiments will enable a principled assessment of how robustness gains trade of against computational cost, and will clarify the operating regimes in which reinforcement-guided data selection remains efective under limited budgets.

Disentangling Generation and Verification Contributions. An important open question concerns the relative contribution of the generation and verification components. While both modules are optimized through policy feedback, their individual roles in driving performance gains are not yet fully disentangled. A promising direction is to decouple these stages by fixing one component while varying the capacity of the other, thereby isolating the efect of generation quality versus validation reliability. This analysis would help determine whether improvements primarily stem from producing more challenging candidates or from more accurate reward estimation via verification.

Table 4: Accuracy (%) across diferent backbone models and training strategies. “No FT” denotes evaluation without fine-tuning, “Paraphrasing” denotes augmentation via paraphrase-based data, and “Ours” denotes reinforcement-guided training. All experiments use fixed 6-shot prompting and an original-to-generated data ratio of 1:4 (r = 4).
<table><tr><td rowspan="2">Dataset</td><td colspan="3">SmolLM2-360M</td><td colspan="3">Qwen3-0.6B</td><td colspan="3">RoBERTa-base</td><td colspan="3">DeBERTa-v3</td><td colspan="3">RoBERTa-large</td></tr><tr><td>No FT</td><td>Para</td><td>Ours</td><td>No FT</td><td>Para</td><td>Ours</td><td>No FT</td><td>Para</td><td>Ours</td><td>No FT</td><td>Para</td><td>Ours</td><td>No FT</td><td>Para</td><td>Ours</td></tr><tr><td>SNLI</td><td>79.42</td><td>82.36</td><td>84.17</td><td>82.95</td><td>85.41</td><td>87.32</td><td>88.48</td><td>90.72</td><td>92.60</td><td>87.91</td><td>89.84</td><td>91.94</td><td>89.37</td><td>91.12</td><td>93.21</td></tr><tr><td>Adversarial NLI</td><td>67.18</td><td>70.04</td><td>72.48</td><td>70.93</td><td>73.41</td><td>75.91</td><td>75.04</td><td>78.12</td><td>80.95</td><td>74.26</td><td>77.03</td><td>79.87</td><td>75.88</td><td>78.94</td><td>81.62</td></tr><tr><td>MultiNLI</td><td>61.94</td><td>64.37</td><td>66.73</td><td>64.82</td><td>67.05</td><td>69.48</td><td>54.67</td><td>68.42</td><td>71.99</td><td>66.71</td><td>69.12</td><td>71.42</td><td>68.09</td><td>70.33</td><td>72.86</td></tr></table>

## B.4 Ablation Study: Efect of the Contextual-Bandit Policy

We analyze the contribution of the contextual-bandit curator on SNLI by replacing the learned failure-mode policy with several alternatives. All variants use the same backbone, retrieval strategy, automated validation, adversarial budget, and retraining protocol. The full model achieves 92.60% accuracy on SNLI.

Let $\{ \mathcal { F } _ { 1 } ^ { ( t ) } , \ldots , \mathcal { F } _ { K _ { t } } ^ { ( t ) } \}$ denote the validated failure-mode clusters at iteration t. Each cluster is represented by a state vector $z _ { t , k } ,$ , and the curator selects actions $a _ { t , k } \in \{ 0 , 1 \}$ indicating whether failure mode $\mathcal { F } _ { k } ^ { ( t ) }$ is sampled for retraining.

Full Model: Contextual-Bandit Policy. The proposed model uses a stochastic policy $\pi _ { \theta }$ over failure modes:

$$
\pi _ { \boldsymbol { \theta } } ( a _ { t , k } = 1 \mid z _ { t , k } ) = \sigma ( f _ { \boldsymbol { \theta } } ( z _ { t , k } ) ) .\tag{62}
$$

After retraining, the policy receives validation reward $G _ { t }$ , which balances robustness gain, forgetting, and data cost. A critic $R _ { \phi }$ estimates the expected return of each selected failure mode:

$$
R _ { \phi } ( z _ { t , k } ) \approx \mathbb { E } [ G _ { t } \mid z _ { t , k } ] .\tag{63}
$$

Random Failure-Mode Policy. This baseline removes learned selection. Failure modes are sampled uniformly under the same adversarial budget:

$$
a _ { t , k } \sim \mathrm { B e r n o u l l i } ( p ) .\tag{64}
$$

Heuristic Failure-Mode Policy. This variant replaces $\pi _ { \theta }$ with a deterministic uncertainty-based rule. Failure modes are ranked by the mean predictive entropy of the target model:

$$
s ( \mathcal { F } _ { k } ^ { ( t ) } ) = \frac { 1 } { \vert \mathcal { F } _ { k } ^ { ( t ) } \vert } \sum _ { ( \boldsymbol { x } , \boldsymbol { o } , \boldsymbol { y } ) \in \mathcal { F } _ { k } ^ { ( t ) } } H \left( M ^ { ( t ) } ( \cdot \mid \boldsymbol { x } , \boldsymbol { o } ) \right) .\tag{65}
$$

The highest-scoring clusters are selected until the adversarial budget is reached.

Frozen Policy and Critic. Here, π<sub>θ</sub> and $R _ { \phi }$ are initialized in the first round and then kept fixed. This tests whether continual validation-based adaptation is necessary.

No Failure-Mode Clustering. This baseline removes the failure-mode abstraction and samples validated failures directly. It preserves target-model filtering and automated validation but does not group failures into recurring modes.

Oracle Failure-Mode Policy. As an upper bound, we approximate the utility of a failure mode using its observed validation improvement after retraining:

$$
s _ { \mathrm { o r a c l e } } ( \mathcal { F } _ { k } ^ { ( t ) } ) = \mathrm { P e r f } ( M _ { k } ^ { ( t + 1 ) } ) - \mathrm { P e r f } ( M ^ { ( t ) } ) ,\tag{66}
$$

where $M _ { k } ^ { ( t + 1 ) }$ denotes a model retrained using samples from failure mode $\mathcal { F } _ { k } ^ { ( t ) }$ . This variant is not deployable because it requires separate retraining for each candidate failure mode.

Table 5 reports the results. Random and heuristic policies underperform the full model, showing that static selection is insuficient. Freezing the policy and critic also degrades performance, indicating that adaptation across rounds is important. Removing failure-mode clustering further reduces accuracy, confirming that selecting recurring failure types is more efective than selecting isolated examples.

## B.5 Ablation with Heuristic Failure-Mode Policies

To assess whether learned policy optimization is necessary, we replace the contextual-bandit policy $\pi _ { \theta }$ with several heuristic failure-mode selection rules. These baselines use the same generated candidates, target-

Table 5: Ablation of the contextual-bandit curator on SNLI.
<table><tr><td>Method</td><td>Failure Modes</td><td>Adaptive Policy</td><td>Critic  $R _ { \phi }$ </td><td>Accuracy (%)</td></tr><tr><td>Full (Ours)</td><td></td><td>√</td><td>√</td><td>92.60</td></tr><tr><td>Random Failure-Mode Policy</td><td>√</td><td>×</td><td>×</td><td>89.70</td></tr><tr><td>Heuristic Entropy Policy</td><td>V</td><td>X</td><td>X</td><td>90.45</td></tr><tr><td>Frozen Policy and Critic</td><td>V</td><td>X</td><td>V</td><td>91.20</td></tr><tr><td>No Failure-Mode Clustering</td><td>X</td><td>√</td><td>V</td><td>90.85</td></tr><tr><td>Oracle Failure-Mode Policy</td><td>V</td><td>GT</td><td>GT</td><td>93.40</td></tr></table>

model failure filtering, automated LLM validation, retrieval weight $\alpha = 0 . 8 3$ , and mixing ratio $\begin{array} { r } { \lambda _ { \operatorname* { m i x } } = \frac { 1 } { 4 } } \end{array}$ as the full method. The only diference is how validated failure modes are selected for retraining.

Let $\mathcal { F } _ { k } ^ { ( t ) }$ denote a validated failure-mode cluster at iteration t. Each heuristic assigns a cluster-level score $s ( \mathcal { F } _ { k } ^ { ( t ) } )$ , and clusters are selected in descending order until the adversarial budget is reached.

Confidence-Based Policy. This policy prioritizes clusters where the target model has low predictive confidence:

$$
s _ { \mathrm { c o n f } } ( \mathcal { F } _ { k } ^ { ( t ) } ) = \frac { 1 } { \vert \mathcal { F } _ { k } ^ { ( t ) } \vert } \sum _ { ( x , o , y ) \in \mathcal { F } _ { k } ^ { ( t ) } } \left( 1 - \operatorname* { m a x } _ { c \in \mathcal { Y } } M ^ { ( t ) } ( c \mid x , o ) \right) .\tag{67}
$$

Loss-Based Policy. This policy selects clusters that induce high average supervised loss:

$$
s _ { \mathrm { l o s s } } ( \mathcal { F } _ { k } ^ { ( t ) } ) = \frac { 1 } { | \mathcal { F } _ { k } ^ { ( t ) } | } \sum _ { ( \boldsymbol { x } , \boldsymbol { o } , \boldsymbol { y } ) \in \mathcal { F } _ { k } ^ { ( t ) } } \ell ( \boldsymbol { M } ^ { ( t ) } ( \boldsymbol { x } , \boldsymbol { o } ) , \boldsymbol { y } ) .\tag{68}
$$

Margin-Based Policy. This policy prioritizes clusters with small separation between the top two predicted classes:

$$
s _ { \mathrm { m a r g i n } } ( \mathcal { F } _ { k } ^ { ( t ) } ) = \frac { 1 } { \vert \mathcal { F } _ { k } ^ { ( t ) } \vert } \sum _ { ( x , o , y ) \in \mathcal { F } _ { k } ^ { ( t ) } } \left( 1 - \left[ p _ { \theta } ^ { ( 1 ) } ( x , o ) - p _ { \theta } ^ { ( 2 ) } ( x , o ) \right] \right) ,\tag{69}
$$

where $p _ { \theta } ^ { ( 1 ) } ( x , o )$ and $p _ { \theta } ^ { ( 2 ) } ( x , o )$ are the highest and second-highest predicted class probabilities.

Learned Bandit Policy. The full method uses the learned contextual-bandit policy:

$$
\pi _ { \boldsymbol { \theta } } ( a _ { t , k } = 1 \mid z _ { t , k } ) = \sigma ( f _ { \boldsymbol { \theta } } ( z _ { t , k } ) ) ,\tag{70}
$$

where $z _ { t , k }$ includes loss, entropy, margin, label distribution, retrieval score, judge agreement, novelty, cluster size, and previous reward statistics. Unlike the heuristic policies, π<sub>θ</sub> is updated using validation reward $G _ { t }$ after retraining.

Table 6: Comparison of heuristic failure-mode policies and the learned contextual-bandit policy.
<table><tr><td>Selection Policy</td><td>SNLI</td><td>ANLI</td><td>MultiNLI</td></tr><tr><td>Confidence-Based Policy</td><td>90.84</td><td>78.12</td><td>69.21</td></tr><tr><td>Loss-Based Policy</td><td>91.02</td><td>78.45</td><td>69.68</td></tr><tr><td>Margin-Based Policy</td><td>90.67</td><td>77.94</td><td>68.97</td></tr><tr><td>Learned Bandit Policy πθ</td><td>92.60</td><td>80.95</td><td>71.99</td></tr></table>

Heuristic policies capture only instantaneous model uncertainty or training dificulty. As a result, they may oversample noisy, redundant, or locally dificult failures that do not produce sustained validation gains. In contrast, the learned contextual-bandit policy is optimized using downstream validation feedback and can adapt across curation rounds. The consistent improvement over heuristic policies shows that adaptive failure-mode selection is more efective than static uncertainty-based selection.

## B.6 Component Analysis of Failure-Mode Curation

Table 7 evaluates the contribution of the main components in the proposed failure-mode contextual bandit curation framework. The full method achieves the best performance across all benchmarks, reaching 92.60% on SNLI, 80.95% on ANLI, and 71.99% on MultiNLI. This confirms that combining retrieval-augmented generation, target-model failure filtering, automated validation, failure-mode clustering, contextual-bandit selection, and controlled original-data mixing provides the strongest robustness gains.

Removing the contextual-bandit policy substantially reduces performance. Random cluster selection performs considerably worse, especially on MultiNLI, indicating that not all failure modes are equally useful for retraining. Selecting clusters by top loss improves over random selection, but remains below the full method, showing that simple dificulty-based heuristics are less efective than validation-driven policy learning. Similarly, replacing failure-mode selection with per-example selection also degrades performance, suggesting that grouping failures into recurring modes provides a more stable and useful unit for adversarial data curation.

The ablations further show that automated judge validation and target-model failure filtering are important for maintaining data quality. Without judge validation, performance drops across all datasets, indicating that noisy or incorrectly labeled generated examples can weaken the retraining signal. Removing failure filtering causes an even larger degradation, showing that explicitly focusing on examples that expose current model errors is central to the proposed approach.

Finally, the retrieval and mixing ablations demonstrate the importance of both informative generation context and forgetting control. Removing retrieved context reduces performance, confirming that retrieved few-shot examples help guide the generator toward more useful adversarial candidates. Training without original-data mixing also hurts performance, supporting the need to balance selected adversarial failures with original training examples in order to improve robustness while limiting forgetting.

Table 7: Ablation study of the proposed failure-mode contextual bandit curation framework.
<table><tr><td>Method</td><td>SNLI</td><td>ANLI</td><td>MultiNLI</td></tr><tr><td>Full method</td><td>92.60</td><td>80.95</td><td>71.99</td></tr><tr><td>No bandit, random clusters</td><td>89.70</td><td>76.50</td><td>66.20</td></tr><tr><td>No bandit, top-loss clusters</td><td>91.02</td><td>78.45</td><td>69.68</td></tr><tr><td>No clustering, per-example selection</td><td>90.85</td><td>78.20</td><td>69.10</td></tr><tr><td>No judge validation</td><td>91.68</td><td>78.91</td><td>70.02</td></tr><tr><td>No failure filtering</td><td>89.15</td><td>76.90</td><td>66.50</td></tr><tr><td>No retrieved context (0-shot)</td><td>88.18</td><td>76.27</td><td>67.87</td></tr><tr><td>No original-data mixing</td><td>90.54</td><td>79.12</td><td>69.21</td></tr></table>

## B.7 Judge Ensemble Configuration

With the retrieval weight fixed at $\alpha = 0 . 8 3$ and the generated-to-original example ratio set to 1:4, we evaluated the impact of varying the number of “judges” (independent LLM validators) on downstream accuracy. All experiments were run on the SNLI test set. We filtered examples by requiring unanimous agreement among the selected judges and then measured classification accuracy on the remaining items.

Table 8: Filtering and accuracy under diferent judge ensemble sizes (SNLI test, 1:4 gen:orig, $\alpha = 0 . 8 3 )$ Judges: G = Gemma-3-27B-IT (Google Research, 2025), Q = Qwen3-32B (Qwen Team, 2025), ${ \mathrm { P } } = { \mathrm { P h i } } $ 4 (Microsoft Research, 2025).
<table><tr><td></td><td># Judges # Examples</td><td>Accuracy (%)</td><td>Judges</td></tr><tr><td>1</td><td>16,147</td><td>91.02</td><td>G</td></tr><tr><td>2</td><td>9,312</td><td>91.49</td><td> $\mathrm { ~ G ~ } + \mathrm { ~ Q ~ }$ </td></tr><tr><td>3</td><td>6,438</td><td>92.13</td><td> ${ \mathrm { ~ G ~ } + \mathrm { ~ Q ~ } + \mathrm { ~ P ~ } }$ </td></tr></table>

As shown in Table 8 and Figure 7, the three-judge ensemble yields the highest accuracy (92.13%) on 6,438 filtered observations. Both the two-judge and single-judge configurations retain more examples but achieve lower accuracies of 91.49% (9,312 examples) and 91.02% (16,147 examples), respectively. Gemma-3-27B-IT consistently remains in all configurations, with Qwen3-32B joining for the two-judge setup and Phi-4 for the three-judge ensemble. We adopt the three-judge configuration for all subsequent evaluations.

![](images/99437fc049871f6dc8c3ac85ad6bfed8ea95c94d7a96df200b14e98f4e295ed5.jpg)  
Figure 7: Accuracy vs. number of judges (SNLI test, $\alpha = 0 . 8 3$ , 1:4 generated:original). Points are annotated with the number of filtered examples.

## B.7.1 Evaluation with Small Judge Models

To study the robustness of our validation pipeline under weaker supervision, we additionally evaluated the judge ensemble using lightweight language models, including Phi-2 Javaheripi et al. (2023), Qwen2.5- 1.5B Yang et al. (2024), and SmolLM2-360M Allal et al. (2025). All experiments were conducted using the same retrieval weight (α = 0.83) and generated-to-original ratio (1:4) as in Table 6.

We followed the same filtering protocol, retaining only examples for which all selected judges unanimously agreed. Classification accuracy was then measured on the remaining test instances. Table 9 reports the results on the SNLI test set.

Table 9: Filtering and accuracy under diferent small judge ensemble sizes (SNLI test, 1:4 gen:orig, $\alpha = 0 . 8 3 )$ Judges: S = SmolLM2-360M, P = Phi-2, Q = Qwen2.5-1.5B.
<table><tr><td># Judges</td><td># Examples</td><td>Accuracy (%)</td></tr><tr><td>1</td><td>17,982</td><td>89.74</td></tr><tr><td>2</td><td>11,436</td><td>90.21</td></tr><tr><td>3</td><td>7,905</td><td> $\mathrm { ~ S ~ } + \mathrm { ~ P ~ }$  90.96  $\mathrm { S + P + Q }$ </td></tr></table>

Compared to large-model ensembles (Table 6), small judge models yield moderately lower accuracy and weaker filtering precision. Nevertheless, performance improves consistently with ensemble size, and even a single lightweight judge provides substantial robustness gains. These results indicate that our framework degrades gracefully under weaker validation models, supporting its applicability in low-resource and costconstrained settings.

## B.8 Dataset Comparison

To gain insights into the relationship between the data generated in our experiment and existing benchmarks, we first extracted the 10 most frequent non-stopwords from each dataset. This qualitative analysis highlights topical overlap and domain shifts. To quantify similarity more rigorously, we computed two complementary metrics across seven collections-SNLI Train, BGE-generated, BM25-generated, SNLI Test, Adversarial NLI, Multi-NLI, and our hybrid BGE+BM25-generated set: TF-IDF cosine similarity and BERTScore F1 (Zhang et al., 2020).

TF-IDF Cosine Similarity. Let each dataset D be represented by a TF-IDF vector $\mathbf { v } _ { D } \in \mathbb { R } ^ { n }$ , where n is the vocabulary size and the ith component is

$$
\begin{array} { r } { v _ { D , i } = \mathrm { T F } _ { D , i } \cdot \log \bigl ( \frac { N } { \mathrm { D F } _ { i } } \bigr ) , } \end{array}
$$

with $\mathrm { T F } _ { D , i }$ the term frequency in D, N the total number of datasets, and $\mathrm { D F } _ { i }$ the number of datasets containing term i. We then define

$$
\mathrm { s i m } _ { \mathrm { T F I D F } } ( D , D ^ { \prime } ) = \frac { \mathbf { v } _ { D } \cdot \mathbf { v } _ { D ^ { \prime } } } { \left\| \mathbf { v } _ { D } \right\| \left\| \mathbf { v } _ { D ^ { \prime } } \right\| } .
$$

Figure 8 shows the resulting $7 \times 7$ matrix. Notably, the hybrid BGE+BM25 set has a TF-IDF similarity of approximately 0.0251 with SNLI Train, 0.0188 with SNLI Test, and 0.0150 with Multi-NLI-intermediate between its BGE-only and BM25-only counterparts.

BERTScore F1. We next measure semantic overlap by applying BERTScore F1, which aligns token embeddings from a pre-trained transformer and computes an $F _ { 1 }$ score:

$$
\mathrm { P } = \frac { 1 } { | x | } \sum _ { t \in x } \operatorname* { m a x } _ { s \in y } \cos ( \mathbf { e } _ { t } , \mathbf { e } _ { s } ) , \mathrm { R } = \frac { 1 } { | y | } \sum _ { s \in y } \operatorname* { m a x } _ { t \in x } \cos ( \mathbf { e } _ { s } , \mathbf { e } _ { t } ) ,
$$

$$
\mathrm { F 1 = 2 \cdot \frac { P R } { P + R } } ,
$$

where $x , y$ are token sequences from two datasets and e are contextual embeddings. Figure 9 displays the $7 \times 7$ BERTScore F1 matrix. The hybrid set scores about 0.8658 with SNLI Train, 0.8534 with SNLI Test, 0.8458 with Adversarial NLI, and 0.8554 with Multi-NLI, again falling between its BGE-only and BM25-only pairs. These results confirm that our validated adversarial examples share both lexical and semantic patterns with standard NLI benchmarks, while still introducing novel, challenging variations.

From Figure 8, we see that both BGE- and BM25-generated data share moderate lexical overlap with the original SNLI Train set (cosine similarities around 0.02-0.03), but diverge more substantially from the Adversarial NLI and Multi-NLI benchmarks. In contrast, Figure 9 shows that semantically these generated datasets align much more closely with SNLI Train and SNLI Test (BERTScore F1 values above 0.85), indicating that although the surface vocabulary varies, the core contextual meaning is well preserved.

## B.9 Generated Dataset Characteristics and Hypothesis Lengths

We first examined the most frequent tokens in each corpus to identify thematic patterns. In the SNLI train (Bowman et al., 2015) and SNLI test (Bowman et al., 2015) sets, words like “man,” “woman,” and “people” dominate, reflecting descriptions of social interactions. The Adversarial NLI dataset (Nie et al., 2019) shifts focus to media and chronology, with top tokens such as “film,” “first,” and “scene,” while the Multi-NLI test set (Williams et al., 2018) uses more abstract, domain-diverse language-terms like “author,” “context,” and “claim” appear frequently.

![](images/f5b19cb09878f61720126dcfcc6008c6225d732b91d0552213f5e9594b70370a.jpg)

Figure 8: Pairwise TF-IDF cosine similarity between datasets.  
![](images/3fc419a77242a3817a6e97ebc28af28b7e3a5cd84d868a417c7092a80d8e5422.jpg)

Figure 9: Pairwise BERTScore F1 between datasets.  
![](images/9afa34d7573421354be783b144494b5d9b119eedd0b51e63ffd19e2a75b4ab3c.jpg)  
Figure 10: Comparison of average hypothesis lengths (in characters and words) across datasets: Generated-BM25, Generated-BGE, SNLI train (Bowman et al., 2015), SNLI test (Bowman et al., 2015), Adversarial NLI (Nie et al., 2019), and Multi-NLI (Williams et al., 2018).

Generated-BGE and BGE+BM25-we again see a high incidence of speculative and gender-related terms (“could,” “would,” “woman,” “he,” “she”), confirming that all retrieval strategies surface similar thematic content with only minor stylistic diferences.

Figure 10 compares the average hypothesis lengths across all seven datasets. Each of the generated sets produces the longest hypotheses-around 98-100 characters (16-17 words)-demonstrating the LLM’s tendency toward more elaborate constructions when given rich few-shot contexts. By contrast, the SNLI train and SNLI test annotations remain quite concise (≈ 37-38 characters, 7-8 words), reflecting the brevity of human-written examples. The Adversarial NLI instances average ≈ 64 characters (11 words), and the Multi-NLI examples average ≈ 56 characters (10 words), underscoring their intermediate complexity. These length patterns highlight how our adversarial RAG pipeline generates richer, more challenging hypotheses while preserving diversity across data sources.

## B.10 Retrieval Accuracy Across Similarity Metrics

For purely lexical retrieval we employ BM25 with parameters $k _ { 1 } = 1 . 5$ and $b = 0 . 7 5$ . The BM25 score for a query p and document x is given by

$$
s _ { \mathrm { B M 2 5 } } ( p , x ) = \sum _ { t \in p } \mathrm { I D F } ( t ) \frac { \mathrm { t f } ( t , x ) \left( k _ { 1 } + 1 \right) } { \mathrm { t f } ( t , x ) + k _ { 1 } \left( 1 - b + b \frac { | x | } { \mathrm { a v g d } } \right) } ,\tag{71}
$$

and for each label $y ^ { \prime }$ we retrieve the top-k documents

$$
\mathcal { C } _ { p } ^ { \mathrm { l e x } } ( y ^ { \prime } ) = \arg \operatorname* { m a x } _ { S \subseteq \mathcal { D } _ { y ^ { \prime } } } \sum _ { x \in S } s _ { \mathrm { B M 2 5 } } ( p , x ) .\tag{72}
$$

For embedding-based retrieval, we first compute cosine similarity

$$
S _ { \mathrm { c o s } } ( E _ { I } , E _ { \mathcal { D } } ) = \frac { E _ { I } \cdot E _ { \mathcal { D } } } { \| E _ { I } \| _ { 2 } \| E _ { \mathcal { D } } \| _ { 2 } } ,\tag{73}
$$

and raw dot product

$$
S _ { \mathrm { d p } } ( E _ { I } , E _ { \mathcal { D } } ) = E _ { I } \cdot E _ { \mathcal { D } } = \sum _ { i = 1 } ^ { d } ( E _ { I } ) _ { i } \left( E _ { \mathcal { D } } \right) _ { i } .\tag{74}
$$

We additionally assess two norm-based distances: the $L _ { 2 }$ distance

$$
d _ { 2 } ( E _ { I } , E _ { \mathcal { D } } ) = \| E _ { I } - E _ { \mathcal { D } } \| _ { 2 } = \sqrt { \sum _ { i = 1 } ^ { d } \bigl ( ( E _ { I } ) _ { i } - ( E _ { \mathcal { D } } ) _ { i } \bigr ) ^ { 2 } } ,\tag{75}
$$

and the $L _ { 1 }$ distance

$$
d _ { 1 } ( E _ { I } , E _ { \mathcal { D } } ) = \| E _ { I } - E _ { \mathcal { D } } \| _ { 1 } = \sum _ { i = 1 } ^ { d } \lvert ( E _ { I } ) _ { i } - ( E _ { \mathcal { D } } ) _ { i } \rvert .\tag{76}
$$

Finally, to capture distributional discrepancies we examine the Bray-Curtis distance

$$
d _ { \mathrm { { B C } } } ( E _ { I } , E _ { \mathcal { D } } ) = \frac { \sum _ { i = 1 } ^ { d } \left| ( E _ { I } ) _ { i } - ( E _ { \mathcal { D } } ) _ { i } \right| } { \sum _ { i = 1 } ^ { d } \left| ( E _ { I } ) _ { i } + ( E _ { \mathcal { D } } ) _ { i } \right| } ,\tag{77}
$$

and the Canberra distance

$$
d _ { \mathrm { C a n } } ( E _ { I } , E _ { \mathcal { D } } ) = \sum _ { i = 1 } ^ { d } \frac { \left| ( E _ { I } ) _ { i } - ( E _ { \mathcal { D } } ) _ { i } \right| } { \left| ( E _ { I } ) _ { i } \right| + \left| ( E _ { \mathcal { D } } ) _ { i } \right| } .\tag{78}
$$

![](images/60679414197aebbebc8eb731e792bfc83371a0d285ea2d0932621a2407d8b69e.jpg)  
Figure 11: Retrieval accuracy (%) by similarity metric for BGE+BM25, BM25, and BGE.

Figure 11 demonstrates that BGE+BM25 outperforms both BM25 alone and BGE alone across all six metrics, achieving 92.60% (cosine), 89.85% (dot product), 85.43% $\left( L _ { 2 } \right)$ , 85.22% $\left( L _ { 1 } \right)$ , 79.21% (Bray-Curtis) and 79.12% (Canberra). Pure BM25 and pure BGE match closely on cosine but degrade more sharply on norm- and distribution-based distances, confirming the robustness of the hybrid lexical-semantic approach.

## B.11 Hyperparameter Optimization and Reproducibility

To ensure fair and reproducible evaluation, all target models are fine-tuned using a standardized hyperparameter optimization protocol. We employ Bayesian optimization via Optuna to search over learning and regularization parameters, using validation accuracy as the objective.

Tokenization and Input Representation. All premise–hypothesis pairs are tokenized using the RoBERTa tokenizer with a maximum sequence length of 128. Inputs are padded and truncated to fixed length to ensure consistent batch construction across runs. Each example is represented by input IDs, attention masks, and class labels.

Training and Evaluation Splits. For eficiency during hyperparameter tuning, we use the full augmented training set and a fixed validation subset of 1,200 examples. Samples with undefined labels are removed prior to evaluation. All datasets are formatted in PyTorch tensors.

Search Space. We optimize the following hyperparameters:

$$
\eta \sim \mathrm { L o g U n i f o r m } ( 1 0 ^ { - 6 } , 1 0 ^ { - 4 } ) ,\tag{79}
$$

$$
E \sim \{ 1 , 2 , 3 , 5 \} ,\tag{80}
$$

$$
B \sim \{ 1 , 2 , 4 , 8 , 1 6 \} ,\tag{81}
$$

$$
\lambda \sim \mathrm { U n i f o r m } ( 1 0 ^ { - 4 } , 1 0 ^ { - 2 } ) ,\tag{82}
$$

where η denotes the learning rate, E the number of training epochs, B the per-device batch size, and λ the weight decay coeficient.

Optimization Procedure. For each trial, a RoBERTa-based classifier is fine-tuned using the HuggingFace Trainer framework. Models are evaluated at the end of each epoch, and the best-performing checkpoint is retained based on validation accuracy. Early stopping is implicitly enforced by selecting the best epoch. We perform 40 independent trials and select the configuration that maximizes validation accuracy.

All experiments are conducted on a single NVIDIA A100 GPU. Each training epoch requires approximately 3.11 minutes on average.

Evaluation Metric. All hyperparameter configurations are evaluated using classification accuracy:

$$
\operatorname { A c c } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \left[ \hat { y } _ { i } = y _ { i } \right] ,\tag{83}
$$

where $\hat { y } _ { i }$ and $y _ { i }$ denote predicted and ground-truth labels for sample $i ,$ respectively.

Reproducibility Measures. To reduce variance across runs, we fix random seeds for data sampling, model initialization, and optimization. All experiments use identical preprocessing, prompt templates, and evaluation splits. Hyperparameter search spaces, optimization budgets, and validation subsets are fully specified to enable exact replication of our results.

The complete training and optimization scripts will be released upon publication.

## B.12 Illustrative Example: Failure-Mode Bandit Curation for NLI

Figure 12 illustrates one iteration of the proposed framework on a Natural Language Inference (NLI) example. The example is intended to show the operational flow of the method and does not represent a full training run.

Given a premise and target label, the generator produces multiple candidate hypotheses conditioned on retrieved few-shot examples. The current target model first filters these candidates by retaining only those that induce an incorrect prediction. The remaining candidates are then checked by an automated LLM judge ensemble to ensure label consistency.

Validated failures are embedded and grouped into failure-mode clusters. A contextual-bandit policy observes the state of each failure mode and selects which modes should be sampled under the adversarial budget. The selected examples are mixed with original training data and used to retrain the target model.

After retraining, validation performance provides reward $G _ { t }$ , which updates the policy $\pi _ { \theta }$ and critic $R _ { \phi }$ Thus, the framework does not select examples by a fixed reward threshold; instead, it learns across iterations which recurring failure modes are most useful for improving robustness.

Although the example is shown for NLI, the same failure filtering, automated validation, failure-mode clustering, and policy-guided sampling mechanism can be applied to other classification and reasoning tasks considered in this work.

## B.13 Sensitivity to Selection Threshold and Reward Noise

We analyze the robustness of our framework with respect to the selection threshold τ and noise in reward estimation. Since τ controls the trade-of between data quality and coverage, and reward estimates are derived from noisy downstream feedback, understanding their impact is critical for stable optimization.

Sensitivity to Selection Threshold. We varied τ over a wide range relative to the empirical reward distribution, selecting values corresponding to the 60th, 70th, 80th, and 90th reward percentiles. Lower thresholds admit more adversarial candidates, while higher thresholds enforce stricter filtering. All experiments were conducted using $\alpha = 0 . 8 3$ and a 1:4 mixing ratio.

Reward Noise Injection. To simulate imperfect reward estimation, we injected additive Gaussian noise into the predicted reward:

$$
\begin{array} { r } { \tilde { r } ( x ) = r ( x ) + \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , } \end{array}\tag{84}
$$

where σ controls noise magnitude. We evaluate $\sigma \in \{ 0 . 0 5 , 0 . 1 , 0 . 2 \}$ , covering mild to severe corruption regimes.

Table 10 reports performance under varying τ and noise levels on SNLI. Performance remains stable across a broad operating region. Moderate deviations from the default threshold $( \tau ^ { \star } )$ incur only minor degradation, and the system tolerates substantial reward noise before significant accuracy loss occurs.

![](images/04e656bac07a1e58467624de656ae44f86dd45219a5bb9a1e016cae637b86acf.jpg)  
Figure 12: Illustration of one failure-mode contextual bandit curation iteration on a Natural Language Inference (NLI) example. Given a premise and label, retrieval-augmented prompting generates candidate hypotheses, which are first filtered by the target model to retain incorrect predictions and then validated by an automated LLM judge ensemble. Validated failures are clustered into failure modes, and a contextualbandit policy selects which modes to sample for retraining under an adversarial budget. Validation reward $G _ { t }$ updates the policy $\pi _ { \theta }$ and critic $R _ { \phi }$ , enabling adaptive selection of high-impact failure modes across iterations.

Table 10: Sensitivity to selection threshold τ and reward noise (SNLI, 1:4 gen:orig, $\alpha = 0 . 8 3 )$
<table><tr><td>τ (Percentile)</td><td>Noise σ</td><td>Accuracy (%)</td></tr><tr><td>60%</td><td>0.00</td><td>91.94</td></tr><tr><td>70%</td><td>0.00</td><td>92.21</td></tr><tr><td>80%  $( \tau ^ { \star } )$ </td><td>0.00</td><td>92.60</td></tr><tr><td>90%</td><td>0.00</td><td>92.17</td></tr><tr><td>80%</td><td>0.05</td><td>92.41</td></tr><tr><td>80%</td><td>0.10</td><td>92.05</td></tr><tr><td>80%</td><td>0.20</td><td>91.62</td></tr></table>

Results indicate that the proposed framework operates in a broad stability regime. The performance plateau around $\tau ^ { \star }$ suggests that the system is not finely tuned to a narrow threshold range. Moreover, robustness to moderate reward noise is consistent with our theoretical bounded-drift analysis (Section 3.4), which guarantees controlled distributional evolution under noisy feedback. Together, these findings demonstrate that our approach is resilient to practical imperfections in reward estimation and threshold calibration.

## B.14 Example - Few-Shot Chat Sequence

BGE based retrieval The chat sequences below present a clear few-shot retrieval sequence for a natural language inference task. They illustrate six premise-hypothesis pairs-two each for entailment, neutral, and contradiction-and conclude with a concise model prompt. This format makes the example selection process transparent and highlights the model’s reasoning in a single, easily readable block. These examples are based solely on BGE retrieval.

<table><tr><td>Example 1: Few-Shot Retrieval &amp; Model Return</td></tr><tr><td>Shot 1</td></tr><tr><td>Premise: A blond little girl enjoying a burrito. Label: entailment.</td></tr><tr><td>Hypothesis: The girl ate a burrito.</td></tr><tr><td>Shot 2 Premise: A young blond girl sitting down while eating.</td></tr><tr><td>Label: entailment. Hypothesis: The girl has food.</td></tr><tr><td>Shot 3 Premise: A blond little girl enjoying a burrito.</td></tr><tr><td>Label: neutral. Hypothesis: The hungry girl ate a burrito at the restaurant.</td></tr><tr><td>Shot 4 Premise: A young blond girl sitting down while eating.</td></tr><tr><td>Label: neutral.</td></tr><tr><td>Hypothesis: The girl is eating at a picnic. Shot 5</td></tr><tr><td>Premise: A blond little girl enjoying a burrito. Label: contradiction.</td></tr><tr><td>Hypothesis: The brunette girl didn&#x27;t like the burrito.</td></tr><tr><td>Shot 6 Premise: A young blond girl sitting down while eating.</td></tr><tr><td>Label: contradiction.</td></tr><tr><td>Hypothesis: The girl runs all over her house while eating because she can never sit down. Llama Generation</td></tr><tr><td>User: Now generate a one-sentence hypothesis that contradicts the premise above. Return only the hypothesis without narration.</td></tr></table>

<table><tr><td>Example 2: Few-Shot Retrieval &amp; Model Return</td><td></td></tr><tr><td>Shot 1 Premise: An older man is drinking orange juice at a restaurant.</td><td></td></tr><tr><td>Label: entailment. Hypothesis: A man is drinking juice.</td><td></td></tr><tr><td>Shot 2</td><td></td></tr><tr><td>Premise: An older man is drinking orange juice at a restaurant. Label: entailment.</td><td></td></tr><tr><td>Hypothesis: A man is drinking juice. Shot 3</td><td></td></tr><tr><td>bright colored shirts smile in the background.</td><td>Premise: An older man sits with his orange juice at a small table in a coffee shop while employees in</td></tr><tr><td>Label: neutral. Hypothesis: An elderly man sits in a small shop.</td><td></td></tr><tr><td>Shot 4</td><td></td></tr><tr><td>bright colored shirts smile in the background.</td><td>Premise: An older man sits with his orange juice at a small table in a coffee shop while employees in</td></tr><tr><td>Label: neutral. Hypothesis: An older man drinks his juice as he waits for his daughter to get off work.</td><td></td></tr><tr><td>Shot 5</td><td>Premise: An older man sits with his orange juice at a small table in a coffee shop while employees in</td></tr><tr><td>bright colored shirts smile in the background.</td><td></td></tr><tr><td>Label: contradiction. Hypothesis: A boy flips a burger.</td><td></td></tr><tr><td>Shot 6</td><td></td></tr><tr><td>Premise: An older man is drinking orange juice at a restaurant.</td><td></td></tr><tr><td>Label: contradiction.</td><td></td></tr><tr><td></td><td></td></tr><tr><td>Hypothesis: Two women are at a restaurant drinking wine.</td><td></td></tr><tr><td>Llama Generation</td><td></td></tr><tr><td></td><td></td></tr><tr><td>User: Now generate a one-sentence hypothesis that contradicts the premise above. Return only the</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>hypothesis without narration.</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>Assistant (Llama): The coffee shop is empty.</td></tr></table>

## Example 3: Few-Shot Retrieval & Model Return

<table><tr><td>Example 3: Few-Shot Retrieval &amp; Model Return Shot 1</td></tr><tr><td>Premise: A few people in a restaurant setting, one of them is drinking orange juice. Label: entailment.</td></tr><tr><td>Hypothesis: The diners are at a restaurant.</td></tr><tr><td>Shot 2 Premise: A few people in a restaurant setting, one of them is drinking orange juice.</td></tr><tr><td>Label: entailment.</td></tr><tr><td>Hypothesis: The diners are at a restaurant.</td></tr><tr><td>Shot 3 Premise: A few people in a restaurant setting, one of them is drinking orange juice.</td></tr><tr><td>Label: neutral. Hypothesis: The people are eating omelettes.</td></tr><tr><td>Shot 4 Premise: A few people in a restaurant setting, one of them is drinking orange juice.</td></tr><tr><td>Label: neutral. Hypothesis: The people are eating omelettes.</td></tr><tr><td>Shot 5 Premise: A few people in a restaurant setting, one of them is drinking orange juice.</td></tr><tr><td>Label: contradiction. Hypothesis: The people are sitting at desks in school.</td></tr><tr><td>Shot 6 Premise: A few people are observing the orange juice section at the grocery store.</td></tr><tr><td>Label: contradiction.</td></tr><tr><td>Hypothesis: The people are at a baseball stadium. Llama Generation</td></tr><tr><td>User: Now generate a one-sentence hypothesis that contradicts the premise above. Return only the hypothesis without narration.</td></tr></table>

Optimized BGE + BM25 Retrieval with Tuned Alpha The paragraph below describes an optimized retrieval approach that combines semantic search using BGE embeddings with lexical scoring via BM25. By introducing a tunable alpha parameter, the method linearly interpolates between embedding-based similarity and keyword matching-enabling a dynamic balance that maximizes relevance across diverse query types. Through cross-validation, the optimal alpha value was determined to achieve superior retrieval performance compared to either technique alone.

<table><tr><td>Example 1: Few-Shot Retrieval &amp; Model Return</td><td></td></tr><tr><td>Shot 1 Premise: A person on a horse jumps over a broken down airplane.</td><td></td></tr><tr><td>Label: entailment. Hypothesis: A person is outdoors, on a horse.</td><td></td></tr><tr><td>Shot 2 Premise: A person on a horse jumps over a broken down airplane.</td><td></td></tr><tr><td>Label: entailment. Hypothesis: A person is outdoors, on a horse.</td><td></td></tr><tr><td>Shot 3 Premise: A person on a horse jumps over a broken down airplane.</td><td></td></tr><tr><td>Label: neutral. Hypothesis: A person is training his horse for a competition.</td><td></td></tr><tr><td>Shot 4 Premise: A person is riding a horse, which is jumping over a grounded airplane.</td><td></td></tr><tr><td>Label: neutral. Hypothesis: The horse jumps over a toy airplane.</td><td></td></tr><tr><td>Shot 5</td><td></td></tr><tr><td>Premise: A person on a horse jumps over a broken down airplane. Label: contradiction.</td><td></td></tr><tr><td>Hypothesis: A person is at a diner, ordering an omelette. Shot 6</td><td></td></tr><tr><td>Premise: A person on a horse jumps over a broken down airplane.</td><td></td></tr><tr><td>Label: contradiction.</td><td></td></tr><tr><td>Hypothesis: A person is at a diner, ordering an omelette.</td><td></td></tr><tr><td>Llama Generation</td><td></td></tr><tr><td>User: Now generate a one-sentence hypothesis that is neutral with the premise above. Return only the</td><td></td></tr><tr><td></td><td></td></tr><tr><td>hypothesis without narration.</td><td></td></tr><tr><td></td><td></td></tr><tr><td>Assistant (Llama): The horse is very well trained.</td><td></td></tr></table>

## Example 2: Few-Shot Retrieval & Model Return

<table><tr><td>Shot 1 Premise: Children smiling and waving at camera.</td></tr><tr><td>Label: entailment. Hypothesis: There are children present.</td></tr><tr><td>Shot 2</td></tr><tr><td>Premise: Children smiling and waving at camera. Label: entailment.</td></tr><tr><td>Hypothesis: There are children present. Shot 3</td></tr><tr><td>Premise: Children smiling and waving at camera. Label: neutral.</td></tr><tr><td>Hypothesis: They are smiling at their parents. Shot 4</td></tr><tr><td>Premise: Children smiling and waving at camera. Label: neutral.</td></tr><tr><td>Hypothesis: They are smiling at their parents. Shot 5</td></tr><tr><td>Premise: Children smiling and waving at camera. Label: contradiction.</td></tr><tr><td>Hypothesis: The kids are frowning. Shot 6</td></tr><tr><td>Premise: Children smiling and making faces to the camera. Label: contradiction.</td></tr><tr><td>Hypothesis: The children are deceased.</td></tr><tr><td>Llama Generation</td></tr><tr><td>User: Now generate a one-sentence hypothesis that contradicts the premise above. Return only the hypothesis without narration. Assistant (Llama): The children are asleep.</td></tr></table>

## Example 3: Few-Shot Retrieval & Model Return

<table><tr><td>Shot 1</td></tr><tr><td>Premise: An older man is drinking orange juice at a restaurant. Label: entailment.</td></tr><tr><td>Hypothesis: A man is drinking juice. Shot 2</td></tr><tr><td>Premise: An older man is drinking orange juice at a restaurant. Label: entailment.</td></tr><tr><td>Hypothesis: A man is drinking juice. Shot 3</td></tr><tr><td>Premise: An older man is drinking orange juice at a restaurant. Label: neutral.</td></tr><tr><td>Hypothesis: A man in a restaurant is waiting for his meal to arrive. Shot 4</td></tr><tr><td>Premise: An elderly man is drinking orange juice at a cafe. Label: neutral.</td></tr><tr><td>Hypothesis: An older gentleman is enjoying his orange juice at a new cafe. Shot 5</td></tr><tr><td>Premise: An older man is drinking orange juice at a restaurant. Label: contradiction.</td></tr><tr><td>Hypothesis: Two women are at a restaurant drinking wine. Shot 6</td></tr><tr><td>Premise: An older man is drinking orange juice at a restaurant. Label: contradiction.</td></tr><tr><td>Hypothesis: Two women are at a restaurant drinking wine.</td></tr><tr><td>Llama Generation User: Now generate a one-sentence hypothesis that is neutral with the premise above. Return only the</td></tr><tr><td>hypothesis without narration.</td></tr><tr><td>Assistant (Llama): The restaurant has a lot of tables.</td></tr></table>

BM25 based retrieval The retrieval process relies exclusively on the BM25 algorithm, which computes scores based on term frequency, document frequency, and document length normalization. By focusing on exact keyword matches and weighting rarer terms more heavily, BM25 efectively ranks documents that share the most relevant vocabulary with the query. This approach is well-suited for tasks where precise lexica overlap is paramount and semantic generalization is less critical. However, it may miss contextually related content when synonyms or paraphrases are used.

<table><tr><td>Example 1: Few-Shot Retrieval &amp; Model Return</td></tr><tr><td>Shot 1 Premise: An older man is drinking orange juice at a restaurant.</td></tr><tr><td>Label: entailment.</td></tr><tr><td>Hypothesis: A man is drinking juice. Shot 2</td></tr><tr><td>Premise: An elderly man is drinking orange juice at a cafe. Label: entailment.</td></tr><tr><td>Hypothesis: An old man is enjoying a beverage at a cafe. Shot 3</td></tr><tr><td>Premise: An older man is drinking orange juice at a restaurant. Label: neutral.</td></tr><tr><td>Hypothesis: A man in a restaurant is waiting for his meal to arrive. Shot 4</td></tr><tr><td>Premise: An older man is drinking orange juice at a restaurant. Label: neutral.</td></tr><tr><td>Hypothesis: A man in a restaurant is waiting for his meal to arrive. Shot 5</td></tr><tr><td>Premise: An older man is drinking orange juice at a restaurant. Label: contradiction.</td></tr><tr><td>Hypothesis: Two women are at a restaurant drinking wine. Shot 6</td></tr><tr><td>Premise: An older man is drinking orange juice at a restaurant. Label: contradiction.</td></tr><tr><td>Hypothesis: Two women are at a restaurant drinking wine. Llama Generation</td></tr><tr><td>User: Now generate a one-sentence hypothesis that entails the premise above. Return only the hypothesis without narration.</td></tr><tr><td>Assistant (Llama): A man is drinking orange juice at a restaurant.</td></tr></table>

## Example 2: Few-Shot Retrieval & Model Return

<table><tr><td>Shot 1 Premise: An older man is drinking orange juice at a restaurant.</td></tr><tr><td>Label: entailment. Hypothesis: A man is drinking juice.</td></tr><tr><td>Shot 2 Premise: An older man is drinking orange juice at a restaurant.</td></tr><tr><td>Label: entailment. Hypothesis: A man is drinking juice.</td></tr><tr><td>Shot 3 Premise: An older man is drinking orange juice at a restaurant.</td></tr><tr><td>Label: neutral. Hypothesis: A man in a restaurant is waiting for his meal to arrive.</td></tr><tr><td>Shot 4 Premise: An older man is drinking orange juice at a restaurant.</td></tr><tr><td>Label: neutral. Hypothesis: A man in a restaurant is waiting for his meal to arrive.</td></tr><tr><td>Shot 5 Premise: An older man is drinking orange juice at a restaurant.</td></tr><tr><td>Label: contradiction. Hypothesis: Two women are at a restaurant drinking wine.</td></tr><tr><td>Shot 6 Premise: An elderly man is drinking orange juice at a cafe.</td></tr><tr><td>Label: contradiction. Hypothesis: An elderly man is drinking apple juice at a bar.</td></tr><tr><td>Llama Generation</td></tr><tr><td>User: Now generate a one-sentence hypothesis that contradicts the premise above. Return only the hypothesis without narration</td></tr><tr><td>Example 3: Few-Shot Retrieval &amp; Model Return</td></tr><tr><td>Shot 1 Premise: A woman in a green jacket and hood over her head looking towards a valley.</td></tr><tr><td>Label: entailment. Hypothesis: The woman is wearing green.</td></tr><tr><td>Shot 2 Premise: A woman in a green jacket and hood over her head looking towards a valley.</td></tr><tr><td>Label: entailment. Hypothesis: The woman is wearing green.</td></tr><tr><td>Shot 3 Premise: A woman in a green jacket and hood over her head looking towards a valley.</td></tr><tr><td>Label: neutral. Hypothesis: The woman is cold.</td></tr><tr><td>Shot 4 Premise: A woman gazes over the valley below.</td></tr><tr><td>Label: neutral.</td></tr><tr><td>Hypothesis: she looks at the valley she owns. Shot 5</td></tr><tr><td>Premise: A woman in a green jacket and hood over her head looking towards a valley. Label: contradiction.</td></tr><tr><td>Hypothesis: The woman is nake. Shot 6</td></tr><tr><td>Premise: A woman in a green jacket and hood over her head looking towards a valley.</td></tr><tr><td>Label: contradiction.</td></tr><tr><td>Hypothesis: The woman is nake. Llama Generation</td></tr><tr><td>User: Now generate a one-sentence hypothesis that is neutral with the premise above. Return only the hypothesis without narration.</td></tr></table>

## B.15 Prompt Design for Task-Specific Candidate Generation

We employ task-specific prompting strategies to guide large language models in generating adversarial candidates consistent with the target supervision signal. Prompts are designed to be concise, label-conditioned, and deterministic, ensuring controllable hypothesis synthesis and high semantic fidelity. All prompts instruct the model to return only the generated output without additional narration.

## B.15.1 SNLI Prompting Strategy

For the Natural Language Inference task, the goal is to generate a single-sentence hypothesis whose semantic relation to the given premise matches a specified target label y ∈ {entailment, neutral, contradiction}. Given a premise p and target label y, we use the following template:

System: You are a language expert that helps create an NLI dataset. Given a premise sentence and a desired label, your job is to provide a one-sentence hypothesis, such that the label is relevant to the relation between the given premise and your generated hypothesis. Make sure to keep the hypothesis short and no longer than a sentence.

User:

Premise: {premise}

Desired label: {label}

Now generate a one sentence hypothesis that {relation} the premise above. Return only the hypothesis without narration.

Here, {label} corresponds to the target class (entailment, neutral, contradiction), and {relation} maps to the appropriate semantic relation (“entails”, “is neutral with”, “contradicts”). The prompt enforces minimal length and discourages explanatory text.

We further employ low-temperature decoding to reduce sampling variance and ensure consistent adversarial patterns across iterations.

## B.15.2 FEVER Prompting Strategy

For the FEVER fact verification task, the objective is to generate evidence claims whose veracity can be evaluated with respect to a given document or knowledge source. Given an evidence context e and a target label y ∈ {SUPPORTS, REFUTES, NOT\_ENOUGH\_INFO}, we use the following template:

System: You are a language expert that helps create fact verification datasets. Given evidence text and a desired label, your job is to generate a single-sentence claim that matches the specified verification outcome. Make sure the claim is concise and factual.

User:

Evidence: {evidence}

Desired label: {label}

Now generate a one sentence claim that is {relation} by the evidence above. Return only the claim without narration.

Here, {label} corresponds to the FEVER classes, and {relation} maps to “supported by”, “refuted by”, or “cannot be verified from”. This formulation encourages the model to synthesize claims that are directly grounded in the provided evidence.

As in the NLI setting, we constrain generation length and apply low-temperature sampling to prioritize precision over diversity.

## B.15.3 Design Rationale

Across tasks, prompt templates are designed to satisfy three principles: (i) explicit conditioning on the target label, (ii) minimal linguistic ambiguity, and (iii) strict output formatting. This enables stable generation, reliable automated validation, and consistent reward estimation, facilitating efective failure-aware adversarial data curation.