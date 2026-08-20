# Evaluating and Explaining Prompt Sensitivity of LLMs Using Interactions

Ruiyang Qin1 Qingzhuo Wang 1 Tian Wang1 Zhihua Wei1 Wen Shen1 †

## Abstract

The remarkable capabilities of large language models (LLMs) are often undermined by their instability. Even subtle and semantically irrelevant changes in prompts can cause dramatic fluctuations in performance, a phenomenon known as prompt sensitivity. Previous studies typically evaluate prompt sensitivity by comparing the LLM's final outputs when prompts change. However, such coarse-grained metrics fail to explain the internal reasons for prompt sensitivity. In this paper, we introduce interactions as a fine-grained tool to analyze prompt sensitivity of LLMs. Specifically, we decompose the output score of the LLM into a set of interactions. Each interaction represents a nonlinear relationship involving a set of input variables. We discover that subtle changes to prompts can trigger severe instability in interactions, even when the outputs of the LLM remain the same. To this end, we propose an Interaction-based Prompt Sensitivity (IPS) metric by quantifying changes in interactions when we introduce subtle changes to prompts. We apply the IPS metric to 50 opensource LLMs and uncover four factors that reduce the prompt sensitivity of LLMs, including supervised fine-tuning, increased model scales, dense architectures, and few-shot learning. More crucially, we discover a common mechanism by which these four factors reduce prompt sensitivity: all four factors tend to reduce the prompt sensitivity of low-order interactions $( i . e .$ , interactions involving few input variables).

## 1. Introduction

LLMs have demonstrated exceptional proficiency in numerous natural language processing tasks (Srivastava et al., 2023; Zhou et al., 2023; Annepaka & Pakray, 2024; Chang et al., 2024), a success largely driven by the effectiveness of prompting. However, this power is undermined by prompt sensitivity. That is, semantically unimportant changes to the prompt can result in divergent outputs (Sclar et al., 2024). Current research (Sclar et al., 2024; Chatterjee et al., 2024; Lu et al., 2024; Alzahrani et al., 2024; Errica et al., 2025; Razavi et al., 2025) on evaluating prompt sensitivity only focuses on the LLM's final output. These output-based metrics typically measure changes in performance, such as task accuracy or output consistency. As coarse-grained measures, these metrics only reveal the consequences of prompt sensitivity $( e . g .$ , the LLM's prediction changes from one answer to another one) but fail to explain its underlying reasons.

In this paper, we aim to evaluate the prompt sensitivity of LLMs from a fine-grained perspective and investigate the underlying reasons why certain factors can decrease the prompt sensitivity of LLMs. Recent research (Chen et al., 2024; Zhou et al., 2024; Ren et al., 2025; 2024b) has utilized interactions to explain the fine-grained inference logic of deep neural networks (DNNs). Inspired by these studies, we introduce the interactions framework to fine-grainedly analyze the prompt sensitivity of LLMs.

Specifically, given a sentence x with n input variables $( e . g .$ words or tokens) indexed by $N = \{ 1 , 2 , \dots , n \}$ , an interaction represents an intricate nonlinear relationship associated with a specific combination of input variables. Consider the sentence $\textbf { \em x } = \{ { \overset { \cdot } { \cdot } } H e$ is a green hand." In this context, the idiom “green hand" carries the meaning of “beginner". The joint presence of the input variables in the set $S = \{ g r e e n , h a n d \} \subseteq N$ triggers a special interaction effect. This interaction effect, denoted as $I _ { S } ,$ pushes the network's inference towards the semantic meaning of "beginner." Li & Zhang (2023) have mathematically demonstrated that the scalar output $v ( { \pmb x } )$ of a DNN is always equivalent to the output of an interaction-based logical model $\begin{array} { r } { \phi ( { \pmb x } ) = \sum _ { S \subset N } I _ { S } } \end{array}$ That is, ${ \begin{array} { r } { v ( \mathbf { x } ) = \phi ( \mathbf { x } ) = \sum _ { S \subseteq N } I _ { S } } \end{array} }$ . Thus, the inference logic of a DNN can be explained by a set of interactions.

Coarse-grained analysis vs. ine-grained analysis. Traditional analysis of prompt sensitivity can only coarsely reflect whether the LLM's prediction alters when the prompt changes. Beyond this, we introduce an interaction-based logical model φ(x) as a fine-grained analytical tool to analyze the stable and unstable interactions encoded by the LLM for each specific sample. As shown in Figure 1, when we keep the same input x but introduce semantically equivalent alterations to the prompt template T (e.g., change “Answers" to “ANSWERS"), traditional coarse-grained analysis can only reveal that the LLM's prediction changes from the correct answer “B" to the incorrect answer “A" but fails to explain why or how this change occurs. In contrast, our interactionbased analysis precisely identifies unstable interactions that may contribute to the prompt sensitivity of the LLM. An interaction is considered stable if its effect changes minimally (e.g., I{lay eggs} shifts from 0.6242 to 0.6076). Conversely, an interaction is deemed unstable if its effect fluctuates dramatically (e.g., I {nor, lay, eggs, dogs} shifts from 0.5175 to -0.0594). Strikingly, we find that unstable interactions exist even when the LLM's final output remains the same. This indicates that our fine-grained analysis reveals potential instability that is entirely invisible to traditional, output-level metrics.

![](images/1c28e7543597047b66ccf9b71dd077fc21188d7bf37b820cdc593e7cb1118f9b.jpg)  
Figure 1. Using interactions for fine-grained analysis of prompt sensitivity. Given an input x and a prompt template T, the LLM's output score v(“B" |x, T) is equivalent to the output of an interaction-based logical model φ(x), i.e., $\begin{array} { r } { v ( \mathbf { \dot { \mathbf { \sigma } } } ^ { \mathrm { ~ } } B ^ { \prime \prime } | \mathbf { \dot { x } } , T ) \dot { = } \phi ( \pmb { x } ) = \sum _ { S \subset N } I _ { S } } \end{array}$ . In this way, we can uncover the underlying reasons for the prompt sensitivity of LLMs by analyzing detailed interaction patterns. Specifically. we divide interactions into those that are stable to prompt changes (i.e. stable interaction) and those that are not (i.e. unstable interaction).

Building on these findings, we leverage the interaction framework to propose a fine-grained metric to evaluate the prompt sensitivity of LLMs, which is termed Interactionbased Prompt Sensitivity (IPS). This metric quantifies changes in interaction patterns when LLMs process different prompts. We apply the proposed IPS metric to evaluate the prompt sensitivity of 50 open-source LLMs. However, drawing conclusions about which LLM families are more or less sensitive from this ranking is challenging, as an LLM's prompt sensitivity stems from multiple, intertwined factors

To this end, we conduct a series of comparative experiments to disentangle these factors, discovering four factors that reduce the prompt sensitivity of LLMs: (1) Supervised finetuning reduces the prompt sensitivity. Instruct/chat models (with supervised fine-tuning) exhibit lower prompt sensitivity than base models. (2) LLMs with larger parameter numbers exhibit lower prompt sensitivity. (3) Dense models are generally less sensitive than mixture-of-experts (MoE) models. (4) Few-shot learning considerably reduces prompt sensitivity compared to 0-shot learning.

More crucially, we explore and uncover a common underlying mechanism that explains how the four aforementioned factors reduce prompt sensitivity: they primarily reduce the instability of low-order interactions (i.e., interactions involving a small number of input variables). This finding is counterintuitive, as our experiments demonstrate that high-order interactions (i.e., interactions involving many input variables) tend to exhibit the highest sensitivity, while low-order interactions are inherently less sensitive. Unexpectedly, these factors further stabilize the already stable low-order interactions, yet remain ineffective in addressing the more pronounced sensitivity of high-order interactions.

## 2. Related Work

Prompt sensitivity of LLMs. Previous studies (Sun et al. 2024; Zhu et al., 2024a; Sclar et al., 2024; Alzahrani et al., 2024) demonstrated that LLMs are highly sensitive to minor perturbations or semantically unimportant alterations to prompts, which can lead to significant performance variation. Such prompt sensitivity presents a considerable risk to the reliability of LLMs. Existing metrics (Sclar et al., 2024; Chatterjee et al., 2024; Lu et al., 2024; Zhuo et al., 2024; Cao et al., 2024; Errica et al., 2025; Razavi et al., 2025) for evaluating the prompt sensitivity of LLMs typically measure shifts in final outputs, such as task accuracy or output consistency. However, these metrics are coarse-grained and fail to probe the LLM's internal logic. In this paper, we propose a fine-grained metric that evaluates prompt sensitivity based on interactions. This framework enables us to uncover the underlying mechanisms of prompt sensitivity in LLMs.

Using game-theoretic interactions to explain DNNs. Traditional methods for explanations (Tenney et al., 2020; Zhang et al., 2020) often lack mathematical guarantees of faithfulness, meaning their outputs may not accurately represent the internal logic employed by the DNN. To this end, Ren et al. (2023a) proposed to use interactions between input variables to explain DNNs and provided a series of theoretical guarantees for the method's validity. Furthermore, it has been empirically discovered (Li & Zhang, 2023) and theoretically proven (Ren et al., 2024a) that a DNN typically encodes only a sparse set of interactions. At the application level, the interaction framework has proven effective in a wide range of complex tasks, including adversarial transferability (Deng et al., 2024), model generalization (Chen et al., 2024; Zhou et al., 2024), model training process (Ren et al., 2025; 2024b), overfitting (Ren et al., 2023b) and other tasks (Shen et al., 2024; Li et al., 2025; Wen et al., 2026). In this paper, we use the interaction framework to analyze the prompt sensitivity of LLMs.

## 3. Interaction-Based Analysis of the Prompt Sensitivity of LLMs

## 3.1. Preliminaries: Interactions

This subsection introduces the definition of interactions, as well as the mathematical guarantees of interaction-based explanation. Given a DNN v and an input sentence x with n input variables $( e . g .$ , words) indexed by $N = \{ 1 , 2 , \dots , n \}$ let $v ( { \pmb x } ) \in \mathbb { R }$ denote the scalar output of the DNN. Here we set $\begin{array} { r } { v ( \pmb { x } ) = \log \frac { p ( y = y ^ { * } | \pmb { x } ) } { 1 - p ( y = y ^ { * } | \pmb { x } ) } \in \mathbb { R } . } \end{array}$ , where $p ( y = y ^ { * } | \pmb { x } )$ represents the probability of generating the ground truth token $y ^ { * }$ given the input x. We define a surrogate logical model φ(x) to match the scalar output $v ( { \pmb x } )$ of the DNN. Recent studies (Li & Zhang, 2023; Ren et al., 2023a) have proven Theorem 3.1, which shows that the output score v(x) on any randomly masked¹ input ${ \pmb x } _ { T }$ can be accurately calculated by the following surrogate logical model $\phi ( \cdot )$

$$
\phi ( { \pmb x } _ { T } ) \triangleq \phi ( { \pmb x } _ { 0 } ) + \sum _ { S \subseteq N } { \mathbb 1 } ( S \mid { \pmb x } _ { T } ) \cdot { \boldsymbol I } _ { S } ,\tag{1}
$$

where the AND trigger function $\mathbb { 1 } ( S \mid \pmb { x } _ { T } ) ~ \in ~ \{ 0 , 1 \}$ represents an AND relationship between input variables in $S ,$ which can also be termed AND interaction pattern. The scalar weight $I _ { S }$ quantifies the effect of an AND relationship, which can also be termed interaction effect. $\mathbf { A } \mathbf { n }$ AND relationship is activated only by the joint presence of all input variables in the set S, $i . e .$ , all input variables in

S are not masked. For instance, given the input sentence $\textstyle { \pmb { x } } = { \overset { . . } { \cdot } } H e$ is a green hand," the co-occurrence of the input variables in the set $S = \{ g r e e n$ , hand} contributes a numerical effect $I _ { S }$ that pushes the surrogate logical model's inference towards the semantic meaning of "beginner." If an AND interaction $S$ is triggered, $i . e . , \ \Im ( S \mid \pmb { x } _ { T } ) = 1$ , the corresponding interaction effect $I _ { S }$ is added to the output of the logical model. Otherwise, if any word in S is masked and the AND interaction is not triggered, i.e., $\mathbf { 1 } ( S \mid \pmb { x } _ { T } ) = 0 ,$ its corresponding interaction effect $I _ { S }$ is not added to the output of the logical model. $\pmb { x } _ { \emptyset }$ represents that all input variables in N are masked.

Theorem 3.1 (Universal matching property, proven in $\mathsf { A p - }$ pendix B). Given an input x with n input variables, we randomly mask any combinations of input variables to generate $2 ^ { n }$ masked inputs $\{ { \pmb x } _ { T } \ | \ T \subseteq N \}$ . For every masked input xT, when the scalar weight $I _ { S }$ in the logical model $\phi ( \cdot )$ are set to $\begin{array} { r } { I _ { S } = \sum _ { S ^ { \prime } \subset S } ( - 1 ) ^ { | S | - | S ^ { \prime } | } \cdot v ( { \pmb x } _ { S ^ { \prime } } ) , \phi ( { \pmb x } _ { \emptyset } ) = v ( { \pmb x } _ { \emptyset } ) } \end{array}$ the output of the logical model $\phi ( \cdot )$ can always match the DNN's output score $v ( \cdot )$

$$
\forall T \subseteq N , v ( { \pmb x } _ { T } ) = \phi ( { \pmb x } _ { T } )\tag{2}
$$

In addition, the sparsity property also provides a theoretical guarantee for the faithfulness of interaction-based explanation. The sparsity property shows that a DNN only encodes a sparse set of interactions with salient effects. That is, only a small subset of all $2 ^ { n }$ interactions in Theorem 3.1, termed salient interactions, have a significant impact on the logical model's output. In contrast, the majority of interactions have negligible effects and are considered noise patterns. The sparsity property of AND interactions has been proven by Ren et al. (2024a). We follow Li & Zhang (2023) to extract AND-OR interactions² from input variables. Such a technique has proven effective in pursuing higher sparsity of interactions, supported by both theoretical proofs (Li & Zhang, 2023) and extensive empirical validation (Ren et al., 2023a; Zhou et al., 2024).

## 3.2. Verifying the Faithfulness of Considering Interactions as Inference Patterns Used by LLMs

Before utilizing the interaction framework to analyze the prompt sensitivity of LLMs, we need to theoretically prove and experimentally validate the faithfulness of using interactions to explain LLMs. In theory, Theorem 3.1 guarantees that the surrogate logical model's output $\phi ( \cdot )$ can always match the LLM's output $v ( \cdot )$ for all $2 ^ { n }$ masked samples. Since the logical model's output $\phi ( \cdot )$ is composed entirely of the sum of all interaction effects as defined in Eq. (1), we can consider interactions as the detailed inference patterns that constitute the LLM's internal logic

![](images/1bab46e9b95303a63d9b601471e5e021b804248074194f62643cef946d963b3b.jpg)

(b)  
![](images/abb6f51a339ef4728bd6f459b110f293d21c55e996c994dbf073e905cfb91160.jpg)

![](images/3b153ca8ba6ec6c9f8f524b1ad6584bd45c6b03a0060cb063599432547544b7b.jpg)

Figure 2. (a) Verifying the sparsity of interactions. We show absolute values of normalized interactions in a descending order. LLMs all encode a small number of salient interactions, while most of the interaction effects are negligible. (b) Verifying the quality of universal matching for any $2 ^ { n }$ masked inputs. The red line plots outputs of the LLM in an ascending order.  
![](images/de92911459c6083d1ca43cd59eacd27053a81bc69ff195c2161c3e79577cc15f.jpg)

(b)  
![](images/7b6424e0905ea62058d3521e116ca7b0c1180c5155e27352da463bcade652927.jpg)  
Figure 3. (a) Case studies of interaction-level analysis. The plots compare the interaction effects encoded by the LLM for the same input x under two semantically equivalent prompt templates, T (blue) and T (red). The bottom row shows examples where the LLM's output remains the same; the top row shows examples where it changes. Results show that interaction effects are highly unstable, even when the output of the LLM remains the same. (b) The distribution of salient interactions types of various LLMs.

Therefore, we can evaluate the prompt sensitivity of LLMs by measuring the instability of interaction patterns.

In practice, we conduct experiments to verify whether LLMs encode sparse interactions. Consider the multiple choice question (MCQ) in Figure 1 as an example. Let $Q$ denote the set of all words in the question, $e . g . , Q = \{ W h i c h ,$ one, of, these, animals, does, NOT, lay, eggs}, M denote the set of all words in the options, e.g., M = {chickens, dogs, frogs, turtles}, and T denote the set of all words in the prompt template, e.g., T = {Answers:, A., B., C., D., Answer: }. Specifically, we use words³ in $Q \cup M$ as input variables and compute the interaction effect $I _ { S }$ of all interactions $S \subseteq Q \cup M$ . Meanwhile, we treat words in prompt template $T$ as background context. We follow Li & Zhang (2023) to extract AND and OR interactions. Thus, given an input $\pmb { x } = Q \cup M$ with n words, we can obtain $2 ^ { n + 1 }$ interactions, including $2 ^ { n }$ AND interactions and $2 ^ { n }$ OR interactions. For each interaction effect $I _ { S }$ , we apply min-max normalization to it. Specifically, $\begin{array} { r } { \widetilde { I } _ { S } \triangleq s g n ( I _ { S } ) \cdot \frac { | I _ { S } | - M i n } { M a x - M i n } } \end{array}$ , where Min and Max are the minimum and maximum absolute values of all $2 ^ { n + 1 }$ interaction effects; sgn $\begin{array} { r } { { \bf \nabla } [ { \bf { \sigma } } _ { I _ { S } } ) = \frac { { \bf { \nabla } } I _ { S } } { | { \bf { \nabla } } _ { S } | } } \end{array}$ represents the sign of $I _ { S } .$ Figure 2 (a) shows the distribution of $| \widetilde { I } _ { S } |$ . Results4 verify that only a small set of interactions have salient effects, while most of the interactions have negligible effects and can be considered as noise patterns. Figure 2 (b) compares the LLM's true output $v ( { \pmb x } _ { T } )$ for all $2 ^ { n }$ masked inputs against the logical model using only the most salient interactions. Even when using the top 3% or top 5% of all interactions, the matching error is minimal. This empirically demonstrates that the LLM's output can be faithfully approximated by a small set of salient interactions.

## 3.3. Using Interactions as a Fine-Grained Tool to Analyze the Prompt Sensitivity of LLMs

Based on the above verification, we deploy interactions as a fine-grained analytical tool. Specifically, for an LLM, we visualize the changes in salient interactions for the same input under a pair of prompt templates. As Figure 3 (a) shows, when the LLM's outputs are different, nearly half of the salient interactions reverse their sign (from positive to negative, or vice versa), another 30%-40% change significantly in magnitude, and only a small fraction (10%-20%) remain stable. Strikingly, even when the LLM's output remains the same, the majority (60%-80%) of the salient interactions are still unstable. This offers preliminary evidence that semantically irrelevant alterations to the prompt template can lead to significant changes in the salient interaction patterns.

![](images/362ce2c9f04b36b520ce77205888f8f016bc1f3cfa359a19151e21842ba3880b.jpg)

![](images/e4410da527373dc2d595937a13b202f74b4bfe26f225b5e2f69920e2ef57e021.jpg)  
Figure 4. Prompt sensitivity of all the 50 open-source LLMs in an ascending order.

To systematically analyze this instability beyond individual cases, we classify interactions into three distinct types: 1) Opposite sign: The interaction effect reverses its sign (e.g. from positive to negative). The sign change represents the most severe form of instability, as it can completely reverse the LLM's internal logic. 2) Same sign & different effect: The interaction maintains its positive or negative influence, but its magnitude changes substantially $( \mathrm { i . e . } _ { \cdot }$ , its effect is more than doubled or less than halved). 3) Same sign & similar effect: The interaction's sign and magnitude both remain stable, which represents robust, stable interactions. Figure 3 (b) plots the distribution of three interaction types over all samples, conditioned on whether the LLM's final output changes or remains the same across a pair of prompts. When the LLM generates different outputs, the interaction patterns are highly unstable. Opposite sign interactions account for approximately 50%, meaning nearly half of salient interactions reverse the sign of their effect. The truly stable Same sign & similar effect interactions account for a mere 2.7%-6.8%, while the remaining 42.6%-47.4% of interactions, though maintaining their sign, change significantly in magnitude. More alarmingly, even when the final output of the LLM remains the same, the instability of interactions still exists. While the situation improves, the majority of interactions still fall into the two unstable categories..

The results demonstrate that output-level analysis is insufficient to capture the unreliable internal patterns of LLMs. Conversely, our interaction-based analysis offers a finegrained lens to uncover latent instability of LLMs, offering a new analytical tool for quantifying the ratio of stable and unstable interactions on a per-sample basis for any LLM.

## 4. Evaluating and Analyzing the Prompt Sensitivity of LLMs

## 4.1. Evaluating Interaction-Based Prompt Sensitivity

We propose an interaction-based metric to measure the prompt sensitivity of LLMs. Given an MCQ dataset $\mathcal { D } ,$ for any input $\mathbf { \pmb { x } } \in \mathcal { D } ,$ it is composed of the question Q and the options M, i.e., $\pmb { x } = Q \cup M$ . Given a prompt template T, we make minor changes to T and obtain a modified prompt template î. By applying the pair of prompt templates T and î to the same input $Q \cup M$ , we can construct two similar prompts. The LLM is supposed to extract similar salient interactions from $Q \cup M$ when processing these two prompts because T and Î have the same semantic meaning. Thus, let $\Omega _ { \mathrm { s a l i e n t } } ( { \pmb x } | T ) = \{ S \in \Omega ( { \pmb x } ) \ | \ | \widetilde { I } _ { S } ( { \pmb x } | T ) | > \tau \}$ represent the set of salient interactions extracted from x given the prompt template T. Here τ is a threshold used to distinguish salient interactions from noise patterns. In the main paper, we set τ as 0.1 for all experiments. Please see $\mathsf { A p - }$ pendix F.8 for hyperparameter experiments of τ. Similarly, let $\Omega _ { \mathrm { s a l i e n t } } ( { \pmb x } | \hat { T } ) = \{ S \in \Omega ( { \pmb x } ) \ | \ | \widetilde { I } _ { S } ( { \pmb x } | \hat { T } ) | > \tau \}$ represent the set of salient interactions extracted from x given the prompt template . Therefore, on the dataset D, we define the LLM's Interaction-based Prompt Sensitivity as IPS.

$$
I P S \triangleq \mathbb { E } _ { x } \left[ \mathbb { E } _ { T , \hat { T } } \left[ \frac { 1 } { | \Omega _ { \mathrm { u n i o n } } | } \sum _ { S \in \Omega _ { \mathrm { u n i o n } } } \frac { | \widetilde { \mathcal { T } } _ { S } ( { \pmb x } | T ) - \widetilde { \mathcal { T } } _ { S } ( { \pmb x } | \hat { T } ) | } { | \widetilde { I } _ { S } ( { \pmb x } | T ) | + | \widetilde { I } _ { S } ( { \pmb x } | \hat { T } ) | } \right] \right] ,\tag{3}
$$

where $\Omega _ { \mathrm { u n i o n } } = \Omega _ { \mathrm { s a l i e n t } } ( { \pmb x } | T ) \cup \Omega _ { \mathrm { s a l i e n t } } ( { \pmb x } | \hat { T } )$ is a unified set by taking the union of the two salient sets $\Omega _ { \mathrm { s a l i e n t } } ( { \pmb x } | T )$ and $\begin{array} { r } { \Omega _ { \mathrm { s a l i e n t } } ( { \pmb x } | \hat { T } ) ; } \end{array}$ the outer expectation, $\mathbb { E } _ { \pmb { x } } .$ , represents an averaging over all inputs $\mathbf { \pmb { x } } \in \mathcal { D } ,$ , and the inner expectation, $\mathbb { E } _ { T , \hat { T } } ;$ represents an averaging over all pairs of prompt templates (T, T). This metric evaluates prompt sensitivity of LLMs by calculating the symmetric mean absolute percentage error of salient interactions over all samples in the dataset D.

Models and Datasets. We conduct experiments on 50 opensource LLMs from 6 model families. This diverse set includes 10 LLMs from the Llama family: Llama-2, Llama-3, Llama-MoE (Touvron et al., 2023; Grattafiori et al., 2024; Zhu et al., 2024b; Qu et al., 2024); 4 LLMs from the Mistral family: Mistral, Mixtral (Jiang et al., 2023; Jiang et al., 2024); 25 LLMs from the Qwen family: Qwen2, Qwen2.5, Qwen3, Qwen1.5-MoE, Qwen3-30B-A3B (Yang et al., 2024; 2025); 9 LLMs from the OLMo family: OLMo, OLMo-2, OLMoE (Groeneveld et al., 2024; OLMo et al., 2025; Muennighoff et al., 2024); 2 LLMs from the InternLM family: InternLM2 (Cai et al., 2024). We evaluate all LLMs on two widely used MCQ benchmarks: ARC (Clark et al. 2018) and MMLU (Hendrycks et al., 2021). Experiments on open-ended tasks are shown in Section 4.3. For each input, we apply five distinct prompt templates to it. These templates maintain the same core content and differ only in minor formatting details, such as letter case and separators. For masking words in the input sentences, we follow the approach of Cheng et al. (2025) and utilize a certain [MASK] token for each LLM. A comprehensive list of all LLMs, along with the specific prompt templates, mask tokens used in experiments is provided in Appendix E.

We apply the IPS metric to evaluate the prompt sensitivity of 50 open-source LLMs. As shown in Figure 4 (a), we observe a wide variance in IPS, ranging from 1.268 (Qwen2.5-72B-Instruct) to 1.752 (Mistral-7B-v0.3). However, no LLM series or families achieve a complete victory. The distribution indicates that an LLM's prompt sensitivity is influenced by multiple underlying factors. Identifying these factors is of great significance for the robustness research of LLMs.

To validate the reliability of the IPS metric, we investigate its correlation with the traditional metrics output consistency, which is defined as the proportion of samples where the LLM generates identical predictions across different prompt templates. Figure 4 (b) shows that the IPS score exhibits a negative correlation with output consistency. This result confirms that IPS aligns well with the output consistency while offering a more fine-grained perspective to quantify prompt sensitivity beyond output matching.

Robustness to the threshold τ. In the main paper, we set the threshold τ as 0.1 for all experiments. To verify that our findings are robust to the selection of τ, we compute interactions across 16 distinct threshold values, ranging from 0.05 to 0.20 with a step size of 0.01. We evaluate the consistency of IPS rankings across these thresholds, observing an average Spearman's rank correlation of 0.9905 and an average Pearson correlation of 0.9957 (Appendix F.8.1). These nearperfect correlations indicate that the relative ranking among LLMs remains highly stable. More crucially, we conduct all the experiment with τ = 0.05 and τ = 0.15. As detailed in Appendix F.8.2, the main conclusions remain same as τ = 0.1, proving that our findings are robust to different τ.

## 4.2. Analyzing the Factors Impacting Prompt Sensitivity

In this section, we investigate four factors that might influence the prompt sensitivity of LLMs, including (1) supervised fine-tuning, (2) model scales, (3) model architectures,

![](images/90287e5fa6a304417781651775bb50c90a95faee6e87e63ec2f67be229367bdb.jpg)  
Figure 5. A comparison of the prompt sensitivity between instruct/chat models and base models. Results show that instruct/chat models are less sensitive than corresponding base models.

![](images/becae1373649b4c1a3638a216f0f3b393f8ab1e04b9a4e549e7fb4b5133fd9f3.jpg)  
Figure 6. A comparison of prompt sensitivity across different model scales. As the model scale increases, the prompt sensitivity within a model series systematically decreases.

and (4) prompting methods.

Factor 1: instruct/chat models vs. base models. Supervised fine-tuning (e.g., instruction tuning) is now a common practice to align base models with human preferences for certain tasks, yielding models often referred to as instruct or chat models (for different tasks or purposes). Therefore, we investigate the impact of supervised fine-tuning by comparing the prompt sensitivity of instruct/chat models with base models. Results⁵ on the ARC dataset in Figure 5 show that almost all instruct/chat models exhibit lower prompt sensitivity than their corresponding base models. This demonstrates that supervised fine-tuning enables the LLM to encode more stable interactions. A valid explanation is that base models are pre-trained on unstructured and raw texts, but instruct/chat models are further fine-tuned on instruction-response datasets or dialogue datasets. Thus, instruct/chat models can precisely understand the function of prompt templates and focus on the task-relevant inputs.

Factor 2: model scales. We investigate the relationship between the model scale (i.e., the number of parameters) and the prompt sensitivity. Results⁵ on the ARC dataset in Figure 6 show that within the same model series, as the model scale increases, the overall prompt sensitivity systematically decreases. This demonstrates that larger LLMs encode more stable interactions, making them less susceptible to superficial changes in the prompt template.

![](images/5eacaaa308eea1962bd0ce5029b944d5477caae0f11e53133dfd5565f0b7ff61.jpg)  
Figure 7. A Comparison of prompt sensitivity between MoE models and dense models. Generally, MoE models tend to be more sensitive than dense models in the same model family.

![](images/895da15912d9966d2e5bba35d6fef8eb7871bda7314ce0347f1ff371c5a0ab34.jpg)  
Figure 8. A comparison of prompt sensitivity between 0-shot learning and few-shot learning. The drop in prompt sensitivity is substantial from 0-shot to 1-shot.

Factor 3: dense models vs. MoE models. MoE models scale up model capacity with minimal computational cost by dynamically activating different subsets of “expert"subnetworks (Cai et al., 2025). In contrast, all parameters in dense models participate in every computation. We aim to investigate the impact of model architectures by comparing the prompt sensitivity of dense models with MoE models. Results³ on the ARC dataset in Figure 7 show that in model families including Llama-2, Llama-3, Qwen and Olmo, all MoE models exhibit higher prompt sensitivity than dense models. This suggests that MoE models encode more unstable interactions. A potential confounding factor is the number of active parameters. Given that smaller models are more sensitive (Factor 2), one might attribute MoE instability to lower active parameters. To mitigate the influence of active parameters, we conduct controlled variable analysis Results (Table 7, Appendix I) confirms that MoE models remain more sensitive than dense models even with similar active parameters. We attribute this to the dynamic routing mechanism. For different prompt templates, the gating network may route the input to different experts so that it is processed by different sub-networks, leading to different interactions encoded and thus higher sensitivity. In conclusion, while MoE models achieve impressive performance with reduced computational overhead, this benefit comes at the cost of weaker stability.

Factor 4: few-shot learning vs. 0-shot learning. Fewshot learning is utilized to improve LLMs’ performance by providing in-context examples to better specify the task (Wang et al., 2020). We investigate the impact of prompting methods on prompt sensitivity by comparing 0-shot learning with few-shot learning6. Results⁵ on the ARC dataset in Figure 8 show that incorporating in-context examples leads to a significant reduction in prompt sensitivity across all tested LLMs. For most of the LLMs, the most substantial drop occurs when moving from 0-shot to 1-shot, while adding more in-context examples yields slower reductions. This suggests that even a single example is sufficient to establish the LLM's understanding of the task, leading it to ignore superficial template variations and focus on the core input.

![](images/f051b083c731bfcf9d68b44f545a927a6f21109d9a4fc4d5f4b9ae655b57ab5c.jpg)  
Figure 9. Prompt sensitivity of LLMs for different order types.

## 4.3. Explore the Underlying Mechanisms of Improved Stability for All Factors

In this section, we aim to explore whether there exists a common reason to explain the underlying mechanisms by which the four aforementioned factors reduce the prompt sensitivity of LLMs. Specifically, we analyze the prompt sensitivity of different types of interactions, so as to reveal the source of the LLM's prompt sensitivity. To this end, we analyze the sensitivity of interactions with different complexities, which are defined as the orders of interactions. The order of an interaction S is defined as the number of input variables involved, i.e., |S|. An interaction with high order indicates an intricate relationship including many input variables, while an interaction with low order represents a simple relationship including few input variables. We further define three types of prompt sensitivity metrics corresponding to different types of interaction orders. Specifically, we partition interactions in $\Omega _ { \mathrm { u n i o n } }$ in Eq. (3) into three distinct groups based on their orders: low-order mid-order, and high-order. Given an input x with n words, $\begin{array} { r } { \Omega _ { \mathrm { u n i o n } } ^ { l o w } \triangleq \{ S \in \Omega _ { \mathrm { u n i o n } } \ | \ 1 \leq | S | \leq \lfloor \frac { 1 } { 3 } n \rfloor \} , \Omega _ { \mathrm { u n i o n } } ^ { m i d } \triangleq \{ S \in \Omega _ { \mathrm { u n i o n } } } \end{array}$ $\begin{array} { r } { \lfloor \frac { 1 } { 3 } n \rfloor < | S | < \lfloor \frac { 2 } { 3 } n \rfloor \} , \Omega _ { \mathrm { u n i o n } } ^ { h i g h } \triangleq \{ S \in \Omega _ { \mathrm { u n i o n } } \ | \ \lfloor \frac { 2 } { 3 } n \rfloor < | S | \leq n \} } \end{array}$ Then we calculate prompt sensitivity for low-order, midorder, and high-order interactions, as $I P S ^ { l o w } , I P S ^ { m i d } , I P S ^ { h i g h }$

Figure 9 presents the prompt sensitivity of different order types on the ARC dataset. Results4 show that the prompt sensitivity of low-order interactions is the lowest, followed by mid-order, while high-order interactions exhibit the highest prompt sensitivity. This indicates that low-order interactions encoded by LLMs are relatively stable when faced with subtle changes to prompt templates, i.e., simple interaction patterns are more robust. Conversely, the high sensitivity of high-order interactions reveals that the LLMs' internal representation of complex patterns is highly unstable.

![](images/a15f6cfd7cfd8f37fd69df04f1a08eb457077c010f97e5aadb0b78c62d885d55.jpg)

![](images/b240853bdc0f8f0f6181e09d667d52b96f9ef506089e51737c7d66b15d548b2a.jpg)

![](images/d9677df1341a7fd359d5da7a7d71a42f43e81a9e0dd54365a0090530778419f9.jpg)  
Figure 10. Comparing the relative change in the prompt sensitivity of low-, mid-, and high-order interactions for different factors.

![](images/0b9319d84bfc3b9d348f8e1e8ffb965e99bf8634b396e0905eee4b4cc9f47b69.jpg)  
Figure 11. A comparison of prompt sensitivity of different order types across different model scales.

Explaining why the four factors can reduce prompt sensitivity. Inspired by the results in Figure 9, we now investigate how the four aforementioned factors influence the prompt sensitivity of low-, mid-, and high-order interactions. This order-level analysis aims to reveal the common mechanism by which these factors reduce the LLM's prompt sensitivity. For each factor, we quantify its effect on low-, mid-, and high-order interactions by computing the relative change in IPS between the LLM with the factor and its counterpart without it. Specifically, given $t y p e \in \{ l o w , m i d , h i g h \}$ , the relative change is defined as $\Delta I P S ^ { t y p e } = ( I P S _ { \mathrm { A } } ^ { t y p e } - I P S _ { \mathrm { B } } ^ { t y p e } ) / I P S _ { \mathrm { B } } ^ { t y p e }$ For Factor 1 (fine-tuned vs. base), A and B are fine-tuned and base models, respectively. For Factor 3 (dense vs. MoE), A and B are dense and MoE models, with the final ∆IPstype being the average over all pairs of a specific dense model and a specific MoE model within a model family. For Factor 4 (few-shot vs. 0-shot), A and B are x-shot $( x \in \{ 1 , 2 , 3 \} )$ and 0-shot learning. The relative change metric is unsuitable for Factor 2 (model scales), as scale is a continuous variable. Instead, we directly analyze the trend of IPS values as model parameters increase.

Results⁴ shown in Figure 10 and Figure 11 converge on a common explanation for how the four factors reduce prompt sensitivity. The most significant reduction in prompt sensitivity is consistently observed in low-order interactions. An obvious, though less pronounced, decrease is also seen at the mid-order level. In contrast, the change of the sensitivity of high-order interactions is relatively minimal, remaining at a high level. It indicates that the stability of low-order interactions is critical to the overall robustness of LLMs.

This phenomenon is unexpected. Although results in Figure 9 show that low-order interactions are naturally more robust than other types of interactions, the four factors above still significantly reduce the sensitivity of low-order interactions. Instead, they fail to reduce the sensitivity of highorder interactions, which are inherently the most sensitive. This phenomenon indicates that stable low-order interactions are much easier for LLMs to learn, while it is difficult for LLMs to make high-order interactions more stable.

Robustness on open-ended tasks. To verify the robustness of our findings, we conduct additional experiments on openended generation tasks. We utilize the Dolly-15k dataset (Conover et al., 2023), which contains a diverse range of non-MCQ tasks, including open Q&A, classification, and others. The results of this analysis, detailed in Appendix K, consistently verify our main conclusions drawn from the MCQ experiments. This strongly suggests that the four factors and the underlying mechanisms of prompt sensitivity we have uncovered can be generalized to open-ended questions.

Robustness to more complex prompt perturbations. In this experimental setup, we use the Dolly-15k dataset and introduce more complex prompt perturbations, specifically semantic paraphrases and instruction reordering (detailed in Appendix L.1), to test the robustness of our conclusions. The results in Figures 12, 13, and 14 consistently affirm our conclusions drawn from the template-based experiments. This indicates that our conclusions are robust to more com-

(a) Instruct/Chat Models vs. Base Models

![](images/fd0fabd1f9d3ddb27d419dacfe56c5765cf740451d5346498d804b037cb3d8bd.jpg)

(b) Model Scales  
![](images/597efcce7a963e419049c160c278772f7259a1baa4b3c5b592e327cab9615cc6.jpg)

(c) Dense Models vs. MoE Models  
![](images/07733873f590b7e327a72cb128ba552c4958b7b55723496df908db0e8dbce9b6.jpg)  
Llama 2 Family  
(d) Few-Shot Learning vs. 0-Shot Learning

![](images/052bae4f3e3ccd995ae74816945db941bdbbacc71433f9d8cb9248926533cc90.jpg)  
Figure 12. (a) A comparison of the prompt sensitivity between instruct/chat models and base models. (b) A comparison of prompt sensitivity across different model scales. (c) A comparison of prompt sensitivity between MoE models and dense models. (d) A comparison of prompt sensitivity between 0-shot learning and few-shot learning.

![](images/8aff15a3c4415767d8806f912fb43d46f957c97ad2efaf1485a5acb380c17111.jpg)  
(a) The relative change of Instruct/Chat Model compared to Base Model

![](images/63fdfdde7376c9b50e913b88048073bf2856c206d42e0a67b4bc214d6e32d073.jpg)  
(b) The relative change of Dense Model compared to MoE Model

![](images/b62ce9a0e9b0ccb547bd75c017fe7b88234919ae087cda0c7b578300993601b5.jpg)  
(c) The relative change of Few-shot learning compared to 0-shot learning

![](images/89615c0b39032f332129344950c910d8606d86b9f0a9a05de30d2c0ddc7c6853.jpg)  
Figure 13. Comparing the relative change in the prompt sensitivity of low-, mid-, and high-order interactions for different factors.

![](images/b822027c6a94de9a01419cbf15140168c10ea6b68029305447773c0d08f717a0.jpg)  
Figure 14. A comparison of prompt sensitivity at the order-level across different model scales.

## plex prompt perturbations.

Strategies for reducing the computational cost. To reduce the computational cost of the interaction framework, recent studies (Chen et al., 2024; Cheng et al., 2025) used the following two strategies: (1) Select informative words as input variables while treating uninformative ones (e.g., stop words) as fixed background context. (2) Merge related words into combined phrases as input variables. The specific selection strategies are detailed in Appendix K.2. Results in Appendix K.3 show that using the above two strategies on open-ended tasks yields the same conclusions. In addition to selection strategies, techniques specifically designed for efficiently computing sparse interactions to bypass exhaustive O(2ⁿ) evaluations (Kang et al., 2025; Butler et al., 2026) represent a clear path for reducing computational cost. Further details are provided in Appendix J.

## 5. Conclusion

In this paper, we propose an interaction-based metric to evaluate the prompt sensitivity of LLMs. We discover that employing supervised fine-tuning, increasing model scale, using dense over MoE architectures, and applying few-shot learning all serve to reduce the prompt sensitivity of LLMs. Our findings offer novel insights into both model designs and prompting methods for improving the robustness of LLMs. More crucially, we find that these factors achieve lower sensitivity primarily by reducing the sensitivity of low-order interactions, while the prompt sensitivity of highorder interactions remains at a relatively high level. In future studies, new training methods could be designed to increase the LLM's reliance on stable low-order interactions or, alternatively, to reduce the instability of high-order interactions.

## Acknowledgements

This work is partially supported by the Shanghai Science and Technology Commission (No. 25511102900), the National Nature Science Foundation of China (No.62376199,62576249), and the Shanghai Municipal Education Commission (No. 24CGA20).

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Alzahrani, N., Alyahya, H., Alnumay, Y., AlRashed, S., Alsubaie, S., Almushayqih, Y., Mirza, F., Alotaibi, N., Al-Twairesh, N., Alowisheq, A., Bari, M. S., and Khan, H. When benchmarks are targets: Revealing the sensitivity of large language model leaderboards. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13787–13805, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-1ong.744. URL https: //aclanthology.org/2024.acl-long.744/.

Ancona, M., Oztireli, C., and Gross, M. Explaining deep neural networks with a polynomial time algorithm for Shapley value approximation. In International conference on machine learning, pp. 272–281. PMLR, 2019.

Annepaka, Y. and Pakray, P. Large language models: A survey of their development, capabilities, and applications. Knowledge and Information Systems, pp. 1–56, 2024.

Bird, S. NLTK: the natural language toolkit. In Proceedings of the COLING/ACL 2006 interactive presentation sessions, pp. 69–72, 2006.

Butler, L., Agarwal, A., Kang, J. S., Erginbas, Y. E., Yu, B., and Ramchandran, K. ProxySPEX: Inference-efficient interpretability via sparse feature interactions in LLMs. In The Thirty-ninth Annual Conference on Neural Information ProcessingSystems, 2026. URL https : //openreview.net/forum?id=KI8qan2EA7.

Cai, W., Jiang, J., Wang, F., Tang, J., Kim, S., and Huang, J. A survey on mixture of experts in large language models. IEEE Transactions on Knowledge and Data Engineering, 2025.

Cai, Z., Cao, M., Chen, H., Chen, K., Chen, K., Chen, X.,

Chen, X., Chen, Z., Chen, Z., Chu, P., et al. InternLM2 technical report. arXiv preprint arXiv:2403.17297, 2024.

Cao, B., Cai, D., Zhang, Z., Zou, Y., and Lam, W. On the worst prompt performance of large language models. Advances in Neural Information Processing Systems, 37: 69022–69042, 2024.

Chang, Y., Wang, X., Wang, J., Wu, Y., Yang, L., Zhu, K., Chen, H., Yi, X., Wang, C., Wang, Y., et al. A survey on evaluation of large language models. ACM transactions on intelligent systems and technology, 15(3):1–45, 2024.

Chatterjee, A., Renduchintala, H. S. V. N. S. K., Bhatia, S., and Chakraborty, T. POSIX: A prompt sensitivity index for large language models. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 14550–14565, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.852. URL https://aclanthology.org/2024.findin gs-emnlp.852/.

Chen, L., Lou, S., Huang, B., and Zhang, Q. Defining and extracting generalizable interaction primitives from DNNs. In The Twelfth International Conference on Learning Representations, 2024. URL ht tp s : //openreview.net/forum?id=OCqyFVFNeF.

Cheng, L., Zhang, J., Ren, Q., and Zhang, Q. Revisiting generalization power of a DNN in terms of symbolic interactions. arXiv preprint arXiv:2502.10162, 2025.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A.. Schoenick, C., and Tafjord, O. Think you have solved question answering? try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Conover, M., Hayes, M., Mathur, A., Xie, J., Wan, J., Shah, S., Ghodsi, A., Wendell, P., Zaharia, M., and Xin, R. Free dolly: Introducing the world's first truly open instructiontuned llm,2023.URL https://www.databricks .com/blog/2023/04/12/dolly-first-ope n-commercially-viable-instruction-tun ed-11m.

Covert, I., Lundberg, S., and Lee, S.-I. Explaining by removing: A unified framework for model explanation. Journal of Machine Learning Research, 22(209):1–90, 2021.

Dabkowski, P. and Gal, Y. Real time image saliency for black box classifiers. Advances in neural information processing systems, 30, 2017.

Deng, H., Zou, N., Du, M., Chen, W., Feng, G., Yang, Z., Li, Z., and Zhang, Q. Unifying fourteen post-hoc attribution methods with Taylor interactions. IEEE Transactions on

Pattern Analysis and Machine Intelligence, 46(7):4625– 4640, 2024.

Errica, F., Sanvito, D., Siracusano, G., and Bifulco, R. What did I do wrong? quantifying LLMs' sensitivity and consistency to prompt engineering. In Chiruzzo, L., Ritter, A., and Wang, L. (eds.), Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 1543–1558, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025.naacl-1ong.73. URL https : // aclanthology.org/2025.naacl-long.73/.

Fong, R., Patrick, M., and Vedaldi, A. Understanding deep networks via extremal perturbations and smooth masks. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2950–2958, 2019.

Fong, R. C. and Vedaldi, A. Interpretable explanations of black boxes by meaningful perturbation. In Proceedings of the IEEE international conference on computer vision, pp. 3429–3437, 2017.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Groeneveld, D., Beltagy, I., Walsh, E., Bhagia, A., Kinney, R., Tafjord, O., Jha, A., Ivison, H., Magnusson, I., Wang, Y., et al. OLMo: Accelerating the science of language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15789–15809, 2024.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M. Song, D., and Steinhardt, J. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021. URL https : //open review.net/forum?id=d7KBjmI3GmQ.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Singh Chaplot, D., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Renard Lavaud, L., Lachaux, M.-A., Stock, P., Le Scao, T., Lavril, T., Wang, T., Lacroix, T., and El Sayed, W. Mistral 7B. arXiv e-prints, art. arXiv:2310.06825, October 2023. doi: 10.48550/arXiv.2310.06825.

Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. 1., Hanna, E. B., Bressand, F., et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Kang, J. S., Butler, L., Agarwal, A., Erginbas, Y. E., Pedarsani, R., Yu, B., and Ramchandran, K. SPEX: Scaling feature interaction explanations for LLMs. In Fortysecond International Conference on Machine Learning, 2025. URL https://openreview.net/forum ?id=pRlKbAwczl.

Li, M. and Zhang, Q. Does a neural network really encode symbolic concepts? In International conference on machine learning, pp. 20452–20469. PMLR, 2023.

Li, Q., Ruan, J., Wu, F., Chen, Y., Wei, Z., and Shen, W. A unified approach to interpreting self-supervised pretraining methods for 3D point clouds via interactions. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 27315–27324, 2025. doi: 10.1109/CVPR52734.2025.02544.

Lu, S., Schuff, H., and Gurevych, I. How are prompts different in terms of sensitivity? In Duh, K., Gomez, H., and Bethard, S. (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 5833–5856, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-1ong.325. URL https://aclanthology.org/2024.naacl-1 ong.325/.

Lundberg, S. M. and Lee, S.-I. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30, 2017.

Muennighoff, N., Soldaini, L., Groeneveld, D., Lo, K., Morrison, J., Min, S., Shi, W., Walsh, P., Tafjord, O., Lambert, N., et al. OLMoE: Open mixture-of-experts language models. arXiv preprint arXiv:2409.02060, 2024.

OLMo, T., Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., Arora, S., Bhagia, A., Gu, Y., Huang, S., Jordan, M., Lambert, N., Schwenk, D., Tafjord, O., Anderson, T., Atkinson, D., Brahman, F., Clark, C., Dasigi, P., Dziri, N., Ettinger, A., Guerquin, M., Heineman, D., Ivison, H., Koh, P. W., Liu, J., Malik, S., Merrill, W., Miranda, L. J. V., Morrison, J., Murray, T., Nam, C., Poznanski, J., Pyatkin, V., Rangapur, A., Schmitz, M., Skjonsberg, S., Wadden, D., Wilhelm, C., Wilson, M., Zettlemoyer, L., Farhadi, A., Smith, N. A., and Hajishirzi, H. 2 olmo 2 furious,2025. URL https://arxiv.org/abs/25 01.00656.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Qu, X., Dong, D., Hu, X., Zhu, T., Sun, W., and Cheng, Y. Llama-MoE v2: Exploring sparsity of Llama from perspective of mixture-of-experts with post-training. arXiv preprint arXiv:2411.15708, 2024.

Razavi, A., Soltangheis, M., Arabzadeh, N., Salamat, S., Zihayat, M., and Bagheri, E. Benchmarking prompt sensitivity in large language models. In European Conference on Information Retrieval, pp. 303–313. Springer, 2025.

Ren, J., Li, M., Chen, Q., Deng, H., and Zhang, Q. Defining and quantifying the emergence of sparse concepts in DNNs. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20280– 20289, 2023a.

Ren, J., Zheng, X., Liu, J., Lizarraga, A., Wu, Y. N., Lin, L. and Zhang, Q. Monitoring primitive interactions during the training of DNNs. Proceedings of the AAAI Conference on Artificial Intelligence, 39(19):20183–20191, Apr. 2025. doi: 10.1609/aaai.v39i19.34223. URL https://ojs.aaai.org/index.php/AAAI/ article/view/34223.

Ren, Q., Deng, H., Chen, Y., Lou, S., and Zhang, Q. Bayesian neural networks avoid encoding complex and perturbation-sensitive concepts. In International Conference on Machine Learning, pp. 28889–28913. PMLR, 2023b.

Ren, Q., Gao, J., Shen, W., and Zhang, Q. Where we have arrived in proving the emergence of sparse interaction primitives in DNNs. In The Twelfth International Conference on Learning Representations, 2024a. URL https : //openreview.net/forum?id=3pWSL8My6B.

Ren, Q., Zhang, J., Xu, Y., Xin, Y., Liu, D., and Zhang, Q. Towards the dynamics of a DNN learning symbolic interactions. Advances in Neural Information Processing Systems, 37:50653–50688, 2024b.

Sclar, M., Choi, Y., Tsvetkov, Y., and Suhr, A. Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. In The Twelfth International Conference on Learning Representations, 2024. URL https : //openreview.net/forum?id=RIu5lyNXjT.

Shen, W., Wei, Z., Ren, Q., Zhang, B., Huang, S., Fan, J., and Zhang, Q. Interpretable rotation-equivariant quaternion neural networks for 3D point cloud processing. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(5):3290–3304, 2024. doi: 10.1109/TPAMI.20 23.3346383.

Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A.

Garriga-Alonso, A., et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on machine learning research, 2023.

Sturmfels, P., Lundberg, S., and Lee, S.-I. Visualizing the impact of feature attribution baselines. Distill, 5(1):e22, 2020.

Sun, J., Shaib, C., and Wallace, B. C. Evaluating the zeroshot robustness of instruction-tuned language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview .net/forum?id=g9diuvxN6D.

Sundararajan, M., Taly, A., and Yan, Q. Axiomatic attribution for deep networks. In International conference on machine learning, pp. 3319–3328. PMLR, 2017.

Tenney, I., Wexler, J., Bastings, J., Bolukbasi, T., Coenen, A., Gehrmann, S., Jiang, E., Pushkarna, M., Radebaugh, C., Reif, E., et al. The language interpretability tool: Extensible, interactive visualizations and analysis for nlp models. In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations, pp. 107–118, 2020.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Wang, Y., Yao, Q., Kwok, J. T., and Ni, L. M. Generalizing from a few examples: A survey on few-shot learning. ACM computing surveys (csur), 53(3):1–34, 2020.

Wen, L., Zheng, L., Li, H., Sun, L., Wei, Z., and Shen, W. Interpreting arithmetic reasoning in large language models using game-theoretic interactions. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net /forum?id=tRvzEL64dY.

Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C., Liu, D., Huang, F., Dong, G., Wei, H., Lin, H., Tang, J., Wang, J., Yang, J., Tu, J., Zhang, J., Ma, J., Yang, J., Xu, J., Zhou, J., Bai, J., He, J., Lin, J., Dang, K., Lu, K., Chen, K., Yang, K., Li, M., Xue, M., Ni, N., Zhang, P., Wang, P., Peng, R., Men, R., Gao, R., Lin, R., Wang, S., Bai, S., Tan, S., Zhu, T., Li, T., Liu, T., Ge, W., Deng, X., Zhou, X., Ren, X., Zhang, X., Wei, X., Ren, X., Liu, X., Fan, Y., Yao, Y., Zhang, Y., Wan, Y., Chu, Y., Liu, Y., Cui, Z., Zhang, Z., Guo, Z., and Fan, Z. Qwen2 technical report, 2024. URL https://arxiv.org/abs/2407.10671.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Zhang, Q., Wang, X., Cao, R., Wu, Y. N., Shi, F., and Zhu, S.-C. Extraction of an explanatory graph to interpret a CNN. IEEE transactions on pattern analysis and machine intelligence, 43(11):3863–3877, 2020.

Zhou, H., Zhang, H., Deng, H., Liu, D., Shen, W., Chan, S.-H., and Zhang, Q. Explaining generalization power of a DNN using interactive concepts. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 17105–17113, 2024.

Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D., and Hou, L. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

Zhu, K., Wang, J., Zhou, J., Wang, Z., Chen, H., Wang, Y., Yang, L., Ye, W., Zhang, Y., Gong, N., and Xie, X. Promptrobust: Towards evaluating the robustness of large language models on adversarial prompts. In Proceedings of the 1st ACM Workshop on Large AI Systems and Models with Privacy and Safety Analysis, LAMPS '24, pp. 57–68, New York, NY, USA, 2024a. Association for Computing Machinery. ISBN 9798400712098. doi: 10.1145/3689217.3690621. URL https: //doi.0rg/10.1145/3689217.3690621.

Zhu, T., Qu, X., Dong, D., Ruan, J., Tong, J., He, C., and Cheng, Y. Llama-MoE: Building mixture-of-experts from Llama with continual pre-training. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 15913–15923, 2024b.

Zhuo, J., Zhang, S., Fang, X., Duan, H., Lin, D., and Chen, K. ProSA: Assessing and understanding the prompt sensitivity of LLMs. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 1950–1976, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findi ngs-emnlp.108. URL https://aclanthology.o rg/2024.findings-emnlp.108/.

## A. Masking Strategies of Input Variables

In attribution method research, it is common to employ a specific token or embedding to mask the input variables of a deep neural network (DNN) (Lundberg & Lee, 2017; Ancona et al., 2019; Fong et al., 2019) and use changes in network outputs on the masked samples to estimate attributions of different input variables. The selection of a masking approach is complex, as each method has its weakness. For example, replacing input variables with the mean baseline value (the average of all samples) or the zero baseline value can introduce out-of-distribution signals, thereby providing the model with artificial information, such as uniform grey or black dots in an image (Dabkowski & Gal, 2017; Ancona et al., 2019; Sundararajan et al., 2017). Additionally, blurring image pixels using a Gaussian kernel (Fong & Vedaldi, 2017; Fong et al., 2019) as the masked state removes high-frequency signals but fails to eliminate low-frequency signals (Covert et al., 2021; Sturmfels et al., 2020).

Given these challenges, we adopt a token replacement strategy, which is standard for the text domain. This involves substituting the target input word with a dedicated [MASK] token at the embedding level. For example, to mask the word "green" in the input "He is a green hand," we would provide the LLM with the modified input "He is a [MASK] hand."This approach effectively nullifies the specific semantic contribution of the target word without introducing out-of-distribution artifacts, ensuring a clean and consistent baseline for our interaction analysis. For the specific [MASK] token for each LLM, please refer to Section 5 for details

## B. Proof of Theorem

## B.1. Proof of Universal Matching Property

In the main body of the paper, for the sake of simplicity and clarity, we introduced the Universal Matching Property (Theorem 1) primarily through the lens of AND interactions. However, our empirical analysis and the underlying theoretical framework are built upon a more comprehensive AND-OR interaction framework. This extended framework, which incorporates both AND and OR interaction patterns, also adheres to the Universal Matching Property.

In this section, we provide the formal proof for the Universal Matching Property of the complete AND-OR interaction framework. This proof is more general and naturally subsumes the proof for the AND interaction framework presented as Theorem 1 in the main text. We will demonstrate that the output of the surrogate logical model, which is the sum of all AND-OR interaction effects, can perfectly match the output of the Deep Neural Network (DNN) for any masked sample.

The surrogate logical model $\phi ( \cdot )$ is defined as follows:

$$
\phi ( \pmb { x } _ { T } ) \triangleq \phi ( \pmb { x } _ { \emptyset } ) + \sum _ { S \subseteq N , S \neq \emptyset } \mathbb { 1 } _ { \mathrm { A N D } } ( S \mid \pmb { x } _ { T } ) \cdot \boldsymbol { I } _ { S } ^ { \mathrm { A N D } } + \sum _ { S \subseteq N , S \neq \emptyset } \mathbb { 1 } _ { \mathrm { O R } } ( S \mid \pmb { x } _ { T } ) \cdot \boldsymbol { I } _ { S } ^ { \mathrm { O R } } ,\tag{4}
$$

where the AND trigger function $\mathbb { 1 } _ { \mathrm { A N D } } ( S \mid \pmb { x } _ { T } ) \in \{ 0 , 1 \}$ represents an AND relationship between input variables in $S ,$ which can also be termed AND interaction pattern; the OR trigger function $\mathbb { 1 } _ { \mathrm { O R } } ( S \mid \pmb { x } _ { T } ) \in \{ 0 , 1 \}$ represents an OR relationship between input variables in S, which can also be termed OR interaction pattern. The scalar weight $I _ { S } ^ { \mathrm { A N D } }$ quantifies the effect of an AND relationship, which can also be termed AND interaction effect; the scalar weight $I _ { S } ^ { \mathrm { O R } }$ quantifies the effect of an OR relationship, which can also be termed OR interaction effect. An AND relationship is activated only by the joint presence of all input variables in the set S, i.e., all input variables in $S$ are not masked. For instance, given the input sentence $\textstyle { \pmb { x } } = { \overset {  } { \hbar } } H e$ is a green $h a n d , "$ the co-occurrence of the input variables in the set $S = \{ g r e e n , h a n d \}$ contributes a numerical effect $I _ { S } ^ { \mathrm { A N D } }$ that pushes the surrogate logical model's inference towards the semantic meaning of “beginner." If an AND interaction S is triggered, $i . e . , \mathbb { 1 } _ { \mathrm { A N D } } ( S \mid \pmb { x } _ { T } ) = 1$ , the corresponding interaction effect $I _ { S } ^ { \mathrm { A N D } }$ is added to the output of the logical model. Otherwise, if any word in S is masked and the AND interaction is not triggered, $i . e . , \mathbb { 1 } _ { \mathrm { A N D } } ( S \mid \pmb { x } _ { T } ) = 0$ , the interaction effect $I _ { S } ^ { \mathrm { A N D } }$ is not added to the output of the logical model. An OR relationship is activated by the presence of any of all input variables in the set S, i.e., any input variables in S are not masked. For instance given the input sentence $\ b { x } = \ \overset {  } { \cdot } T h e$ service was terrible and the food was awful," the presence of any input variables in the set $S = \{ t e r r i b l e , a w f u l \}$ contributes a numerical effect $I _ { S } ^ { \mathrm { O R } }$ that pushes the surrogate logical model's inference towards a negative sentiment classification. If an OR interaction S is triggered $i . e . , \mathbb { 1 } _ { 0 \mathrm { R } } ( S \mid \pmb { x } _ { T } ) = 1$ , the corresponding interaction effect $I _ { S } ^ { \mathrm { O R } }$ is added to the output of the logical model. Otherwise, if all words in $S$ are masked and the OR interaction is not triggered, i.e., $\mathbf { 1 } _ { \mathrm { O R } } ( S \mid \pmb { x } _ { T } ) = 0$ , the interaction effect $I _ { S } ^ { \mathrm { O R } }$ is not added to the output of the logical model. $\scriptstyle { \pmb x } _ { \emptyset }$ represents that all input variables in N are masked.

Definition of universal matching property for AND-OR interactions. When the scalar weights in the surrogate logical model $\phi ( \cdot )$ are set to $\begin{array} { r } { I _ { S } ^ { \mathrm { A N D } } = \sum _ { T \subset S } ( - 1 ) ^ { | S | - | T | } v _ { \mathrm { a n d } } ( \pmb { x } _ { T } ) } \end{array}$ and $\begin{array} { r } { I _ { S } ^ { \mathrm { O R } } = - \sum _ { T \subset S } ( - 1 ) ^ { | S | - | T | } v _ { \mathrm { o r } } ( \pmb { x } _ { N \backslash T } ) } \end{array}$ , the output of $\phi ( \cdot )$ can always match the output score of the DNN $v ( \cdot ) , i . e . , \forall T \subseteq N , v ( { \pmb x } _ { T } ) = \bar { \phi ( { \pmb x } _ { T } ) }$ . Here $v _ { \mathrm { a n d } } ( { \pmb x } _ { T } ) + v _ { \mathrm { o r } } ( { \pmb x } _ { T } ) = v ( { \pmb x } _ { T } )$ 1 We need to prove that given an input sample x, for each masked sample $\{ { \pmb x } _ { T } | T \subseteq N \}$ , the network output score $v ( x _ { T } ) \in \mathbb { R }$ can be well matched by the surrogate logical model $\phi ( { \pmb x } _ { T } )$ . The surrogate logical model $\phi ( { \pmb x } _ { T } )$ uses the sum of AND interactions and OR interactions to accurately explain/match the network output score $v ( x _ { T } )$

$$
\begin{array} { r l } & { \forall T \subseteq N , v ( \pmb { x } _ { T } ) = \phi ( \pmb { x } _ { T } ) . } \\ & { \phi ( \pmb { x } _ { T } ) = \phi ( \pmb { x } _ { \emptyset } ) + \displaystyle \sum _ { S \subseteq N , S \neq \emptyset } \mathbb { 1 } _ { \mathrm { A N D } } ( S \mid \pmb { x } _ { T } ) \cdot I _ { S } ^ { \mathrm { A N D } } + \displaystyle \sum _ { S \subseteq N , S \neq \emptyset } \mathbb { 1 } _ { \mathrm { O R } } ( S \mid \pmb { x } _ { T } ) \cdot I _ { S } ^ { \mathrm { O R } } , } \\ & { \qquad = \underbrace { v ( \pmb { x } _ { \emptyset } ) + \displaystyle \sum _ { S \subseteq T , S \neq \emptyset } I _ { S } ^ { \mathrm { A N D } } } _ { v _ { \mathrm { a n d } } ( \pmb { x } _ { T } ) } + \underbrace { \sum _ { S \subseteq N , S \cap T \neq \emptyset } I _ { S } ^ { \mathrm { O R } } } _ { v _ { \mathrm { o r } } ( \pmb { x } _ { T } ) } } \end{array}\tag{5}
$$

Proof. (1) Universal matching property of AND interactions. For all $2 ^ { n }$ masked samples $\{ \pmb { x } _ { T } \mid T \subseteq N \}$ , what we need to prove is that the output ${ v } _ { \mathrm { a n d } } ( { \pmb x } _ { T } )$ of a DNN can be universally explained by all the interactions in $T \subseteq N , i . e .$ $\begin{array} { r } { \forall S \subseteq T , S \neq \emptyset , v _ { \mathrm { a n d } } ( x _ { T } ) = \sum _ { S \subseteq T , S \neq \emptyset } I _ { S } ^ { \mathrm { A N D } } ( { \boldsymbol x } ) = v ( { \boldsymbol x } _ { \emptyset } ) + \sum _ { S \subseteq T , S \neq \emptyset } I _ { S } ^ { \mathrm { A N D } } . \mathrm { H e r e } , v ( { \boldsymbol x } _ { \emptyset } ) = v _ { \mathrm { a n d } } ( { \boldsymbol x } _ { \emptyset } ) } \end{array}$

According to the definition of the AND interaction, $\begin{array} { r } { I _ { S } ^ { \mathrm { A N D } } ( \pmb { x } ) = \sum _ { L \subset S } ( - 1 ) ^ { | S | - | L | } v _ { \mathrm { a n d } } ( \pmb { x } _ { L } ) } \end{array}$ . To simplify the computation of the sum of AND interactions $\begin{array} { r } { \sum _ { S \subset T , S \neq \emptyset } I _ { S } ^ { \mathrm { A N D } } ( \boldsymbol { x } ) = \sum _ { S \subset T , S \neq \emptyset } \bar { \sum } _ { L \subset S } ( - 1 ) ^ { | S | - | L | } v _ { \mathrm { a n d } } ( \boldsymbol { x } _ { L } ) } \end{array}$ , we exchange the order of summation of the set $L \subseteq S \subseteq T$ and the set $S \supseteq L .$ Given a set of input variables L, we compute all linear combinations of all sets $S$ containing L with respect to the model outputs ${ v } _ { \mathrm { a n d } } ( { \pmb x } _ { S } )$ , i.e., $\begin{array} { r l } { \sum _ { S : L \subset S \subset T } ( - 1 ) ^ { \top | S | - | L | } v _ { \mathrm { a n d } } ( \pmb { x } _ { L } ) } & { { } } \end{array}$ . Then, we compute all summations over the set $L \subseteq T$ as $\begin{array} { r } { \sum _ { S \subseteq T , S \neq \emptyset } I _ { S } ^ { \mathrm { A N D } } ( \boldsymbol { x } ) = \sum _ { L \subseteq T } \sum _ { S : L \subseteq S \subseteq T } ( - 1 ) ^ { | S | - | L | } v _ { \mathrm { a n d } } ( \boldsymbol { x } _ { L } ) } \end{array}$ . Then, we can compute different cases of $L \subseteq S \subseteq T$ as follows:

$$
\begin{array} { r } { L = T = S , \sum _ { S : L \subset S \subset T } ( - 1 ) ^ { | S | - | L | } v _ { \mathrm { a n d } } ( \pmb x _ { L } ) = ( - 1 ) ^ { | T | - | T | } v _ { \mathrm { a n d } } ( \pmb x _ { L } ) = v _ { \mathrm { a n d } } ( \pmb x _ { L } ) . } \end{array}
$$

(2) When $L \subseteq S \subseteq T , L \neq T $ , let us consider the linear combinations of all sets S with number |S| for the model output ${ v } _ { \mathrm { a n d } } ( { \pmb x } _ { L } )$ , respectively. Let $m : = | S | - | L | , ( 0 \leq m \leq | T | - | L | )$ , then there are a total of $C _ { | T | - | L | } ^ { m }$ combinations of all sets S of order |S|. Given $L ,$ accumulating the model outputs ${ v } _ { \mathrm { a n d } } ( { \pmb x } _ { L } )$ corresponding to all $S \supseteq L $ we can get $\begin{array} { r } { \sum _ { S : L \subseteq S \subseteq T } ( - 1 ) ^ { | S | - | L | } v _ { \mathrm { a n d } } ( \pmb { x } _ { L } ) = v _ { \mathrm { a n d } } ( \pmb { x } _ { L } ) \cdot \underbrace { \sum _ { m = 0 } ^ { | T | - | L | } C _ { | T | - | L | } ^ { m } ( - 1 ) ^ { m } } _ { = 0 } = 0 . } \end{array}$

Considering all the cases, the complete derivation of the sum of AND interactions is as follows.

$$
\begin{array} { r l } & { \quad \sum _ { S \subseteq T , S \neq \emptyset } I _ { S } ^ { \mathrm { A N D } } } \\ & { = \sum _ { S \subseteq T , S \neq \emptyset } \sum _ { L \subseteq S } ( - 1 ) ^ { | S | - | L | } v _ { \mathrm { a n d } } ( x _ { L } ) } \\ & { = \sum _ { L \subseteq T } \sum _ { S : L \subseteq S \subseteq T } ( - 1 ) ^ { | S | - | L | } v _ { \mathrm { a n d } } ( x _ { L } ) - v _ { \mathrm { a n d } } ( x _ { \emptyset } ) } \\ & { = \underbrace { v _ { \mathrm { s u d } } ( x _ { T } ) } _ { L = T } + \sum _ { L \subseteq T , L \neq T } v _ { \mathrm { a n d } } ( x _ { L } ) \cdot \underbrace { \sum _ { m = 0 } ^ { | T | - | L | } C _ { | T | - | L | } ^ { m } ( - 1 ) ^ { m } } _ { = 0 } - v _ { \mathrm { a n d } } ( x _ { \emptyset } ) } \\ & { = v _ { \mathrm { a n d } } ( x _ { T } ) - v ( x _ { \emptyset } ) } \end{array}\tag{6}
$$

Therefore, we have proven that $\begin{array} { r } { \forall \varnothing \neq T \subseteq N , v _ { \mathrm { a n d } } ( \pmb { x } _ { T } ) = v ( \pmb { x } _ { \varnothing } ) + \sum _ { S \subseteq T , S \neq \emptyset } I _ { S } ^ { \mathrm { A N D } } . } \end{array}$

(2) Universal matching theorem of OR interactions. What we need to prove is that $\forall T \subseteq N , v _ { \mathrm { o r } } ( \pmb { x } _ { T } ) =$ $\begin{array} { r } { \sum _ { S \in \{ S : S \cap T \neq \emptyset \} \cup \{ \emptyset \} } I _ { S } ^ { \mathrm { O R } } = \sum _ { S : S \cap T \neq \emptyset } I _ { S } ^ { \mathrm { O R } } } \end{array}$ . Here $I _ { \varnothing } ^ { \mathrm { O R } } = v _ { \mathrm { o r } } ( { \pmb x } _ { \varnothing } ) = 0$

According to the definition of the OR interaction, $\begin{array} { r } { I _ { S } ^ { \mathrm { O R } } : = - \sum _ { L \subset S } ( - 1 ) ^ { | S | - | L | } v _ { \mathrm { o r } } ( \pmb { x } _ { N \backslash L } ) } \end{array}$ . To simplify the computa-1 tion of the sum of OR interactions $\begin{array} { r } { \sum _ { S : S \cap T \neq \emptyset } I _ { S } ^ { \mathrm { O R } } = \sum _ { S : S \cap T \neq \emptyset } \left| - \sum _ { L \subseteq S } ( - 1 ) ^ { | S | - | L | } v _ { \mathrm { o r } } ( \pmb { x } _ { N \setminus L } ) \right| } \end{array}$ , we also exchange the order of summation of the set $L \subseteq S \subseteq N$ and the set $\bar { S ^ { \prime } } \colon S \cap T = \varnothing$ .Given a set of input variables $L ,$ we compute all linear combinations of all sets S containing L with respect to the model outputs ${ v _ { \mathrm { o r } } } ( \pmb { x } _ { N \backslash L } )$ , i.e., $\begin{array} { r } { \sum _ { S : S \cap T \not = \emptyset , N \supset S \supset L } ( - 1 ) ^ { | S | - | L | } v _ { \mathrm { o r } } ( \pmb { x } _ { N \setminus L } ) } \end{array}$ . Then, we compute all summations over the set $L \subseteq N$ as $\begin{array} { r } { \sum _ { S : S \cap T \neq \emptyset } I _ { S } ^ { \mathrm { O R } } = } \end{array}$ $\begin{array} { r } { - \sum _ { L \subseteq N } \sum _ { S : S \cap T \not = \emptyset , N \supseteq S \supseteq L } ( - 1 ) ^ { | S | - | L | } v _ { \mathrm { o r } } ( \pmb { x } _ { N \setminus L } ) } \end{array}$ . Then, we can compute different cases of $L \subseteq S \subseteq N , S \cap T \neq \emptyset$ as follows:

(1) When $L = N$ (then $\begin{array} { r } { S = N ) , \sum _ { S : S \cap T \neq \emptyset , S \supset L } ( - 1 ) ^ { | S | - | L | } v _ { \mathrm { o r } } ( x _ { N \setminus L } ) = ( - 1 ) ^ { | N | - | N | } v _ { \mathrm { o r } } ( x _ { \emptyset } ) = v _ { \mathrm { o r } } ( x _ { \emptyset } ) = 0 , } \end{array}$ Here $I _ { \varnothing } ^ { \mathrm { O R } } = v _ { \mathrm { o r } } ( { \pmb x } _ { \varnothing } ) = 0 .$

(2) When $L = N \setminus T$ , for all sets $S : S \supseteq L , S \cap T \neq \emptyset$ (then $S \neq N \setminus T , S \neq L )$ , let us consider the linear combinations of all sets $S$ with number |S| for the model output $v _ { \mathrm { o r } } ( { \pmb x } _ { T } )$ , respectively. Let $| S ^ { \prime } | : = | S | - | L | , ( 1 \leq | S ^ { \prime } | \leq | T | )$ then there are a total of $C _ { | T | } ^ { | S ^ { \prime } | }$ combinations of all sets S of order |S|. Thus, $\begin{array} { r } { \sum _ { S : S \cap T \neq \emptyset , S \supseteq L } ( - 1 ) ^ { | S | - | L | } v _ { \mathrm { o r } } ( \pmb { x } _ { N \setminus L } ) = } \end{array}$ vor(xT) · ∑|s|=1 |T| (−1)|S′| = −vor(xT).

(3) When $L \cap T \neq \emptyset , L \neq N .$ , for all sets $S : S \supseteq L , S \cap T \neq \emptyset$ , let us consider the linear combinations of all sets S with number $| S |$ for the model output $v _ { \mathrm { o r } } ( \pmb { x } _ { T } )$ , respectively. Let us split $\vert S \vert - \vert L \vert$ into $| S ^ { \prime } |$ and $| S ^ { \prime \prime } | , i . e . , | S | - | L | =$ $| S ^ { \prime } | + | S ^ { \prime \prime } |$ , where $S ^ { \prime } = \{ i | i \in S , i \notin L , i \in N \setminus T \} , S ^ { \prime \prime } = \{ i | i \in S , i \notin L , i \in T \}$ (then $0 \leq | S ^ { \prime \prime } | \leq | T | - | T \cap$ $L | )$ and $S ^ { \prime } + S ^ { \prime \prime } + L = S$ . Thus, there are a total of $C _ { \vert T \vert - \vert T \cap L \vert } ^ { \vert S ^ { \prime \prime } \vert }$ combinations of all sets $S ^ { \prime \prime }$ of order $| S ^ { \prime \prime } |$ . Thus, $\begin{array} { r } { \sum _ { S : S \cap T \neq \emptyset , S \geq L } ( - 1 ) ^ { | S | - | L | } v _ { \infty } ( x _ { N \setminus L } ) = v _ { \infty } ( x _ { N \setminus L } ) \cdot \sum _ { S ^ { \prime } \subseteq N \setminus T \setminus L } \sum _ { | S ^ { \prime \prime } | = 0 } ^ { | T | - | T \cap L | } C _ { | T | - | T \cap L | } ^ { | S ^ { \prime \prime } | } ( - 1 ) ^ { | S ^ { \prime } | + | S ^ { \prime \prime } | } = 0 . } \end{array}$

(4) When $L \cap T = \emptyset , L \neq N \setminus T$ , let us split $\left| S \right| - \left| L \right|$ into |S′| and $| S ^ { \prime \prime } | , i . e . , | S | - | L | = | S ^ { \prime } | + | S ^ { \prime \prime } |$ , where $S ^ { \prime } =$ $\{ i | i \in S , i \notin L , i \in N \setminus T \} , S ^ { \prime \prime } = \{ i | i \in S , i \in T \}$ (then $0 \leq | S ^ { \prime \prime } | \leq | T | )$ and $S ^ { \prime } + S ^ { \prime \prime } + L = S$ . Thus, there are a total of $C _ { | T | } ^ { | S ^ { \prime \prime } | }$ combinations of all sets $S ^ { \prime \prime }$ of order $| S ^ { \prime \prime } |$ . Thus, $\begin{array} { r } { \sum _ { S : S \cap T \neq \emptyset , S \supset L } ( - 1 ) ^ { | S | - | L | } v _ { \mathrm { o r } } ( \pmb { x } _ { N \setminus L } ) = v _ { \mathrm { o r } } ( \pmb { x } _ { N \setminus L } ) . } \end{array}$ $\begin{array} { r } { \sum _ { S ^ { \prime } \subseteq N \setminus T \setminus L } \underbrace { \sum _ { | S ^ { \prime \prime } | = 0 } ^ { | T | } C _ { | T | } ^ { | S ^ { \prime \prime } | } ( - 1 ) ^ { | S ^ { \prime } | + | S ^ { \prime \prime } | } } _ { = 0 } = 0 . } \end{array}$

Considering all the cases, the complete derivation of the sum of OR interactions is as follows.

$$
\begin{array} { r l }  \sum _ { s , w ^ { \prime } , s ^ { \prime \prime } } \frac { \partial w ^ { \prime } } { \partial s } = \sum _ { s , w ^ { \prime \prime } , s ^ { \prime \prime } } \Bigg [ \sum _  s , z , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime } , s ^ { \prime \prime } , s ^ { \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime \prime } , s ^ { \prime \prime \prime } , s ^ { \prime \prime \prime \prime } , s ^ { \prime \prime \prime \prime } , s ^ { \prime \prime \prime \prime } , s ^ { \prime \prime \prime \prime } , s ^ { \prime \prime \prime \prime } , s ^ { \prime \prime \prime \prime } , s ^ { \prime \prime \prime \prime } , s ^ { \prime \prime \prime \prime } , s ^ { \prime \prime \prime \prime }  \end{array}\tag{7}
$$

Therefore, we have proven that $\begin{array} { r } { \forall T \subseteq N , v _ { \mathrm { o r } } ( { \pmb x } _ { T } ) = \sum _ { S : S \cap T \not = \emptyset } I _ { S } ^ { \mathrm { O R } } . } \end{array}$

(3) Universal matching theorem of AND-OR interactions. With the universal matching property of AND interactions and the universal matching property of OR interactions, we can easily get $v ( \pmb { x } _ { T } ) = \phi ( \pmb { x } _ { T } ) = v _ { \mathrm { a n d } } ( \pmb { x } _ { T } ) + v _ { \mathrm { o r } } ( \pmb { x } _ { T } ) = v ( \pmb { x } _ { \emptyset } ) +$ $\begin{array} { r } { \sum _ { S \subseteq T , S \neq \emptyset } I _ { S } ^ { \mathrm { A N D } } + \sum _ { S \subseteq N , S \cap T \neq \emptyset } I _ { S } ^ { \mathrm { O R } } } \end{array}$ , thus, we obtain the universal matching property of AND-OR interactions. □

## B.2. Proof of Sparsity Property

Given all the masked samples $\{ { \pmb x } _ { T } \mid T \subseteq N \}$ , the surrogate logical model $\phi ( { \pmb x } _ { T } )$ only utilizes a small set of salient AND interactions in $\Omega ^ { \mathrm { A N D } }$ and salient OR interactions in $\Omega ^ { \tilde { \mathrm { O R } } }$ to approximate the network output score $v ( x _ { T } )$ . That is, the network's output can be well approximated by a small set of AND-OR interactions.

$$
v ( \pmb { x } _ { T } ) = \phi ( \pmb { x } _ { T } ) \approx v ( \pmb { x } _ { \emptyset } ) + \sum _ { S \subseteq T , S \neq \emptyset , S \in \Omega _ { \mathrm { A N D } } } I _ { S } ^ { \mathrm { A N D } } + \sum _ { S \subseteq T , S \neq \emptyset , S \in \Omega _ { \mathrm { B R } } } I _ { S } ^ { \mathrm { O R } }\tag{8}
$$

Proof. It has been proven by Ren et al. (2024a) that under three common conditions7, the output score ${ v } _ { \mathrm { a n d } } ( { \pmb x } _ { T } )$ of a welltrained DNN on all $2 ^ { n }$ masked samples $\{ { \pmb x } _ { T } | T \subseteq N \}$ could be universally estimated by a small number of AND interactions $T \in \Omega ^ { \mathrm { A N D } }$ with salient interaction effects $\begin{array} { r } { I _ { S } ^ { \mathrm { A N D } } , \mathrm { ~ } _ { S } ^ { \mathrm { } } . t . , | \Omega ^ { \mathrm { A N D } } | \ll 2 ^ { n } , i . e . , \forall T \subseteq \mathbf { \dot { \Omega } } N , v _ { \mathrm { a n d } } ( { \pmb x } _ { T } ) = \sum _ { S \subseteq T , S \neq \emptyset } I _ { S } ^ { \mathrm { A N D } } \approx } \end{array}$ $\begin{array} { r } { \sum _ { S \subseteq T , S \not = \emptyset , S \in \Omega ^ { \mathrm { A N D } } } I _ { S } ^ { \mathrm { A N D } } } \end{array}$ . According to Eq. (6), $\begin{array} { r } { v _ { \mathrm { a n d } } ( \pmb { x } _ { T } ) = v ( \pmb { x } _ { \emptyset } ) + \sum _ { S \subseteq T , S \not = \emptyset } I _ { S } ^ { \mathrm { A N D } } } \end{array}$ . Therefore, $v _ { \mathrm { a n d } } ( { \pmb x } _ { T } ) \approx v ( { \pmb x } _ { \emptyset } ) +$ $\begin{array} { r } { \sum _ { S \subseteq T , S \not = \emptyset , S \in \Omega ^ { \mathrm { A N D } } } I _ { S } ^ { \mathrm { A N D } } } \end{array}$

Besides, as proven in Section C, the OR interaction can be considered as a special AND interaction. Thus, the confidence score $v _ { \mathrm { o r } } ( { \pmb x } _ { T } )$ of a well-trained DNN on all $2 ^ { n }$ masked samples $\{ { \pmb x } _ { T } | T \ \subseteq \ N \}$ could be universally estimated by a small number of OR interactions $T \in \Omega ^ { \mathrm { O R } }$ with salient interaction effects $I _ { S } ^ { \mathrm { O R } } , \bar { s . t . } , | \Omega ^ { \mathrm { O R } } | \ll 2 ^ { n }$ . Similarly, $v _ { \mathrm { o r } } ( \pmb { x } _ { T } ) =$ $\begin{array} { r } { \sum _ { S \subseteq T , S \not \in \emptyset } I _ { S } ^ { \mathrm { O R } } \approx \sum _ { S \subseteq T , S \not \in \emptyset , S \in \Omega ^ { \mathrm { o R } } } I _ { S } ^ { \mathrm { O R } } } \end{array}$

Thus, for each randomly masked sample $\pmb { x } _ { T } , T \subseteq N$ , the surrogate logical model $\phi ( { \pmb x } _ { T } )$ can use a small number of salient AND-OR interactions to approximate the network output score $v ( x _ { T } )$ $i . e . , v ( { \pmb x } _ { T } ) = \phi ( { \pmb x } _ { T } ) = v _ { \mathrm { a n d } } ( { \pmb x } _ { T } ) + v _ { \mathrm { o r } } ( { \pmb x } _ { T } ) \approx$ $\begin{array} { r } { ( { \bf x } _ { \emptyset } ) + \sum _ { S \subseteq T , S \neq \emptyset , S \in \Omega _ { \mathrm { A N D } } } \bar { I } _ { S } ^ { \mathrm { A N D } } + \sum _ { S \subseteq T , S \neq \emptyset , S \in \Omega _ { \mathrm { o R } } } { I } _ { S } ^ { \mathrm { O R } } } \end{array}$

## C. OR Interactions Can Be Considered as Special AND Interactions

If we reverse the definition of the masked state and the unmasked state of the input variable, the OR interaction $I _ { S } ^ { \mathrm { O R } }$ can be considered as a special kind of AND interaction $I _ { S } ^ { \mathrm { A N D } }$

Given an input sample $\pmb { x } \in \mathbb { R } ^ { n }$ and the output score of a DNN as $v ( \cdot )$ , if we randomly mask input variables in æ, we can get all $2 ^ { n }$ masked samples. Let $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } \mathrm { ~ \it ~  ~ } }$ denote the certain masked input sample when input variables in $N \backslash S$ are all masked and input variables in S are kept unchanged.

$$
( \pmb { x } _ { S } ) _ { i } = \left\{ \begin{array} { l l } { \pmb { x } _ { i } , } & { i \in S } \\ { { b } _ { i } , } & { i \in N \setminus S } \end{array} \right.\tag{9}
$$

where b $\in \mathbb { R } ^ { n }$ are baseline values to represent the masked state of input variables.

If we reverse the definition of the masked state and the unmasked state of an input variable, $i . e .$ , we consider b as the input sample and consider x as the masked state, then the masked sample $\widetilde { \pmb { x } } _ { S }$ can be defined as follows.

$$
( \widetilde { \pmb x } _ { S } ) _ { i } = \left\{ \begin{array} { l l } { b _ { i } , } & { i \in S } \\ { x _ { i } , } & { i \in N \setminus S } \end{array} \right.\tag{10}
$$

Thus, we can get ${ \pmb x } _ { N \backslash S } = \widetilde { \pmb x } _ { S }$ . To simplify the analysis, let us assume $v _ { \mathrm { a n d } } ( { \pmb x } _ { S } ) = v _ { \mathrm { o r } } ( { \pmb x } _ { S } ) = 0 . 5 v ( { \pmb x } _ { S } )$ , then the OR

- 11ama-2-7b-chat vs. 11ama-2-7b   
- 11ama-2-13b-chat vs. 11ama-2-13b   
- 11ama-2-70b-chat vs. 11ama-2-70b   
- 11ama-3-8b-instruct vs. 11ama-3-8b

interaction $I _ { S } ^ { \mathrm { O R } }$ can be regarded as a specific AND interaction $I _ { S } ^ { \mathrm { A N D } } ( \widetilde { \pmb { x } } )$ as follows.

$$
\begin{array} { r l } & { I _ { S } ^ { \mathrm { O R } } ( \mathbf { x } ) = - \displaystyle \sum _ { T \subseteq S } ( - 1 ) ^ { | S | - | T | } v _ { \mathrm { o r } } ( \pmb { x } _ { N \backslash T } ) , } \\ & { \qquad = - \displaystyle \sum _ { T \subseteq S } ( - 1 ) ^ { | S | - | T | } v _ { \mathrm { o r } } ( \widetilde { \pmb { x } } _ { T } ) , } \\ & { \qquad = - \displaystyle \sum _ { T \subseteq S } ( - 1 ) ^ { | S | - | T | } v _ { \mathrm { a n d } } ( \widetilde { \pmb { x } } _ { T } ) , } \\ & { \qquad = - I _ { S } ^ { \mathrm { A N D } } ( \widetilde { \pmb { x } } ) . } \end{array}\tag{11}
$$

Now we have proven that OR interactions can be considered as special AND interactions.

## D. Details of Extracting the Sparsest AND-OR Interactions

We follow Li & Zhang (2023) to extract AND-OR interactions. Given a masked sample ${ \mathbf { } } x _ { T } ,$ the output score of the network $v ( x _ { T } )$ can be decomposed into a combination of AND interaction and OR interaction, $i . e . , v ( { \pmb x } _ { T } ) = v _ { \mathrm { a n d } } ( { \pmb x } _ { T } ) + v _ { \mathrm { o r } } ( { \pmb x } _ { T } )$ Specifically, $v _ { \mathrm { a n d } } ( { \pmb x } _ { T } ) = 0 . 5 \cdot v ( { \pmb x } _ { T } ) + \gamma _ { T }$ and $v _ { \mathrm { o r } } ( { \pmb x } _ { T } ) = 0 . 5 \cdot v ( { \pmb x } _ { T } ) - \gamma _ { T }$ , where $\{ \gamma _ { T } \ | \ T \subseteq N \}$ is a set of learnable parameters. The parameters $\{ \gamma _ { T } \}$ were trained through minimizing the following LASSO-like loss to obtain sparse interactions:

$$
\operatorname* { m i n } _ { \{ \gamma _ { T } \} } \sum _ { S \subseteq N } | I _ { S } ^ { \mathrm { A N D } } ( { \pmb x } ) | + | I _ { S } ^ { \mathrm { O R } } ( { \pmb x } ) | ,\tag{12}
$$

where $\begin{array} { r l r } { I _ { S } ^ { \mathrm { A N D } } ( \pmb { x } ) } & { { } = } & { \sum _ { T \subsetneq S } ( - 1 ) ^ { | S | - | T | } v _ { \mathrm { a n d } } ( \pmb { x } _ { T } ) \quad = \quad \sum _ { T \subsetneq S } ( - 1 ) ^ { | S | - | T | } ( 0 . 5 \cdot \triangledown \cdot \triangledown \cdot \triangledown \cdot \triangledown \cdot \ \gamma _ { T } ) } \end{array}$ and $\begin{array} { r l } { I _ { S } ^ { \mathrm { O R } } ( { \pmb x } ) } & { { } = } \end{array}$ $\begin{array} { r } { - \sum _ { T \subseteq S } ( - 1 ) ^ { | S | - | T | } v _ { \mathrm { o r } } ( x _ { N \setminus T } ) = - \sum _ { T \subseteq S } ( - 1 ) ^ { | S | - | T | } ( 0 . 5 \cdot \bar { v } ( { \pmb x } _ { T } ) - \gamma _ { T } ) } \end{array}$ . Thus, we can extract the sparsest set of AND-ōR interactions.

## E. Experimental Details

## E.1. Computing Infrastructure

We conducted all our experiments on four NVIDIA Tesla V100-DGXS GPUs, each with 32 GB of VRAM. The software environment consisted of NVIDIA Driver version 570.133.07 and CUDA 12.8.

For all of the evaluated LLMs, we used a torch.float16 data type, which provides a standard level of precision for inference tasks.

## E.2. Model Details

We conduct experiments on 50 open-source LLMs from 6 major model families. A comprehensive list of all evaluated models is provided in Table 1. To facilitate a controlled analysis of the factors influencing prompt sensitivity, we group these models into specific subsets for each comparison, as detailed below.

(1) Instruct/Chat vs. Base Models. To investigate the impact of the alignment process, we form pairs of instruct/chat models and their corresponding base models. This comparison includes models from the Llama, Mistral, Qwen, InternLM, and Olmo families. The main LLMs used for this comparison are:

• Llama Family:

• Mistral Family:

• Qwen Family:

Table 1. A comprehensive list and characteristics of LLMs, grouped by model family and model series.
<table><tr><td>Model Family</td><td>Model Series</td><td>Model</td><td>Type</td><td>Architecture</td><td>Scale</td></tr><tr><td rowspan="9"></td><td rowspan="6">Llama 2 (6 models)</td><td>Llama-2-7b</td><td>Base</td><td>Dense</td><td>7B</td></tr><tr><td>Llama-2-7b-chat</td><td>Chat</td><td>Dense</td><td>7B</td></tr><tr><td>Llama-2-13b</td><td>Base</td><td>Dense</td><td>13B</td></tr><tr><td>Llama-2-13b-chat</td><td>Chat</td><td>Dense</td><td>13B</td></tr><tr><td>Llama-2-70b</td><td>Base</td><td>Dense</td><td>70B</td></tr><tr><td>Llama-2-70b-chat</td><td>Chat</td><td>Dense</td><td>70B</td></tr><tr><td>Llama-3-8b</td><td>Base</td><td>Dense</td><td>8B</td></tr><tr><td rowspan="3"></td><td>Llama-3-8b-instruct</td><td>Instruct</td><td>Dense</td><td>8B</td></tr><tr><td>Llama-moe-v1-3_5b-2_8-sft Instruct</td><td></td><td>MoE</td><td>3.5B (Activated)</td></tr><tr><td>Llama-moe-v2-3_8b-2_8-sft Instruct</td><td></td><td>MoE</td><td>3.8B (Activated)</td></tr><tr><td rowspan="4">Mistral (4 models)</td><td rowspan="2">Mistral (2 models)</td><td>Mistral-7b-v0.3</td><td>Base</td><td>Dense</td><td>7B</td></tr><tr><td>Mistral-7b-v0.3-instruct</td><td>Instruct</td><td>Dense</td><td>7B</td></tr><tr><td rowspan="2">Mixtral (2 models)</td><td>Mixtral-8x7b</td><td>Base</td><td>MoE</td><td>13B (Activated)</td></tr><tr><td>Mixtral-8x7b-instruct</td><td>Instruct</td><td>MoE</td><td>13B (Activated)</td></tr><tr><td rowspan="10"></td><td rowspan="4">Qwen 1.5 MoE (2 models)</td><td>Qwen1.5-moe-a2.7b-chat</td><td>Chat</td><td>MoE</td><td>2.7B (Activated)</td></tr><tr><td>Qwen1.5-moe-a2.7b</td><td>Base</td><td>MoE</td><td>2.7B (Activated)</td></tr><tr><td>Qwen2-7b</td><td>Base</td><td>Dense</td><td>7B</td></tr><tr><td>Qwen2-0.5b-instruct</td><td>Instruct</td><td>Dense</td><td>0.5B</td></tr><tr><td rowspan="5">Qwen 2 (5 models)</td><td>Qwen2-1.5b-instruct</td><td>Instruct</td><td>Dense</td><td>1.5B</td></tr><tr><td>Qwen2-7b-instruct</td><td>Instruct</td><td>Dense</td><td>7B</td></tr><tr><td>Qwen2-72b-instruct</td><td>Instruct</td><td>Dense</td><td>72B</td></tr><tr><td>Qwen2.5-0.5b-instruct</td><td>Instruct</td><td>Dense</td><td>0.5B</td></tr><tr><td>Qwen2.5-1.5b-instruct</td><td>Instruct</td><td>Dense</td><td>1.5B</td></tr><tr><td rowspan="8">Qwen 2.5 (7 models) Qwen (25 models)</td><td>Qwen2.5-3b-instruct</td><td>Instruct</td><td>Dense</td><td>3B</td></tr><tr><td>Qwen2.5-7b-instruct</td><td>Instruct</td><td>Dense</td><td>7B</td></tr><tr><td>Qwen2.5-14b-instruct</td><td>Instruct</td><td>Dense</td><td>14B</td></tr><tr><td>Qwen2.5-32b-instruct</td><td>Instruct</td><td>Dense</td><td>32B</td></tr><tr><td>Qwen2.5-72b-instruct</td><td>Instruct</td><td>Dense</td><td>72B</td></tr><tr><td>Qwen3-0.6b-instruct</td><td>Instruct</td><td>Dense</td><td>0.6B</td></tr><tr><td>Qwen3-1.7b-instruct Qwen3-4b</td><td>Instruct</td><td>Dense</td><td>1.7B</td></tr><tr><td></td><td>Base</td><td>Dense</td><td>4B</td></tr><tr><td rowspan="8">Qwen 3 (9 models)</td><td>Qwen3-4b-instruct</td><td>Instruct</td><td>Dense</td><td>4B</td></tr><tr><td>Qwen3-8b</td><td>Base</td><td>Dense</td><td>8B</td></tr><tr><td>Qwen3-8b-instruct</td><td>Instruct</td><td>Dense</td><td>8B</td></tr><tr><td>Qwen3-14b</td><td>Base</td><td>Dense</td><td>14B</td></tr><tr><td>Qwen3-14b-instruct</td><td>Instruct</td><td>Dense</td><td>14B</td></tr><tr><td>Qwen3-32b-instruct</td><td>Instruct</td><td>Dense</td><td>32B</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-30b-a3b-instruct Qwen 3 A3B (2 models) Qwen3-30b-a3b</td><td>Instruct</td><td>MoE MoE</td><td>3B (Activated)</td></tr><tr><td rowspan="8">Olmo (9 models)</td><td rowspan="4">Olmo v1 (3 models)</td><td></td><td>Base</td><td></td><td>3B (Activated)</td></tr><tr><td>Olmo-1b</td><td>Base</td><td>Dense</td><td>1B</td></tr><tr><td>Olmo-7b</td><td>Base</td><td>Dense</td><td>7B</td></tr><tr><td>Olmo-7b-instruct</td><td>Instruct</td><td>Dense</td><td>7B</td></tr><tr><td rowspan="5">Olmo v2 (4 models)</td><td>Olmo-2-1b-instruct</td><td>Instruct</td><td>Dense</td><td>1B</td></tr><tr><td>Olmo-2-7b-instruct</td><td>Instruct</td><td>Dense</td><td>7B</td></tr><tr><td>Olmo-2-13b-instruct</td><td>Instruct</td><td>Dense</td><td>13B</td></tr><tr><td>Olmo-2-32b-instruct</td><td>Instruct</td><td>Dense</td><td>32B</td></tr><tr><td>Olmoe-7b</td><td>Base</td><td>MoE</td><td>1B (Activated)</td></tr><tr><td colspan="2"></td><td>Olmoe-7b-instruct</td><td>Instruct</td><td>MoE</td><td>1B (Activated)</td></tr><tr><td colspan="2">InternLM (2 models) InternLM 2 (2 models)</td><td>Internlm2-7b Internlm2-chat-7b</td><td>Base Chat</td><td>Dense Dense</td><td>7B 7B</td></tr></table>

- qwen3-4b-instruct vs. qwen3-4b   
- qwen3-8b-instruct vs. qwen3-8b   
- qwen3-14b-instruct vs. qwen3-14b   
- qwen1.5-moe-a2.7b-chat vs. qwen1.5-moe-a2.7b   
- qwen3-30b-a3b-instruct vs. qwen3-30b-a3b

## • Olmo Family:

- olmo-7b-instruct vs. olmo-7b

- olmoe-7b-instruct vs. olmoe-7b

## • InternLM Family:

- internlm2-chat-7b vs. internlm2-7b

(2) Dense vs. MoE Models. To analyze the effect of architecture, we compare dense and Mixture-of-Experts (MoE) models. primarily within the same model family to control for other variables. The main LLMs used for this comparison are:

## • Llama Family:

- Dense: 11ama-2-7b, 11ama-2-7b-chat, 11ama-2-13b, 11ama-2-13b-chat, 1lama-2-70b,   
11ama-2-70b-chat,11ama-3-8b,11ama-3-8b-instruct.

-MoE:11ama-moe-v1-3\_5b-2\_8-sft,11ama-moe-v2-3\_8b-2\_8-sft.

## • Mistral Family:

-Dense: mistral-7b-v0.3,mistral-7b-v0.3-instruct.

- MoE:mixtral-8x7b,mixtral-8x7b-instruct.

## • Qwen Family:

\- Dense: qwen2-7b, qwen2-7b-instruct, qwen2-72b-instruct, qwen2.5-7b-instruct, qwen2.5-14b-instruct, qwen2.5-32b-instruct, qwen2.5-72b-instruct, qwen3-4b, qwen3-4b-instruct, qwen3-8b, qwen3-8b-instruct, qwen3-14b, qwen3-14b-instruct, qwen3-32b-instruct.

- MoE: qwen1.5-moe-a2.7b, qwen1.5-moe-a2.7b-chat, qwen3-30b-a3b,  
qwen3-30b-a3b-instruct.

## • Olmo Family:

\- Dense: olmo-7b, olmo-7b-instruct, olmo-2-7b-instruct, olmo-2-13b-instruct, olmo-2-32b-instruct.

```yaml
- MoE: olmoe-7b,olmoe-7b-instruct.
```

(3) Model Scale. To study the impact of model scale, we analyze a series of LLMs from the same family and with the same training paradigm but with varying parameter counts. The main LLMs used for this comparison are:

• Llama-2 (Base): 7b, 13b, 70b.

• Llama-2 (Chat): 7b, 13b, 70b.

• Qwen2 (Instruct): 0 .5b, 1.5b, 7b, 72b.

• Qwen2.5 (Instruct): 0 .5b, 1. 5b, 3b, 7b, 14b, 32b, 72b.

• Qwen3 (Instruct): 0 . 6b, 1.7b, 4b, 8b, 14b, 32b.

• Olmo-2 (Instruct): 1b, 7b, 13b, 32b.

## E.3. Generation Configuration of LLMs

To ensure reproducible results, we employ a greedy search strategy for all LLMs. This is achieved by setting the “do\_sample" parameter to “False" in our generation configuration. When “do\_sample=False", the LLM selects the token with the highest probability as the next token in the sequence. By adopting this greedy approach, we eliminate the randomness inherent in sampling-based methods. The configuration ensures that for a given input, the same LLM will generate the exact same output every time, which is a critical requirement for the replicability of our experiments.

## E.4. How to Mask Input Words For Different LLMs

To compute interactions, we follow the approach of Cheng et al. (2025) and mask the words in N \S by replacing them with a LLM-specific [MASK] token. Our selection of this token follows a prioritized strategy: (1) We preferentially use the LLM's designated unknown (<unk>) token. (2) If an unknown token is not available or suitable, we use the padding (<pad>) token as a fallback. Since the specific token strings and their corresponding IDs vary across different LLMs, the exact mask token used for each LLM is detailed below:

• For 1lama-2-7b, 11ama-2-7b-chat, 1lama-moe-v1-3.5b-2.8-sft, mistral-7b-v0.3, mistral-7b-v0.3-instruct,mixtral-8x7b,mixtral-8x7b-instruct,internlm2-7b, internlm2-chat-7b, we use the <unk> token (ID: 0) to mask words.

• For 1lama-2-13b, 1lama-2-13b-chat, 1lama-2-70b, 1lama-2-70b-chat, we use the < |pad\_t oken | > token (ID: 0) to mask words.

• Forllama-3-8b,1lama-3-8b-instruct,weusethe<|pad\_token|>/<|reserved\_special\_token\_250|> token (ID: 128255) to mask words.

• For 1lama-moe-v2-3.8b−2.8-sft, we use the <|pad\_token|>/<|eot\_id|> token (ID: 128009) to mask words.

• For qwen2–7b, we use the < | PAD\_TOKEN | > token (ID: 151 64 6) to mask words.

• For a large group of Qwen models, including qwen2-0.5b-instruct, qwen2-1.5b-instruct, qwen2-7b-instruct, qwen2-72b-instruct, qwen2.5series, qwen3-0.6b-instruct, qwen3-1.7b-instruct, qwen3-4b series, qwen3-32b-instruct, qwen1.5-moe series, and qwen3-30b-a3b series, we use the <|pad\_token|>/<|endoftext|> token (ID: 151643) to mask words.

• For qwen3-8b, qwen3-8b-instruct, qwen3-14b, qwen3-14b-instruct, we use the < |pad\_token |>/< |vision-pad|> token (ID: 151 654) to mask words.

• For the Olmo V1 series models, including olmo-1b, olmo-7b, olmo-7b-instruct, olmoe-7b, and olmoe-7b-instruct, we use the < |padding |> token (ID: 1) to mask words.

• For the Olmo V2 series models, including olmo-2-1b-instruct, olmo-2-7b-instruct, olmo-2-13b-instruct, and olmo-2-32b-instruct, we use the <|pad\_token|>/<|endoftext|> token (ID: 10 0257) to mask words.

## E.5. Prompt Templates

To systematically evaluate the prompt sensitivity of LLMs, we designed a set of five distinct prompt templates. As illustrated in Figure 15, these templates are derived from a base prompt template (i.e., Prompt Template 1) through a series of subtle, semantically irrelevant modifications. These variations include changes in letter case, e.g., “Answers" vs. “ANSWERS" and alterations to separators, e.g., “:" vs. “::" or the format of option markers, e.g., “A." vs. “A)". Crucially, these changes only affect the superficial formatting while preserving the core semantic meaning of the prompt template.

In our experimental procedure, for a given input, which consists of a question and options, we apply each of the five prompt templates to generate five prompts. For every unique pair of these five prompts, we then calculate the prompt sensitivity by quantifying the change in the interactions among the input variables (i.e., words within the question and options). This procedure allows us to precisely measure how much the LLM's interaction patterns of the core input are perturbed by superficial changes in the prompt template, thus evaluating the prompt sensitivity of LLMs.

Prompt Template 1: Prompt Template 2: Prompt Template 3: Prompt Template 4: Prompt Template 5:  
{Question} {Question} {Question} {Question} {Question}  
Answers: ANSWERS: Answers:: ANSWERS: : Answers:  
A.{Option A} A.{Option A} A.{Option A} A.{Option A} A){Option A}  
B.{Option B} B.{Option B} B.{Option B} B.{Option B} B){Option B}  
C.{Option C} C.{Option C} C.{Option C} C.{Option C} C){Option C}  
D.{Option D} D.{Option D} D.{Option D} D.{Option D} D){Option D}  
Answer: ANSWER: Answer:: ANSWER:: Answer:  
Figure 15. Different prompt templates. Red parts show the difference between the current prompt template with the first prompt template, i.e., Prompt Template 1.

## E.6. Few-shot Learning Templates

For this experiment, we selected the pair of prompt templates that exhibited the highest average prompt sensitivity in the 0-shot setting, aiming to test if few-shot learning could help the most severe situation. To investigate whether few-shot learning can mitigate high prompt sensitivity, we conducted a follow-up experiment. We selected Prompt Template 1 and Prompt Template 4 from Figure 15 for this analysis, as this pair exhibited the highest average prompt sensitivity in our 0-shot setting. This allowed us to test the efficacy of few-shot learning in the most challenging scenario.

Based on these two base templates, we constructed few-shot learning prompts with one, two, and three in-context examples (i.e., 1-shot, 2-shot, and 3-shot learning), as illustrated in Figure 16. The examples were formulated using certain questions and their corresponding answers, randomly selected from a set of datasets that are not included in the test set. The structure of each example is related to its corresponding prompt template. For instance, the first example (Example 1) is formatted differently for each template:

• Example 1 For Prompt Template 1:

Question: Which type of precipitation consists of frozen rain drops?   
Answers:   
A.sleet   
B.hail   
C.snow   
D.fog   
Answer: A

• Example 1 For Prompt Template 4 (Note the different format):

Question: Which type of precipitation consists of frozen rain drops?   
ANSWERS::   
A.sleet   
B.hail   
C.snow   
D.fog   
ANSWER:: A

The other two examples (Example 2 and Example 3) are presented below:

• Example 2 For Prompt Template 1:

Question: Decayed prehistoric plants have helped in the formation of   
Answers:   
A.coal, shale, and quartz.   
B.coal, oil, and gas.   
C.shale, quartz, and coal.   
D.oil, shale, and granite.   
Answer: B

## • Example 2 For Prompt Template 4:

Question: Decayed prehistoric plants have helped in the formation of   
ANSWERS: :   
A.coal, shale, and quartz.   
B.coal, oil, and gas.   
C.shale, quartz, and coal.   
D.oil, shale, and granite.   
ANSWER:: B

## • Example 3 For Prompt Template 1:

Question: Which describes a material that is not a food?   
Answers:   
A.It stores energy but not nutrients.   
B.It does not store energy or nutrients.   
C.It stores energy and nutrients.   
D.It does not store energy but stores nutrients.   
Answer: B

## • Example 3 For Prompt Template 4:

Question: Which describes a material that is not a food?   
ANSWERS: :   
A.It stores energy but not nutrients.   
B.It does not store energy or nutrients.   
C.It stores energy and nutrients.   
D.It does not store energy but stores nutrients.   
ANSWER:: B

![](images/93a3c7faab1ad770e581f4aadb8cd94d05ad977c70a7a84215c789816f13ef6b.jpg)  
Figure 16. Prompt templates of few-shot learning.

## 1-shot Learning

## Prompt Template 1 :

Instruction: Read the following question and   
the four options provided. Choose the single   
best answer and provide only its corresponding   
letter.   
Example:   
{Example 1}   
Question: {Question}   
Answers:   
A.{Option A}   
B.{Option B}   
C.{Option C}   
D.{Option D}   
Answer:

## 2-shot Learning

## Prompt Template 1 :

Instruction: Read the following question and   
the four options provided. Choose the single   
best answer and provide only its corresponding   
letter.   
Example:   
{Example 1}   
{Example 2}   
Question: {Question}   
Answers:   
A.{Option A}   
B.{Option B}   
C.{Option C}   
D.{Option D}   
Answer:

## 3-shot Learning

## Prompt Template 1 :

## Prompt Template 2 :

Instruction: Read the following question and   
the four options provided. Choose the single   
best answer and provide only its corresponding   
letter.   
Example:   
{Example 1}   
Question: {Question}   
ANSWERS: :   
A.{Option A}   
B.{Option B}   
C.{Option C}   
D.{Option D}   
ANSWER::

## Prompt Template 2 :

Instruction: Read the following question and   
the four options provided. Choose the single   
best answer and provide only its corresponding   
letter.   
Example:   
{Example 1}   
{Example 2}   
Question: {Question}   
ANSWERS: :   
A.{Option A}   
B.{Option B}   
C.{Option C}   
D.{Option D}   
ANSWER: :

## Prompt Template 2 :

## F. More Experimental Results

## F.1. More Results on the Verification of the Sparsity of Interactions

Here are more results on the verification of the sparsity of interactions. As illustrated in Figure 17, the results verify that only a small set of interactions have salient effects, while most of the interactions have negligible effects and can be considered as noise patterns.

![](images/c71ec6ebc3aeb6f26cc04c60be4fd5e965d11b187a829242084169806ad8371e.jpg)  
Figure 17. Verifying the sparsity of interactions. We show absolute values of normalized interactions in a descending order. LLMs all encode a small number of salient interactions, while most of the interaction effects are negligible.

## F.2. More Results on the Verification of the Sparsity of Interactions

Here are more results on the verification of quality of universal matching. Figure 18 compares the LLM's true output v(æT) for all masked inputs against the logical model using only the most salient interactions. Even when using just the top 3% or top 5% of all interactions, the matching error is minimal. This empirically demonstrates that the LLM's output can be faithfully approximated by a small, sparse set of salient interactions.

![](images/905676098029bfedd7d5069a6913a1ace2d29226530aeb2d8725527eaab2dbb3.jpg)  
Figure 18. Verifying the quality of universal matching for any $2 ^ { n }$ masked inputs. The red line plots outputs of the LLM in an ascending order.

## F.3. Detailed Case Study

Figure 19 is the detailed case study of how to use our interaction-based analytical tool. It offers preliminary evidence that semantically irrelevant alterations to the prompt template can lead to significant changes in the salient interaction patterns, even when the input and output remains unchanged. This reveals the existence of unstable interactions, which we propose as the underlying cause of prompt sensitivity.

Case 1  
![](images/8e7f0ab7fc934c58481ed8351b3fbc1b2a891a12359362ebf705d4e64fd62976.jpg)

Input x + Prompt Template T →  
![](images/ce4bb995f52e1d316e89f170e97fffb3187da02af6248054f95c5dfdb30cd4f0.jpg)

![](images/cb35f4c751024c6aa5894bf94d57f48f22415eb6dff2e416cd044858fe2c7884.jpg)

Case 2  
Input x + Prompt Template T →  
![](images/29c3f615ba6e7665ede29353e91824745bd1fc8e38539cc56e0057ef0b3f6155.jpg)

Input x + Prompt Template T →  
![](images/f89397bd885d704d0d9b6e75defdb17225c92983a1c13b10d5e394812cf5cdd3.jpg)

![](images/4efe96bb196fd42d66bf9d9834b4ae271b8a2fca4fbd5cbe6783c63233f3c6a1.jpg)  
Figure 19. A case study of interaction-level analysis revealing latent instability. The same input x is formatted with two semantically identical templates, T and T, differing only in letter case (e.g.,“Answer" vs. “ANSWER"). Although the LLM generates the same correct output $\mathrm { ( ^ { 6 6 } D ^ { 7 } ) }$ in both cases, the composition of the interaction-based logical model φ(x) reveals significant internal divergence. Many interaction effects are highly unstable, changing in either sign or magnitude. This highlights a critical risk of prompt sensitivity that is invisible to output-level

## F.4. More Results on the Prompt Sensitivity of Different Orders

Here are more results on the prompt sensitivity of different orders on the ARC dataset. As illustrated in Figure 20, it shows that the prompt sensitivity of low-order interactions is the lowest, followed by mid-order, while high-order interactions exhibit the highest prompt sensitivity. This indicates that low-order interactions encoded by LLMs are highly stable when faced with subtle changes to prompt templates, i.e., simple interaction patterns are more robust. Conversely, the high sensitivity of high-order interactions reveals that the LLMs' internal representation of complex patterns is highly unstable.

Model Set 2  
![](images/e8206e8ce97289a447aec35b91adaeb2cd625af6b8e808b96d0d6a5e3eff2b3c.jpg)  
Model Set 1

![](images/9de2776a38b3540375046a66309179638196d4a89a7828fd2f0d5be97c352f65.jpg)  
Figure 20. A comparison of the prompt sensitivity of three order types. Results show that low-order interactions are the least sensitive, while high-order interactions are the most sensitive.

## F.5. More Results on Relative Change in the Prompt Sensitivity of Low-, Mid-, and High-Order Interactions for Different Factors.

(a) The relative change of Instruct/Chat Model compared to Base Model  
![](images/1e48a4c1b0a4525c613d0357c4452e7615f81fa5e9cae42459ace9d61676ef43.jpg)  
(c) The relative change of Few-shot learning compared to 0-shot learning

(b) The relative change of Dense Model compared to MoE Model  
![](images/6115fa29f709eadb0abfc223afa0f492864eb3c0671fbe8cc12ba89ef48823d6.jpg)

![](images/c36ab9316f64e9fad925a5d49dacf0465d2b68ca721d700f484df0a942d3d4e7.jpg)  
Figure 21. Comparing the relative change in the prompt sensitivity of low-, mid-, and high-order interactions for different factors.

F.6. More Results on the Prompt Sensitivity of Different Order Types across Different Model Scales.

![](images/599a82864b850852789e0de430c17c96e2aaeaaa7e04a2ee4a2c1e2c48403f1c.jpg)  
Figure 22. A comparison of prompt sensitivity of different order types across different model scales.

F.7. More Results on the Prompt Sensitivity of Different Orders for Each Individual LLM when Applying Few-Shot Learning  
![](images/ede743a956d517bbbe92d7012765d82235faec1918c01879aa92ce9dccf32c52.jpg)  
Figure 23. A comparison of prompt sensitivity of low-, mid-, and high-order interactions between 0-shot learning and few-shot learning. Prompt sensitivity at all three order levels shows an clear drop when applying few-shot learning.

## F.8. Hyperparameter Experiments of the threshold τ

To rigorously evaluate the robustness of our Interaction-based Prompt Sensitivity (IPS) metric, we conducted a hyperparameter sweep on the threshold τ.

## F.8.1. DETAILED MODEL RANKINGS UNDER VARYING THRESHOLDS

We aggregated the ranking and scoring consistency across all 16 thresholds using five metrics. As shown in Table 2, the high correlation coefficients and low error rates demonstrate that the IPS metric is highly robust to the choice of τ.

Table 2. Summary of consistency metrics across 16 different τ thresholds (0.05–0.20). The high values in correlation metrics and low RMSE indicate that the relative ranking of model sensitivity remains stable regardless of the specific threshold used.
<table><tr><td>Metric</td><td>Value</td><td>Interpretation</td></tr><tr><td>Spearman&#x27;s ρ</td><td>0.9905</td><td>Rank Correlation: Measures the average similarity of the overall ranking trends. A value close to 1.0 indicates near-perfect monotonic consistency.</td></tr><tr><td>Pearson&#x27;s r</td><td>0.9957</td><td>Linearity: Measures the linear correlation of the raw IPS scores, indicating that the scale of sensitivity shifts linearly across thresholds.</td></tr><tr><td>Kendall&#x27;s τ</td><td>0.9465</td><td>Pairwise Consistency: Indicates the probability that any pair of models maintains their relative order (better/worse)</td></tr><tr><td>RMSE</td><td>1.72</td><td>across different thresholds. Ranking Stability: On average, a model&#x27;s rank fluctu- ates by only ±1.72 positions across different threshold</td></tr><tr><td>Top-10 Overlap 96.7%</td><td></td><td>settings. SOTA Stability: The set of the top-10 most stable models remains 96.7% identical, ensuring reliable identification of the best-performing models.</td></tr></table>

To provide a granular view of robustness, Table 3 details the IPS scores across 10 distinct thresholds ranging from τ = 0.05 to τ = 0.20. Models are sorted based on their stability at the baseline threshold τ = 0.05. The data reveals that while absolute scores fluctuate, the relative ranking of model stability remains highly consistent.

Table 3. Detailed IPS scores for 50 LLMs across 10 different thresholds. The consistency in color gradients (implied by values) across rows confirms the robustness of the metric.
<table><tr><td rowspan="2">Model Name</td><td colspan="10">IPS Score (↓) at Threshold τ</td></tr><tr><td>0.05</td><td>0.06</td><td>0.07</td><td>0.08</td><td>0.09</td><td>0.10</td><td>0.12</td><td>0.15</td><td>0.18</td><td>0.20</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>1.328</td><td>1.312</td><td>1.297</td><td>1.286</td><td>1.276</td><td>1.268</td><td>1.255</td><td>1.240</td><td>1.228</td><td>1.222</td></tr><tr><td>Qwen2-72B-Instruct</td><td>1.362</td><td>1.346</td><td>1.333</td><td>1.322</td><td>1.312</td><td>1.304</td><td>1.290</td><td>1.278</td><td>1.271</td><td>1.265</td></tr><tr><td>Qwen2-7B-Instruct</td><td>1.411</td><td>1.400</td><td>1.391</td><td>1.383</td><td>1.376</td><td>1.368</td><td>1.357</td><td>1.341</td><td>1.327</td><td>1.318</td></tr><tr><td>Qwen3-32B-Instruct</td><td>1.424</td><td>1.419</td><td>1.415</td><td>1.411</td><td>1.408</td><td>1.406</td><td>1.403</td><td>1.401</td><td>1.396</td><td>1.395</td></tr><tr><td>Qwen3-14B-Instruct</td><td>1.443</td><td>1.438</td><td>1.434</td><td>1.429</td><td>1.427</td><td>1.425</td><td>1.419</td><td>1.415</td><td>1.412</td><td>1.407</td></tr><tr><td>Llama-2-70B-Chat</td><td>1.450</td><td>1.446</td><td>1.441</td><td>1.438</td><td>1.436</td><td>1.433</td><td>1.428</td><td>1.420</td><td>1.408</td><td>1.398</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td>1.463</td><td>1.454</td><td>1.446</td><td>1.440</td><td>1.434</td><td>1.430</td><td>1.423</td><td>1.415</td><td>1.411</td><td>1.409</td></tr><tr><td>Llama-3-8B-Instruct</td><td>1.486</td><td>1.483</td><td>1.480</td><td>1.477</td><td>1.475</td><td>1.473</td><td>1.467</td><td>1.460</td><td>1.447</td><td>1.436</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>1.490</td><td>1.481</td><td>1.473</td><td>1.465</td><td>1.459</td><td>1.454</td><td>1.446</td><td>1.436</td><td>1.430</td><td>1.425</td></tr><tr><td>Qwen3-4B-Instruct</td><td>1.496</td><td>1.493</td><td>1.490</td><td>1.488</td><td>1.485</td><td>1.484</td><td>1.480</td><td>1.473</td><td>1.470</td><td>1.464</td></tr><tr><td>Qwen2-1.5B-Instruct</td><td>1.497</td><td>1.494</td><td>1.492</td><td>1.487</td><td>1.483</td><td>1.477</td><td>1.465</td><td>1.447</td><td>1.429</td><td>1.416</td></tr><tr><td>Llama-2-70B</td><td>1.505</td><td>1.504</td><td>1.504</td><td>1.505</td><td>1.505</td><td>1.507</td><td>1.509</td><td>1.516</td><td>1.523</td><td>1.527</td></tr><tr><td>Qwen3-8B-Instruct</td><td>1.515</td><td>1.511</td><td>1.508</td><td>1.505</td><td>1.502</td><td>1.499</td><td>1.494</td><td>1.491</td><td>1.491</td><td>1.490</td></tr></table>

Continued on next page

Table 3 – continued from previous page
<table><tr><td rowspan=1 colspan=11>IPS Score (↓) at Threshold τ</td></tr><tr><td rowspan=1 colspan=11>Model Name                    0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.15 0.18 0.20</td></tr><tr><td rowspan=1 colspan=2>Qwen3-8B                       1.553</td><td rowspan=1 colspan=1>1.554</td><td rowspan=1 colspan=1>1.554</td><td rowspan=1 colspan=1>1.554</td><td rowspan=1 colspan=1>1.554</td><td rowspan=1 colspan=1>1.555</td><td rowspan=1 colspan=1>1.557</td><td rowspan=1 colspan=1>1.558</td><td rowspan=1 colspan=1>1.557</td><td rowspan=1 colspan=1>1.555</td></tr><tr><td rowspan=1 colspan=1>Qwen3-4B</td><td rowspan=1 colspan=1>1.555</td><td rowspan=1 colspan=1>1.562</td><td rowspan=1 colspan=1>1.566</td><td rowspan=1 colspan=1>1.570</td><td rowspan=1 colspan=1>1.573</td><td rowspan=1 colspan=1>1.575</td><td rowspan=1 colspan=1>1.578</td><td rowspan=1 colspan=1>1.579</td><td rowspan=1 colspan=1>1.574</td><td rowspan=1 colspan=1>1.572</td></tr><tr><td rowspan=1 colspan=1>Llama-2-13B-Chat</td><td rowspan=1 colspan=1>1.559</td><td rowspan=1 colspan=1>1.555</td><td rowspan=1 colspan=1>1.551</td><td rowspan=1 colspan=1>1.547</td><td rowspan=1 colspan=1>1.544</td><td rowspan=1 colspan=1>1.541</td><td rowspan=1 colspan=1>1.535</td><td rowspan=1 colspan=1>1.528</td><td rowspan=1 colspan=1>1.520</td><td rowspan=1 colspan=1>1.514</td></tr><tr><td rowspan=1 colspan=1>Qwen3-14B</td><td rowspan=1 colspan=1>1.559</td><td rowspan=1 colspan=1>1.558</td><td rowspan=1 colspan=1>1.557</td><td rowspan=1 colspan=1>1.556</td><td rowspan=1 colspan=1>1.556</td><td rowspan=1 colspan=1>1.555</td><td rowspan=1 colspan=1>1.553</td><td rowspan=1 colspan=1>1.552</td><td rowspan=1 colspan=1>1.549</td><td rowspan=1 colspan=1>1.548</td></tr><tr><td rowspan=1 colspan=1>Qwen2-7B</td><td rowspan=1 colspan=1>1.570</td><td rowspan=1 colspan=1>1.574</td><td rowspan=1 colspan=1>1.576</td><td rowspan=1 colspan=1>1.577</td><td rowspan=1 colspan=1>1.578</td><td rowspan=1 colspan=1>1.578</td><td rowspan=1 colspan=1>1.579</td><td rowspan=1 colspan=1>1.580</td><td rowspan=1 colspan=1>1.574</td><td rowspan=1 colspan=1>1.569</td></tr><tr><td rowspan=1 colspan=1>Qwen3-30B-A3B-Instruct</td><td rowspan=1 colspan=1>1.575</td><td rowspan=1 colspan=1>1.575</td><td rowspan=1 colspan=1>1.576</td><td rowspan=1 colspan=1>1.577</td><td rowspan=1 colspan=1>1.579</td><td rowspan=1 colspan=1>1.580</td><td rowspan=1 colspan=1>1.582</td><td rowspan=1 colspan=1>1.583</td><td rowspan=1 colspan=1>1.583</td><td rowspan=1 colspan=1>1.582</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-7B-Instruct</td><td rowspan=1 colspan=1>1.580</td><td rowspan=1 colspan=1>1.575</td><td rowspan=1 colspan=1>1.572</td><td rowspan=1 colspan=1>1.569</td><td rowspan=1 colspan=1>1.567</td><td rowspan=1 colspan=1>1.565</td><td rowspan=1 colspan=1>1.564</td><td rowspan=1 colspan=1>1.562</td><td rowspan=1 colspan=1>1.563</td><td rowspan=1 colspan=1>1.563</td></tr><tr><td rowspan=1 colspan=1>Llama-2-13B</td><td rowspan=1 colspan=1>1.592</td><td rowspan=1 colspan=1>1.602</td><td rowspan=1 colspan=1>1.610</td><td rowspan=1 colspan=1>1.617</td><td rowspan=1 colspan=1>1.623</td><td rowspan=1 colspan=1>1.628</td><td rowspan=1 colspan=1>1.636</td><td rowspan=1 colspan=1>1.645</td><td rowspan=1 colspan=1>1.653</td><td rowspan=1 colspan=1>1.655</td></tr><tr><td rowspan=1 colspan=1>Olmo-1B</td><td rowspan=1 colspan=1>1.601</td><td rowspan=1 colspan=1>1.610</td><td rowspan=1 colspan=1>1.617</td><td rowspan=1 colspan=1>1.623</td><td rowspan=1 colspan=1>1.628</td><td rowspan=1 colspan=1>1.632</td><td rowspan=1 colspan=1>1.639</td><td rowspan=1 colspan=1>1.647</td><td rowspan=1 colspan=1>1.652</td><td rowspan=1 colspan=1>1.655</td></tr><tr><td rowspan=1 colspan=1>Olmo-7B-Instruct</td><td rowspan=1 colspan=1>1.604</td><td rowspan=1 colspan=1>1.599</td><td rowspan=1 colspan=1>1.595</td><td rowspan=1 colspan=1>1.592</td><td rowspan=1 colspan=1>1.589</td><td rowspan=1 colspan=1>1.587</td><td rowspan=1 colspan=1>1.583</td><td rowspan=1 colspan=1>1.579</td><td rowspan=1 colspan=1>1.578</td><td rowspan=1 colspan=1>1.576</td></tr><tr><td rowspan=1 colspan=1>Llama-3-8B</td><td rowspan=1 colspan=1>1.618</td><td rowspan=1 colspan=1>1.628</td><td rowspan=1 colspan=1>1.637</td><td rowspan=1 colspan=1>1.644</td><td rowspan=1 colspan=1>1.650</td><td rowspan=1 colspan=1>1.655</td><td rowspan=1 colspan=1>1.664</td><td rowspan=1 colspan=1>1.674</td><td rowspan=1 colspan=1>1.679</td><td rowspan=1 colspan=1>1.682</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-3B-Instruct</td><td rowspan=1 colspan=1>1.622</td><td rowspan=1 colspan=1>1.618</td><td rowspan=1 colspan=1>1.615</td><td rowspan=1 colspan=1>1.611</td><td rowspan=1 colspan=1>1.608</td><td rowspan=1 colspan=1>1.606</td><td rowspan=1 colspan=1>1.601</td><td rowspan=1 colspan=1>1.595</td><td rowspan=1 colspan=1>1.590</td><td rowspan=1 colspan=1>1.585</td></tr><tr><td rowspan=1 colspan=1>Olmo-2-13B-Instruct</td><td rowspan=1 colspan=1>1.625</td><td rowspan=1 colspan=1>1.627</td><td rowspan=1 colspan=1>1.628</td><td rowspan=1 colspan=1>1.629</td><td rowspan=1 colspan=1>1.630</td><td rowspan=1 colspan=1>1.631</td><td rowspan=1 colspan=1>1.632</td><td rowspan=1 colspan=1>1.633</td><td rowspan=1 colspan=1>1.632</td><td rowspan=1 colspan=1>1.631</td></tr><tr><td rowspan=1 colspan=1>Olmo-2-7B-Instruct</td><td rowspan=1 colspan=1>1.626</td><td rowspan=1 colspan=1>1.627</td><td rowspan=1 colspan=1>1.627</td><td rowspan=1 colspan=1>1.628</td><td rowspan=1 colspan=1>1.629</td><td rowspan=1 colspan=1>1.629</td><td rowspan=1 colspan=1>1.629</td><td rowspan=1 colspan=1>1.627</td><td rowspan=1 colspan=1>1.628</td><td rowspan=1 colspan=1>1.626</td></tr><tr><td rowspan=1 colspan=1>Olmo-2-32B-Instruct</td><td rowspan=1 colspan=1>1.626</td><td rowspan=1 colspan=1>1.626</td><td rowspan=1 colspan=1>1.625</td><td rowspan=1 colspan=1>1.624</td><td rowspan=1 colspan=1>1.623</td><td rowspan=1 colspan=1>1.623</td><td rowspan=1 colspan=1>1.622</td><td rowspan=1 colspan=1>1.620</td><td rowspan=1 colspan=1>1.619</td><td rowspan=1 colspan=1>1.618</td></tr><tr><td rowspan=1 colspan=1>Qwen3-30B-A3B</td><td rowspan=1 colspan=1>1.629</td><td rowspan=1 colspan=1>1.631</td><td rowspan=1 colspan=1>1.634</td><td rowspan=1 colspan=1>1.637</td><td rowspan=1 colspan=1>1.639</td><td rowspan=1 colspan=1>1.642</td><td rowspan=1 colspan=1>1.646</td><td rowspan=1 colspan=1>1.651</td><td rowspan=1 colspan=1>1.655</td><td rowspan=1 colspan=1>1.656</td></tr><tr><td rowspan=1 colspan=1>Mistral-7B-v0.3-Instruct</td><td rowspan=1 colspan=1>1.630</td><td rowspan=1 colspan=1>1.629</td><td rowspan=1 colspan=1>1.629</td><td rowspan=1 colspan=1>1.629</td><td rowspan=1 colspan=1>1.629</td><td rowspan=1 colspan=1>1.631</td><td rowspan=1 colspan=1>1.633</td><td rowspan=1 colspan=1>1.638</td><td rowspan=1 colspan=1>1.641</td><td rowspan=1 colspan=1>1.644</td></tr><tr><td rowspan=1 colspan=1>InternLM2-Chat-7B</td><td rowspan=1 colspan=1>1.634</td><td rowspan=1 colspan=1>1.639</td><td rowspan=1 colspan=1>1.641</td><td rowspan=1 colspan=1>1.644</td><td rowspan=1 colspan=1>1.647</td><td rowspan=1 colspan=1>1.650</td><td rowspan=1 colspan=1>1.656</td><td rowspan=1 colspan=1>1.660</td><td rowspan=1 colspan=1>1.664</td><td rowspan=1 colspan=1>1.667</td></tr><tr><td rowspan=1 colspan=1>Qwen2-0.5B-Instruct</td><td rowspan=1 colspan=1>1.639</td><td rowspan=1 colspan=1>1.650</td><td rowspan=1 colspan=1>1.658</td><td rowspan=1 colspan=1>1.663</td><td rowspan=1 colspan=1>1.666</td><td rowspan=1 colspan=1>1.669</td><td rowspan=1 colspan=1>1.669</td><td rowspan=1 colspan=1>1.665</td><td rowspan=1 colspan=1>1.658</td><td rowspan=1 colspan=1>1.651</td></tr><tr><td rowspan=1 colspan=1>Mixtral-8x7B-Instruct</td><td rowspan=1 colspan=1>1.641</td><td rowspan=1 colspan=1>1.638</td><td rowspan=1 colspan=1>1.636</td><td rowspan=1 colspan=1>1.633</td><td rowspan=1 colspan=1>1.632</td><td rowspan=1 colspan=1>1.631</td><td rowspan=1 colspan=1>1.629</td><td rowspan=1 colspan=1>1.628</td><td rowspan=1 colspan=1>1.629</td><td rowspan=1 colspan=1>1.630</td></tr><tr><td rowspan=1 colspan=1>Qwen1.5-MoE-A2.7B</td><td rowspan=1 colspan=1>1.645</td><td rowspan=1 colspan=1>1.658</td><td rowspan=1 colspan=1>1.668</td><td rowspan=1 colspan=1>1.675</td><td rowspan=1 colspan=1>1.682</td><td rowspan=1 colspan=1>1.687</td><td rowspan=1 colspan=1>1.696</td><td rowspan=1 colspan=1>1.705</td><td rowspan=1 colspan=1>1.712</td><td rowspan=1 colspan=1>1.717</td></tr><tr><td rowspan=1 colspan=1>Qwen1.5-MoE-A2.7B-Chat</td><td rowspan=1 colspan=1>1.648</td><td rowspan=1 colspan=1>1.654</td><td rowspan=1 colspan=1>1.657</td><td rowspan=1 colspan=1>1.660</td><td rowspan=1 colspan=1>1.663</td><td rowspan=1 colspan=1>1.665</td><td rowspan=1 colspan=1>1.667</td><td rowspan=1 colspan=1>1.668</td><td rowspan=1 colspan=1>1.664</td><td rowspan=1 colspan=1>1.660</td></tr><tr><td rowspan=1 colspan=1>Qwen3-1.7B-Instruct</td><td rowspan=1 colspan=1>1.648</td><td rowspan=1 colspan=1>1.645</td><td rowspan=1 colspan=1>1.642</td><td rowspan=1 colspan=1>1.641</td><td rowspan=1 colspan=1>1.640</td><td rowspan=1 colspan=1>1.639</td><td rowspan=1 colspan=1>1.639</td><td rowspan=1 colspan=1>1.637</td><td rowspan=1 colspan=1>1.638</td><td rowspan=1 colspan=1>1.637</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-0.5B-Instruct</td><td rowspan=1 colspan=1>1.652</td><td rowspan=1 colspan=1>1.663</td><td rowspan=1 colspan=1>1.671</td><td rowspan=1 colspan=1>1.677</td><td rowspan=1 colspan=1>1.681</td><td rowspan=1 colspan=1>1.684</td><td rowspan=1 colspan=1>1.688</td><td rowspan=1 colspan=1>1.689</td><td rowspan=1 colspan=1>1.687</td><td rowspan=1 colspan=1>1.685</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-1.5B-Instruct</td><td rowspan=1 colspan=1>1.658</td><td rowspan=1 colspan=1>1.663</td><td rowspan=1 colspan=1>1.667</td><td rowspan=1 colspan=1>1.671</td><td rowspan=1 colspan=1>1.674</td><td rowspan=1 colspan=1>1.676</td><td rowspan=1 colspan=1>1.679</td><td rowspan=1 colspan=1>1.684</td><td rowspan=1 colspan=1>1.686</td><td rowspan=1 colspan=1>1.689</td></tr><tr><td rowspan=1 colspan=1>Llama-MoE-v2-3_8B-2_8-SFT</td><td rowspan=1 colspan=1>1.665</td><td rowspan=1 colspan=1>1.669</td><td rowspan=1 colspan=1>1.673</td><td rowspan=1 colspan=1>1.676</td><td rowspan=1 colspan=1>1.678</td><td rowspan=1 colspan=1>1.681</td><td rowspan=1 colspan=1>1.685</td><td rowspan=1 colspan=1>1.691</td><td rowspan=1 colspan=1>1.696</td><td rowspan=1 colspan=1>1.698</td></tr><tr><td rowspan=1 colspan=1>Llama-2-7B-Chat</td><td rowspan=1 colspan=1>1.670</td><td rowspan=1 colspan=1>1.677</td><td rowspan=1 colspan=1>1.683</td><td rowspan=1 colspan=1>1.687</td><td rowspan=1 colspan=1>1.690</td><td rowspan=1 colspan=1>1.691</td><td rowspan=1 colspan=1>1.692</td><td rowspan=1 colspan=1>1.693</td><td rowspan=1 colspan=1>1.691</td><td rowspan=1 colspan=1>1.689</td></tr><tr><td rowspan=1 colspan=1>Olmo-7B</td><td rowspan=1 colspan=1>1.671</td><td rowspan=1 colspan=1>1.679</td><td rowspan=1 colspan=1>1.685</td><td rowspan=1 colspan=1>1.690</td><td rowspan=1 colspan=1>1.695</td><td rowspan=1 colspan=1>1.699</td><td rowspan=1 colspan=1>1.706</td><td rowspan=1 colspan=1>1.714</td><td rowspan=1 colspan=1>1.718</td><td rowspan=1 colspan=1>1.721</td></tr><tr><td rowspan=1 colspan=1>Mixtral-8x7B</td><td rowspan=1 colspan=1>1.677</td><td rowspan=1 colspan=1>1.682</td><td rowspan=1 colspan=1>1.686</td><td rowspan=1 colspan=1>1.689</td><td rowspan=1 colspan=1>1.692</td><td rowspan=1 colspan=1>1.695</td><td rowspan=1 colspan=1>1.701</td><td rowspan=1 colspan=1>1.709</td><td rowspan=1 colspan=1>1.715</td><td rowspan=1 colspan=1>1.720</td></tr><tr><td rowspan=1 colspan=1>Olmoe-7B</td><td rowspan=1 colspan=1>1.680</td><td rowspan=1 colspan=1>1.689</td><td rowspan=1 colspan=1>1.696</td><td rowspan=1 colspan=1>1.702</td><td rowspan=1 colspan=1>1.707</td><td rowspan=1 colspan=1>1.711</td><td rowspan=1 colspan=1>1.720</td><td rowspan=1 colspan=1>1.733</td><td rowspan=1 colspan=1>1.742</td><td rowspan=1 colspan=1>1.746</td></tr><tr><td rowspan=1 colspan=1>Qwen3-0.6B-Instruct</td><td rowspan=1 colspan=1>1.682</td><td rowspan=1 colspan=1>1.678</td><td rowspan=1 colspan=1>1.675</td><td rowspan=1 colspan=1>1.673</td><td rowspan=1 colspan=1>1.671</td><td rowspan=1 colspan=1>1.668</td><td rowspan=1 colspan=1>1.665</td><td rowspan=1 colspan=1>1.659</td><td rowspan=1 colspan=1>1.654</td><td rowspan=1 colspan=1>1.653</td></tr><tr><td rowspan=1 colspan=1>Llama-2-7B</td><td rowspan=1 colspan=1>1.682</td><td rowspan=1 colspan=1>1.692</td><td rowspan=1 colspan=1>1.699</td><td rowspan=1 colspan=1>1.705</td><td rowspan=1 colspan=1>1.711</td><td rowspan=1 colspan=1>1.716</td><td rowspan=1 colspan=1>1.724</td><td rowspan=1 colspan=1>1.733</td><td rowspan=1 colspan=1>1.740</td><td rowspan=1 colspan=1>1.745</td></tr><tr><td rowspan=1 colspan=1>Olmo-2-1B-Instruct</td><td rowspan=1 colspan=1>1.698</td><td rowspan=1 colspan=1>1.710</td><td rowspan=1 colspan=1>1.719</td><td rowspan=1 colspan=1>1.727</td><td rowspan=1 colspan=1>1.734</td><td rowspan=1 colspan=1>1.739</td><td rowspan=1 colspan=1>1.747</td><td rowspan=1 colspan=1>1.756</td><td rowspan=1 colspan=1>1.760</td><td rowspan=1 colspan=1>1.760</td></tr><tr><td rowspan=1 colspan=1>InternLM2-7B</td><td rowspan=1 colspan=1>1.706</td><td rowspan=1 colspan=1>1.711</td><td rowspan=1 colspan=1>1.716</td><td rowspan=1 colspan=1>1.720</td><td rowspan=1 colspan=1>1.724</td><td rowspan=1 colspan=1>1.728</td><td rowspan=1 colspan=1>1.734</td><td rowspan=1 colspan=1>1.743</td><td rowspan=1 colspan=1>1.751</td><td rowspan=1 colspan=1>1.758</td></tr><tr><td rowspan=1 colspan=1>Llama-MoE-v1-3_5B-2_8-SFT</td><td rowspan=1 colspan=1>1.715</td><td rowspan=1 colspan=1>1.720</td><td rowspan=1 colspan=1>1.724</td><td rowspan=1 colspan=1>1.728</td><td rowspan=1 colspan=1>1.732</td><td rowspan=1 colspan=1>1.735</td><td rowspan=1 colspan=1>1.743</td><td rowspan=1 colspan=1>1.753</td><td rowspan=1 colspan=1>1.761</td><td rowspan=1 colspan=1>1.765</td></tr><tr><td rowspan=1 colspan=1>Olmoe-7B-Instruct</td><td rowspan=1 colspan=1>1.716</td><td rowspan=1 colspan=1>1.717</td><td rowspan=1 colspan=1>1.719</td><td rowspan=1 colspan=1>1.721</td><td rowspan=1 colspan=1>1.722</td><td rowspan=1 colspan=1>1.723</td><td rowspan=1 colspan=1>1.724</td><td rowspan=1 colspan=1>1.726</td><td rowspan=1 colspan=1>1.730</td><td rowspan=1 colspan=1>1.732</td></tr><tr><td rowspan=1 colspan=1>Mistral-7B-v0.3</td><td rowspan=1 colspan=1>1.720</td><td rowspan=1 colspan=1>1.729</td><td rowspan=1 colspan=1>1.737</td><td rowspan=1 colspan=1>1.743</td><td rowspan=1 colspan=1>1.748</td><td rowspan=1 colspan=1>1.752</td><td rowspan=1 colspan=1>1.760</td><td rowspan=1 colspan=1>1.768</td><td rowspan=1 colspan=1>1.775</td><td rowspan=1 colspan=1>1.779</td></tr></table>

## F.8.2. VERIFYING THE GENERALIZABILITY OF THE METHODS AND CONCLUSIONS ON DIFFERENT THRESHOLD τ.

In Sections 4.2 and 4.3, we set the threshold τ to 0.1 to distinguish salient interactions from noise. This threshold directly influences the proportion of interactions classified as salient interactions. A higher τ value usually generates a smaller set of salient interactions with more significant effects. Li & Zhang (2023) conducted experiments which show that conclusions are not sensitive to the choice of τ. Our choice of τ is guided by the empirical sparsity of interactions. Figure 2 (a) shows a sharp “elbow" in the distribution of interaction effects, clearly separating a small set of high-magnitude salient interactions from a long tail of near-zero noise interactions. A threshold τ chosen from the range of 0.05 to 0.15 effectively captures this salient set, satisfying the sparsity assumption without being overly restrictive. To ensure the robustness of our findings. we conducted hyperparameter experiments with τ = 0.05 and τ = 0.15. Our main conclusions remain consistent across different threshold values.

Here are the results for τ = 0.15.  
![](images/c7f8c57a624b64c5ff3080d33618c068cf151c01d5441966288d0b788a3f0faf.jpg)  
Figure 24. A comparison of the prompt sensitivity between instruct/chat models and base models. Results show that instruct/chat models are less sensitive than corresponding base models.

![](images/eccb19f16f505b3f3947cc116fac1afea63068570b66e90f1ce1040f6919936b.jpg)  
Figure 25. A comparison of prompt sensitivity across different model scales. As the model scale increases, the prompt sensitivity within a model series systematically decreases.

![](images/336ace6de2ea50454147a09880686e10c99297077c14ab8ab67a5925ca32401f.jpg)  
Figure 26. A Comparison of prompt sensitivity between MoE models and dense models. Generally, MoE models tend to be more sensitive than dense models in the same model family.

![](images/73294134fe16e2883e907358c9b33d8206be5fe2ed9f752af1d95d7dbbefd6d1.jpg)  
Figure 27. A comparison of prompt sensitivity between 0-shot learning and few-shot learning. The drop in prompt sensitivity is substantial from 0-shot to 1-shot.

(a) The relative change of Instruct/Chat Model compared to Base Model  
![](images/420500d4944f1687a9478b4718609e8dc2c6006f4db7d254f56a02a7d819c0ee.jpg)

(b) The relative change of Dense Model compared to MoE Model  
![](images/56e3fe3e4d34e821defdab308ab0eef2d5f9effe86519df6471a2282d7eb5398.jpg)

(c) The relative change of Few-shot learning compared to 0-shot learning  
![](images/43429fb01c737f3e05bf752bdca75919fa0ef94b31b89d59a71c16707a9eb152.jpg)  
Figure 28. Comparing the relative change in the prompt sensitivity of low-, mid-, and high-order interactions for different factors.

Model Set 2  
![](images/eee546b01900b3f1d69b03a355049c416de89e38cadee7594aa2c76b75c410f9.jpg)  
Model Set 1

![](images/4505c6d9113ae9a9bde93715f2d931d5a4338a76949f7ac92b525410925bf80a.jpg)  
Figure 29. A comparison of the prompt sensitivity of three order types. Results show that low-order interactions are the least sensitive, while high-order interactions are the most sensitive.

Tip: within each model series, lighter color represents smaller model scale  
![](images/3bcea4f37f2d0fe7e1cbf746859172f9ae7ee53ae417b32984ba45c17a2198ef.jpg)  
Figure 30. A comparison of prompt sensitivity at the order-level across different model scales.

Here are the results for τ = 0.05.  
![](images/ced8375d0adff81607328c6b6631862f22182e05a0d1921de73792e4b1e57219.jpg)  
Figure 31. A comparison of the prompt sensitivity between instruct/chat models and base models. Results show that instruct/chat models are less sensitive than corresponding base models.

Tip: within each model series, lighter color represents smaller model scale  
![](images/b8e06e3e7af697348ed9ba607b0438038595667a8e8070158a1ed3f809c0fbee.jpg)  
Figure 32. A comparison of prompt sensitivity across different model scales. As the model scale increases, the prompt sensitivity within a model series systematically decreases.

![](images/eaa0c6260702c2289d22675153e9a876021ed14b4fe4648f514ef53e940d44cd.jpg)  
Figure 33. A Comparison of prompt sensitivity between MoE models and dense models. Generally, MoE models tend to be more sensitive than dense models in the same model family.

![](images/e0d17fb7d488a2ce629dfb710caeeb924b25960e87d380ce32b8b5074deb62c9.jpg)  
Figure 34. A comparison of prompt sensitivity between 0-shot learning and few-shot learning. The drop in prompt sensitivity is substantial from 0-shot to 1-shot.

(a) The relative change of Instruct/Chat Model compared to Base Model  
![](images/e641a47c22535c3a1560e25500610edea4ac97c8001155d86c61114b3db60136.jpg)

(b) The relative change of Dense Model compared to MoE Model  
![](images/bc84bed96b80b8753a8aa268dc9fc411b43dcbce5aac2207a92c2d490b7d38cc.jpg)

(c) The relative change of Few-shot learning compared to 0-shot learning  
![](images/aad854014738ae27d9b2fad5f9afe5d77aa80d0a29199910f0f287320ca95357.jpg)  
Figure 35. Comparing the relative change in the prompt sensitivity of low-, mid-, and high-order interactions for different factors.

Model Set 2  
![](images/a6ee539be06b746cebd1a3985dbd7a6036897d18214331bb42f10523fee84e72.jpg)

![](images/63fa06baff242be3d749b5bed928fcb31eab28f1e36393cc7ec73ee10e587855.jpg)

Figure 36. A comparison of the prompt sensitivity of three order types. Results show that low-order interactions are the least sensitive, while high-order interactions are the most sensitive.  
![](images/3bfe50a10c76aa973db91e869430053092705466ae800b8a01d2b9debc440c8c.jpg)  
Figure 37. A comparison of prompt sensitivity at the order-level across different model scales.

## F.9. Results on MMLU Dataset

Here are the results on the MMLU dataset and τ is set to 0.1, we can observe the same conclusions on this dataset.

![](images/73e8146e361d0fa43b2510c2edebb96ec176c6a9a3d624e73d3967caa682bd2e.jpg)  
Figure 38. A comparison of the prompt sensitivity between instruct/chat models and base models. Results show that instruct/chat models are less sensitive than corresponding base models.

![](images/817e99279b3c5fd2ae769d3ab441051e885bbe83f81dcc237bcb782b998e18a0.jpg)  
Figure 39. A comparison of prompt sensitivity across different model scales. As the model scale increases, the prompt sensitivity within a model series systematically decreases.

![](images/43e50adc7311643c27ed844d57ae1e88dd08da49bca3be02d8f0a466d0d2cb98.jpg)  
Figure 40. A Comparison of prompt sensitivity between MoE models and dense models. Generally, MoE models tend to be more sensitive than dense models in the same model family.

![](images/cc7e51ec8f1c51e1b49f00c459cc7858699f54aa964e5aa566d0588cb5c22de6.jpg)  
Figure 41. A comparison of prompt sensitivity between 0-shot learning and few-shot learning. The drop in prompt sensitivity is substantial from 0-shot to 1-shot.

(a) The relative change of Instruct/Chat Model compared to Base Model  
![](images/09b47e100c08f2842f0baab666da188379b291aeed1c90283881e42587ee4b5c.jpg)

(b) The relative change of Dense Model compared to MoE Model  
![](images/6c9f46463a7daee81f038c72fa8a54a064345f627ebaf04e21adf78375f9819c.jpg)

(c) The relative change of Few-shot learning comparēd to 0-shot learning  
![](images/11b4014ca042e398b789d528e6268c7d54fbb5259c7dda76a877455b9f937ded.jpg)  
Figure 42. Comparing the relative change in the prompt sensitivity of low-, mid-, and high-order interactions for different factors.

Model Set 1  
Model Set 2  
![](images/c2f6d719d4d665785914c9a3e0c27ae8268ccd60a4b0bf61079de7f3cfb47741.jpg)

![](images/8c78fa7be0ea934e982eee6afd05b46e4a9a25d26cc91c5c6118a8c311035c79.jpg)

Figure 43. A comparison of the prompt sensitivity of three order types. Results show that low-order interactions are the least sensitive, while high-order interactions are the most sensitive.  
![](images/c0ab9a984fa5d77c80955a7b28b1972bbff6a95dcb4fe862058e67a258f1b1dc.jpg)  
Figure 44. A comparison of prompt sensitivity at the order-level across different model scales.

## G. Prompt Sensitivity of altering tokens vs. adding tokens

Disaggregating prompt sensitivity by the type of perturbation offers deeper insights into model behavior. Following this direction, we conduct a fine-grained decomposition of our experimental results, comparing prompt sensitivity of two distinct categories to prompt alterations:

1. Altering tokens: This involves modifying the capitalization of words, such as from "Answer" to "ANSWER ".

2. Adding tokens: This involves adding symbolic components, for instance, changing a colon from " : " to " : : ".

<table><tr><td>Dense Model</td><td>Altering Tokens</td><td>Adding Tokens</td><td>Difference</td></tr><tr><td rowspan="5">internlm2-chat-7b llama-2-13b-chat llama-2-7b-chat llama-3-8b-instruct mistral-7b-v0.3-instruct</td><td>1.673</td><td>1.676</td><td>+0.003</td></tr><tr><td>1.520</td><td>1.433</td><td>-0.087</td></tr><tr><td>1.633</td><td>1.710</td><td>+0.077</td></tr><tr><td>1.450</td><td>1.467</td><td>+0.017</td></tr><tr><td>1.538</td><td>1.635</td><td>+0.097</td></tr><tr><td rowspan="5">olmo-2-7b-instruct olmo-7b-instruct qwen2-7b-instruct</td><td>1.618</td><td>1.655</td><td>+0.037</td></tr><tr><td>1.462</td><td>1.496</td><td>+0.034</td></tr><tr><td>1.366</td><td>1.404</td><td>+0.038</td></tr><tr><td>1.558</td><td>1.567</td><td>+0.009</td></tr><tr><td>1.505</td><td>1.550</td><td>+0.045</td></tr></table>

Table 4. IPS Scores of Dense Models on Different Perturbation Types. Higher scores indicate greater sensitivity. The more sensitive perturbation type for each model is highlighted in bold.

<table><tr><td>MoE Model</td><td>Altering Tokens</td><td>Adding Tokens</td><td>Difference</td></tr><tr><td>llama-moe-v1-3.5b-sft</td><td>1.736</td><td>1.724</td><td>-0.012</td></tr><tr><td>llama-moe-v2-3.8b-sft</td><td>1.698</td><td>1.655</td><td>-0.043</td></tr><tr><td>mixtral-8x7b-instruct</td><td>1.658</td><td>1.617</td><td>-0.041</td></tr><tr><td>olmoe-7b-instruct</td><td>1.725</td><td>1.699</td><td>-0.026</td></tr><tr><td>qwen1.5-moe-a2.7b-chat</td><td>1.669</td><td>1.634</td><td>-0.035</td></tr><tr><td>qwen3-30b-a3b-instruct</td><td>1.608</td><td>1.638</td><td>+0.030</td></tr></table>

Table 5. IPS Scores of MoE Models on Different Perturbation Types. The more sensitive perturbation type for each model is highlighted in bold.

Our results highlight a clear architectural divide: (1) Dense models are more sensitive to adding tokens. As shown in Table 4, 9 out of the 10 analyzed dense models exhibit greater sensitivity to the "adding tokens" category. This suggests a strong, consistent trend where changes to the template's structure have a more pronounced impact on the internal interactions of dense architectures. (2) MoE models are more sensitive to altering tokens. In stark contrast, Table 5 shows that 5 out of the 6 MoE models are more sensitive to "altering tokens". This consistent pattern suggests that MoE architectures are more susceptible to variations in the change of word capitalization.

## H. Comparison between IPS and Other Metrics

To demonstrate the unique value of our interaction-based approach, we compare the Interaction-based Prompt Sensitivity (IPS) against standard coarse-grained metrics derived from internal representations. Specifically, we measure the Cosine Similarity and $L _ { 2 }$ Distance of the final-layer hidden states under prompt perturbations. While these metrics are commonly used to assess representation robustness, our analysis reveals that they fail to capture the nuanced mechanisms of prompt sensitivity in LLMs.

## H.1. Empirical Inconsistency of Representation-based Metrics

We re-evaluate Factor 1 (Base vs. Instruct/Chat models) using these representation-based metrics. The results, summarized in Table 6, demonstrate a significant lack of consistency compared to the robust trends observed via IPS.

As shown in Table $^ { 6 , }$ neither Cosine Similarity nor $L _ { 2 }$ Distance provides a reliable proxy for prompt sensitivity:

Table 6. Comparison of stability metrics (Cosine Similarity and $L _ { 2 }$ Distance) for Base vs. Instruct/Chat models. Unlike IPS, these metrics fail to show a consistent trend regarding the impact of Supervised Fine-Tuning.
<table><tr><td rowspan="2">Model Family</td><td colspan="2">Base Model</td><td colspan="2">Instruct/Chat Model</td></tr><tr><td>Cosine (↑)</td><td> $L _ { 2 }$  Dist. (↓)</td><td>Cosine (↑)</td><td> $L _ { 2 }$  Dist. (↓)</td></tr><tr><td>Llama-2-7B</td><td>0.770</td><td>75.12</td><td>0.781</td><td>66.51</td></tr><tr><td>Llama-2-13B</td><td>0.942</td><td>17.35</td><td>0.927</td><td>19.69</td></tr><tr><td>Llama-2-70B</td><td>0.911</td><td>33.65</td><td>0.795</td><td>43.14</td></tr><tr><td>Llama-3-8B</td><td>0.856</td><td>79.77</td><td>0.880</td><td>73.56</td></tr><tr><td>Mistral-7B-V0.3</td><td>0.811</td><td>207.89</td><td>0.843</td><td>154.43</td></tr><tr><td>Mixtral-8x7B</td><td>0.965</td><td>138.71</td><td>0.875</td><td>136.03</td></tr><tr><td>InternLM2-7B</td><td>0.898</td><td>142.51</td><td>0.987</td><td>75.83</td></tr><tr><td>Olmo-7B</td><td>0.793</td><td>36.88</td><td>0.661</td><td>48.33</td></tr><tr><td>Olmoe-7B</td><td>0.672</td><td>56.67</td><td>0.510</td><td>85.73</td></tr><tr><td>Qwen2-7B</td><td>0.864</td><td>138.73</td><td>0.888</td><td>136.49</td></tr><tr><td>Qwen3-4B</td><td>0.914</td><td>60.13</td><td>0.779</td><td>73.82</td></tr><tr><td>Qwen3-8B</td><td>0.995</td><td>33.95</td><td>0.962</td><td>41.42</td></tr><tr><td>Qwen3-14B</td><td>0.947</td><td>67.85</td><td>0.932</td><td>51.97</td></tr><tr><td>Qwen1.5-MoE-A2.7B</td><td>0.765</td><td>160.31</td><td>0.911</td><td>86.98</td></tr><tr><td>Qwen3-30B-A3B</td><td>0.864</td><td>81.13</td><td>0.932</td><td>39.04</td></tr></table>

• Contradictions between metrics: For models like Mixtral-8x7B, the two metrics contradict each other—Cosine Similarity suggests the Base model is more stable, while $L _ { 2 }$ Distance favors the Instruct model.

• Inconsistency with established trends: While our IPS analysis (and general consensus) identifies Instruct/Chat models as more robust to prompt variations, representation metrics frequently suggest the opposite. For instance, in the Llama-2-13B and Owen3-8B pairs, the Base models exhibit higher cosine similarity and lower $L _ { 2 }$ distance than their Instruct counterparts.

• Random fluctuations: There is no discernible pattern across model families. For Llama-2-7B, the Chat version appears more stable via Cosine Similarity but less stable via $L _ { 2 }$ Distance.

These contradictions indicate that global measures of hidden state changes are too coarse to serve as accurate indicators of the model's functional sensitivity.

## H.2. Superiority of the Interaction-based Framework

The empirical limitations of representation-based metrics highlight the theoretical advantages of our proposed framework. The superiority of IPS stems from two fundamental differences:

1. Explanability (“Why" vs. “What"): Hidden state similarity merely measures what has changed—the magnitude or direction of the aggregate internal representation vector. It treats the model as a black box regarding the reasoning process. In contrast, our interaction framework explains why the output fluctuates. By decomposing predictions into interactions, we can pinpoint specific combinations of input tokens (inference patterns) that become unstable. This fine-grained insight allows us to distinguish between benign representation shifts and those that disrupt the model's logical coherence.

2. Faithfulness to the Output: Representation metrics lack a direct mathematical link to the final prediction. A small shift in Euclidean distance can sometimes lead to a flipped prediction, while a large shift might not. Conversely, our method is grounded in the Universal Matching Property (Theorem 1). This theorem guarantees that the sum of all interactions perfectly reconstructs the LLM's output score. Consequently, IPS provides a faithful evaluation of the decision-making logic, ensuring that the measured sensitivity directly reflects the instability in the model's actual predictive mechanism.

## I. Detailed Discussion on the Impact of Model Architecture

We notice that in Figure 7, the behavior of Mistral family is different from the other family: the dense models are more sensitive than the MoE models at low-order and mid-order level. We attribute this to the number of activated parameters, extending our finding from Factor 2 that larger models are less sensitive. While most MoE models (e.g., in Llama and Qwen families) are more sensitive due to fewer activated parameters, the Mistral case is reversed: Mixtral-8x7B activates more parameters than its dense counterpart Mistral-7B-V0.3 (13B vs. 7B), resulting in lower prompt sensitivity.

To more rigorously isolate the influence of model architecture on the prompt sensitivity from model size or other factors, we conduct a controlled variable analysis. We select specific pairs from the Qwen and OLMo families that share the most similar model scales (i.e., activated parameters) and analogous training paradigms (i.e., base vs. base, instruct/chat vs. instruct/chat). This targeted comparison enables us to minimize confounding factors and focus directly on the architectural impact.

Results in Table 7 consistently show that MoE models exhibit higher prompt sensitivity than dense models. This suggests that the increased prompt sensitivity of MoE architectures is not merely a consequence of smaller number of activated parameters Instead, it further strengthens our conclusion that the MoE architecture inherently increases the prompt sensitivity of LLMs.

Table 7. Controlled variable analysis of prompt sensitivity between MoE and dense models.
<table><tr><td>Model Name</td><td>Architecture</td><td>Act. Params.</td><td>Type</td><td>IPS (ARC) ↓</td><td>IPS (MMLU) ↓</td></tr><tr><td rowspan="2">Qwen1.5-moe-a2.7b-chat Qwen2.5-3b-instruct</td><td>MoE</td><td>2.7B</td><td>Chat</td><td>1.665</td><td>1.640</td></tr><tr><td>Dense</td><td>3B</td><td>Instruct</td><td>1.606</td><td>1.625</td></tr><tr><td rowspan="2">Olmoe-7b Olmo-1b</td><td>MoE</td><td>1B</td><td>Base</td><td>1.714</td><td>1.717</td></tr><tr><td>Dense</td><td>1B</td><td>Base</td><td>1.632</td><td>1.658</td></tr><tr><td rowspan="2">Qwen3-30b-a3b Qwen3-4b</td><td>MoE</td><td>3B</td><td>Base</td><td>1.642</td><td>1.680</td></tr><tr><td>Dense</td><td>4B</td><td>Base</td><td>1.568</td><td>1.599</td></tr><tr><td rowspan="2">Qwen3-30b-a3b-instruct Qwen3-4b-instruct</td><td>MoE</td><td>3B</td><td>Instruct</td><td>1.580</td><td>1.617</td></tr><tr><td>Dense</td><td>4B</td><td>Instruct</td><td>1.483</td><td>1.551</td></tr></table>

## J. Solutions for reducing the computational cost of the method

The limitation of our current study lies in the computational cost of the interaction framework. The method's complexity scales exponentially with the number of input variables n, as it requires evaluating 2ⁿ masked inputs. However, applying this method to very long text inputs would demand a high computational load.

Future work can address this scalability challenge through several promising avenues. These strategies aim to reduce the effective number of input variables without fundamentally changing the faithfulness of the analysis:

(1) Selective Input Variable Analysis. One approach is to analyze only a subset of informative input variables (i.e., words) while treating uninformative ones (e.g., stop words) as fixed background context. Previous research has demonstrated that this selection does not significantly impair the faithfulness of the interaction framework (Chen et al., 2024).

(2) Phrase-level Aggregation. Instead of analyzing individual words, we can operate at a coarser perspective by merging related words into combined phrasal units. This reduces the total number of input variables while preserving key semantic meaning.

For methods (1) and (2), we have put them into practice. In our experiments on long-form open-ended questions, we apply these two methods to effectively control the number of input variables. The specific selection strategies are detailed in Appendix K and Appendix L. It demonstrates that these techniques can substantially reduce computational complexity without affecting the key conclusions.

(3) Approximation Methods. In addition to selection strategies, techniques specifically designed for efficiently computing sparse interactions to bypass exhaustive O(2ⁿ) evaluations (Kang et al., 2025; Butler et al., 2026), represent a clear path for reducing computational cost.

These strategies represent promising directions for extending the powerful capabilities of interaction-based analysis to a wider range of long-text NLP tasks.

## K. Details of the Experiments on Open-Ended Generation Tasks

## K.1. Experimental Setup

This section illustrates the detailed setup of open-ended question-answering tasks, which more closely resemble realworld user scenarios. We employed the Databricks Dolly-15k dataset, an open-source collection of instruction-following records. This dataset spans multiple behavioral categories as defined in the InstructGPT paper (Ouyang et al., 2022), including brainstorming, classification, closed QA, open QA, and summarization, providing a diverse and realistic dataset for evaluating model robustness. We test the results mainly on the Llama-2 family.

## K.2. Selection of Input Variables

Given that the text length in open-ended instructions far exceeds that of MCQ tasks, a direct analysis of all words would lead to exponential computational costs. To address this challenge, we applied the optimization strategies discussed in our limitations section: (1) Selective Input Variable Analysis and (2) Phrase-level Aggregation.

Our approach is guided by a systematic procedure to choose a fixed number of key input variables from the full input. Specifically, for each input sentence, we select meaningful words or phrases to construct the set of input variables N. A word is considered “meaningful" if it is not an NLTK (Bird, 2006) stop word or a punctuation mark. The remaining parts of the text, such as generic instruction templates (e.g., "Below is an instruction...") and stop words, are treated as fixed background context. During the interaction analysis, only the variables within the set N are masked.

For a concrete example, consider the following prompt:

Below is an instruction that describes a task. Write a response that   
appropriately completes the request.   
### Instruction:   
Identify from the following list characters from The X-Files who are bald or   
balding: Walter Skinner, John Fitzgerald Byers, Dana Scully, Melvin Frohike,   
Darius Michaud, Peter Watts, Conrad Strughold, Queequeg   
### Response:

## Selection Process for Input Variables:

1. Method (1) Application: We designate the generic instruction template and functional stop words (e.g., "from", "the", "who", "are") as background context, excluding them from the input variable set.

2. Method (2) Application: We aggregate words forming core semantic concepts into single phrasal units, such as the key entity "Walter Skinner" and the critical condition "bald or balding".

## Final Input Variables:

```json
[
"Identify", "following list", "characters","X-Files",
"bald or balding","Walter Skinner", "John Fitzgerald Byers",
"Dana Scully","Melvin Frohike", "Darius Michaud",
"Peter Watts","Conrad Strughold", "Queequeg"
]
```

## K.3. Experimental Results

By applying the aforementioned input variable selection strategies (Methods 1 and 2) in our experiments on the Dolly dataset, we successfully managed the analytical complexity for each long-text input, leading to a substantial reduction in computational cost.

Crucially, the experimental outcomes derived from open-ended questions and the optimized setup remained highly consistent with the main conclusions drawn from our MCQ-based experiments.

Tip: within each model series, lighter color represents smaller model scale

Here are the results and τ is set to 0.1, we can observe the same conclusions in this experiment

![](images/6e048ae0c70191a012b35c926cc7fd7d55c91df61edf0c336c93e0b8a4491ecd.jpg)  
Figure 45. A comparison of the prompt sensitivity between instruct/chat models and base models. Results show that instruct/chat models are less sensitive than corresponding base models.

![](images/266f7493210797b4fb52e193769d475682ed5f752abb4e2e0252a3dee5aff87c.jpg)  
Figure 46. A comparison of prompt sensitivity across different model scales. As the model scale increases, the prompt sensitivity within a model series systematically decreases.

![](images/7ea5e03c0847711a843f4f9b9164e3a5264736a68528d966c61041cdc92d8544.jpg)  
Llama 2 Family

![](images/a19f2f5512f75c2a295c0ceaa8095e6298431a4d9c48f76f398985de30759155.jpg)  
Figure 48. A comparison of prompt sensitivity between 0-shot learning and few-shot learning. The drop in prompt sensitivity is substantial from 0-shot to 1-shot.

Figure 47. A Comparison of prompt sensitivity between MoE models and dense models. Generally, MoE models tend to be more sensitive than dense models in the same model family.  
![](images/cc6dee5431e34a775695a532286206a85e99f730825621537f5f03a02fc7da2e.jpg)  
Figure 49. A comparison of the prompt sensitivity of three order types. Results show that low-order interactions are the least sensitive, while high-order interactions are the most sensitive.

(a) The relative change of Instruct/Chat Model compared to Base Model  
![](images/44c6198b381999370651fe652f5846854576face7b698cb68eb99487d26c967c.jpg)  
(b) The relative change of Dense Model compared to MoE Model

![](images/f90767ab8091ba51534109ceedc93c886b1ad32a50a3d10f3d8a788f221347c0.jpg)

(c) The relative change of Few-shot learning compared to 0-shot learning  
![](images/e0b7b93e4a2a1d3775ba651cb38d2e37942faa10d44d02c544556cc1fa367d8a.jpg)  
Figure 50. Comparing the relative change in the prompt sensitivity of low-, mid-, and high-order interactions for different factors.

![](images/44170f10b157094fe109cfc8ac9a01c0242f28a07ba3e1f8422f8ad199d5ec2b.jpg)

Tip: within each model series, lighter color represents smaller model scale  
![](images/1f806c4434852f734b90c33b24be3cc495251adf9e9229192da7dc9b53ffa1a2.jpg)

![](images/ccd51f93455f08a0e5b7c1e99bc0d0b900eff7aa13bcaf09a56f02373b64af83.jpg)  
Figure 51. A comparison of prompt sensitivity at the order-level across different model scales.

```markdown
Original Prompt:
Below is an instruction that describes a task. Write a
response that appropriately completes the request.
### Instruction:
Identify from the following list characters from The X-Files
who are bald or balding: Walter Skinner, John Fitzgerald
Byers, Dana Scully, Melvin Frohike, Darius Michaud,
Peter Watts, Conrad Strughold, Queequeg
### Response:
Perturbation 1: Semantic Paraphrase:
Below is an instruction that describes a task. Write a
response that appropriately completes the request.
### Instruction:
Select from the list below figures from The X-Files
that are losing their hair or bald: Walter Skinner,
John Fitzgerald Byers, Dana Scully, Melvin Frohike,
Darius Michaud, Peter Watts, Conrad Strughold, Queequeg
### Response:
Perturbation 2: Instruction Reordering:
Below is an instruction that describes a task. Write a
response that appropriately completes the request.
### Instruction:
From The X-Files, identify characters who are bald or balding
from the following list: Walter Skinner, John Fitzgerald
Byers, Dana Scully, Melvin Frohike, Darius Michaud,
Peter Watts, Conrad Strughold, Queequeg
### Response:
```

## L. Details of the Experiments Beyond Prompt Template Modifications

## L.1. Experimental Setup

To assess the generalizability of our findings beyond superficial template modifications, we conducted an additional experiment on the Dolly-15k dataset. This setup introduces more realistic and complex prompt perturbations that better reflect authentic user interactions, specifically semantic paraphrasing and instruction reordering. We employed these techniques to test the robustness of our conclusions under more challenging conditions. An illustrative example of a semantic paraphrase used in our experiment is shown below:

An illustrative example of the full prompt structure and the applied perturbations is shown below.

## L.2. Selection of Input Variables

For this experiment, we employed the same input variable selection strategy as detailed in Appendix K.2. This approach combines Selective Input Variable Analysis and Phrase-level Aggregation to manage the computational complexity associated with longer, open-ended instructions while preserving the core semantic elements for interaction analysis.

## L.3. Experimental Results

The experimental outcomes from this more challenging setup verify the main conclusions of our paper. Despite the increased complexity of the perturbations, we consistently observed that the four identified factors (supervised fine-tuning, increased

model scale, dense architectures, and few-shot learning) reduce prompt sensitivity, primarily by stabilizing low-order interactions. This replication demonstrates that our conclusions are not confined to simple template variations but hold true in scenarios that more closely mirror authentic user interactions, thereby strengthening the generalizability of our work

Here are the results and τ is set to 0.1, we can observe the same conclusions in this experiment.

![](images/d56f98c4b4a6e775e5a8bcb9c6c62416c11fd3a3ffe202c8314e921faa8012bf.jpg)  
Figure 52. A comparison of the prompt sensitivity between instruct/chat models and base models. Results show that instruct/chat models are less sensitive than corresponding base models.

![](images/77eb9ead6611a14e9a8283863ce5736a65c5b54f0f54d746a3d3687330fde85a.jpg)  
Figure 53. A comparison of prompt sensitivity across different model scales. As the model scale increases, the prompt sensitivity within a model series systematically decreases.

![](images/166dc81cbac503ad480828e5ab5e4d0445f5af0abe13ceae4eb77dffd3479734.jpg)  
Figure 54. A Comparison of prompt sensitivity between MoE models and dense models. Generally, MoE models tend to be more sensitive than dense models in the same model family.

![](images/e59fecadb66da84b0441dafb42c941ef8811b4b0255a17c56df3782004531932.jpg)  
Figure 55. A comparison of prompt sensitivity between 0-shot learning and few-shot learning. The drop in prompt sensitivity is substantial from 0-shot to 1-shot.

![](images/ec0e7e66db100e57990229dd7107c39dd4ff31d949e332565f55308d9a640475.jpg)  
Figure 56. A comparison of the prompt sensitivity of three order types. Results show that low-order interactions are the least sensitive, while high-order interactions are the most sensitive.

(a) The relative change of Instruct/Chat Model compared to Base Model  
![](images/a75adbc5fe0cd072015d0eb316e22bd146cab956fbf018c100e5b75896ef6184.jpg)  
(b) The relative change of Dense Model compared to MoE Model

![](images/6524a992f67b68f618592b7515b2c256f858082533d15a750934d8917978b11e.jpg)

(c) The relative change of Few-shot learning compared to 0-shot learning  
![](images/42074d8850a99ea477f1800acde23b274370edebeeaa313aad7e8a6561d0d884.jpg)  
Figure 57. Comparing the relative change in the prompt sensitivity of low-, mid-, and high-order interactions for different factors.

![](images/df40d1489e478c69f76165f6a9baac8d3f91e4f15246f7506c42f3a54e88582d.jpg)

Tip: within each model series, lighter color represents smaller model scale  
![](images/33d1d718be599b0f9160d8cf1f8e5135fe7b80476e58e733b118e5c38551f851.jpg)

![](images/3feb65a1cb03e9b374b1d1e171de0adb44ff2cbafc395a0172c8fc1c073e3dcb.jpg)  
Figure 58. A comparison of prompt sensitivity at the order-level across different model scales.