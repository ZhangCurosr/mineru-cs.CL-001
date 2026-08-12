# Every Token Counts: Exact Likert-Scale Distributions for Measuring LLM Attitudes and Biases

Davood Wadi McGill University

Mohsen Ghodrat University Canada West

Matthew Philp Marketing Management Toronto Metropolitan University

## Abstract

As Large Language Models (LLMs) are increasingly deployed as autonomous agents, accurately evaluating their latent values and biases is critical. The NLP community typically evaluates models using large, unstructured benchmarks. While effective for general capabilities, these datasets fundamentally conflate causal mechanisms: even when an aggregate bias is detected, unstructured evaluations cannot disentangle whether it stems from baseline traits, contextual confounders, or complex interactions. To address this, we introduce an analytically exact framework for the controlled behavioral evaluation of LLMs. We bridge human psychometrics with LLM mechanics by resolving gaps in design, measurement, and analysis. First, we replace unstructured prompting with fully crossed factorial experiments to systematically isolate causal main and interaction effects. Second, we eliminate Monte Carlo text sampling noise by operating directly on exact, token-level Probability Mass Functions (PMFs). Third, we derive a multivariate ordinal consensus metric and a distributional ANOVA to process these PMFs analytically. We validate our framework with a case study on consumer ethnocentrism across five LLMs, demonstrating how our approach isolates systemic country-of-origin biases that aggregate benchmarks otherwise obscure.

## 1 Introduction

As Large Language Models (LLMs) are increasingly deployed as autonomous agents (Wang et al., 2024; Xi et al., 2025; Wadi and Ma, 2026), evaluating their latent biases and subjective behaviors is critical. A pressing question in this new era is how to accurately quantify latent behavioral constructs, such as personality traits, moral values, or sociopolitical biases, to ensure AI fairness and safe global deployment.

Historically, the NLP community has evaluated models using large, unstructured benchmark datasets (Hendrycks et al., 2020; Bommasani et al., 2023).<sup>1</sup> While this scale-first paradigm is highly effective for measuring general capabilities (e.g., trivia, coding), it is fundamentally ill-equipped for precise behavioral evaluation (Santurkar et al., 2023; Durmus et al., 2024). Even when unstructured benchmarks successfully detect an aggregate bias in a model, they inherently lack the analytical power to disentangle which underlying factors actually caused that behavior, obscuring whether a bias is a hardwired training defect or a contextdependent adaptation (Wadi and Ma, 2026).

Consider an intuitive example: evaluating an LLM for gender bias using a large, unstructured dataset of professional scenarios. A benchmark might successfully detect a statistically significant bias against female subjects on aggregate. However, the unstructured dataset cannot disentangle why this bias exists. Because the prompt contexts vary across the dataset, "female" subjects might frequently be paired with specific scenarios, such as "corporate leadership." Is the model fundamentally biased against women across all contexts (a main effect of gender)? Is the model simply generating more critical text about leadership roles in general, regardless of gender (a main effect of context)? Or does the model treat genders equally most of the time, but severely penalize the specific combination of "female" and "leadership" (an interaction effect)? A benchmark dataset conflates these distinct causal mechanisms.

To disentangle what drives a model’s behavior, we propose a shift from unstructured dataset scaling to controlled experimentation. We formulate LLM evaluation as a fully crossed factorial design, where every distinct variable (e.g., demographic, context, LLM persona) is treated as an independent factor. We systematically evaluate every possible combination using validated psychometric scales (Cronbach, 1950; Batterton and Hale, 2017). This experimental control grants us the analytical power to mathematically isolate causal main effects and higher-order interactions. This fills an important gap in behavioral attribution that aggregate benchmarks lack.

In standard NLP evaluation, scaling up a dataset with thousands of generated prompts is an excellent way to test general model capabilities. However, if we try to apply this "scale-first" approach to a psychological survey by adding thousands of synthetic, unvalidated questions, we risk altering the core trait we are trying to measure, a problem known as reducing construct validity (Flake and Fried, 2020).

To keep our measurements reliable, we rely on carefully calibrated questions used by behavioral scientists (e.g., Likert scales). Moreover, translating behavioral methods from humans to LLMs has several major methodological gaps, which we formalize as experimental design, measurement, and analysis. We formulate a statistical framework to execute controlled behavioral experiments on LLMs, while addressing these gaps.

First, to address the design gap, our framework formulates LLM evaluation as a fully crossed factorial experiment. By treating variables (e.g., the specific LLM being tested, the target demographic in the prompt) as distinct experimental factors, we systematically separate a model’s baseline behavior from causal main effects and higher-order interactions. This experimental control grants us the analytical power to mathematically isolate biases free of confounding contexts.

Second, to address the measurement gap, we bypass the issues of generative text sampling. Human behavioral science relies on sampling because the underlying probability distribution of a human mind is inaccessible. NLP has adopted this paradigm via Monte Carlo text sampling to simulate a "population" of responses (Wadi and Fredette, 2025), often injecting additional noise via the temperature parameter. However, an LLM possesses a constant, fully observable internal state for a given prompt. Generating multiple text responses is effectively taking repeated, noisy draws from a static mathematical distribution. Our framework, instead, operates directly on exact, token-level Probability Mass Functions (PMFs) to eliminate sampling error.

Third, to address the analytical gap, we formulate the statistical mechanics required to process these exact PMFs. When NLP researchers evaluate token probabilities, they typically rely on metrics like Shannon entropy over valid tokens (Farquhar et al., 2024). However, entropy fails on Likert scales because it ignores ordinality. For example, entropy cannot distinguish between an LLM heavily polarized between "Strongly Agree" (5) and "Strongly Disagree" (1) versus one smoothly concentrated around "Neutral" (3). We bridge this gap by introducing a multivariate consensus metric that respects ordinal distance. We also formulate distributional ANOVA to analytically compute the exact probability distributions of the isolated causal effects.

To empirically validate our general-purpose framework, we present a case study on consumer ethnocentrism, the sociological phenomenon where an entity systematically favors its country of origin and rejects foreign entities (Shankarmahesh, 2006). Applying established ethnocentrism scales to five modern LLMs, our framework shows how models’ countries of origin inherently interact with target countries, thereby allowing us to test for the magnitude of ethnocentrism for each LLM. We mathematically show that controlled experiments can isolate geographic biases without relying on large, unstructured datasets.

We contribute a unified, deterministic framework for the causal behavioral evaluation of LLMs on ordinal psychometric instruments. The framework casts evaluation as a fully crossed factorial experiment operating directly on exact token-level Probability Mass Functions (PMFs), and its methodological core is a distributional ANOVA that isolates causal effects as full probability distributions. The pipeline:

• decomposes the composite construct distribution into baseline, main-effect, and interactioneffect distributions via a distributional Hoeffding decomposition with Optimal-Transportbased paired contrasts and a formal ANOVAconsistency guarantee (Theorem 3.1);

• propagates item-level PMFs into an exact composite-score distribution using discrete convolution; and

• addresses a failure mode of Shannon entropy on Likert scales by quantifying ordinal response certainty using a multivariate extension of the Consensus metric (Tastle and Wierman, 2007).

## 2 Related work

Recent works have utilized psychometric tests to measure a wide array of constructs (Jiang et al., 2024; Li et al., 2024; Helwe et al., 2025). However, recent studies demonstrate that eliciting psychometric responses via text generation is highly unstable. Simple prompt perturbations can significantly downgrade the consistency of LLM responses on persona axes (Shu et al., 2024). Moreover, traditional psychometric testing on LLMs suffers from low ecological validity when relying on standard generative outputs (Jung et al., 2026). Part of this instability stems from generative sampling, which is a methodological artifact borrowed from human surveying, and obscures their actual underlying distribution (Deas and McKeown, 2025).

Uncertainty and probability in LLMs Recognizing the limitations of text-based sampling, a growing body of literature has turned to analyz ing exact token probabilities to better understand LLMs’ internal states. Kuribayashi et al. (2024) demonstrated that utilizing exact next-word probabilities is far superior to instruction-tuned prompting when simulating human cognitive behaviors. Similarly, recent work cautions that prompting LLMs directly for their uncertainty yields poorly calibrated text (Liu et al., 2025), whereas their exact internal token probabilities remain highly predictive (Bentegeac et al., 2025; Kadavath et al., 2022). To quantify the variance within these predictive probability distributions, the NLP community relies almost exclusively on Shannon entropy or its semantic derivatives (Farquhar et al., 2024). Entropy is highly effective for tasks where the output space consists of distinct, categorical facts (e.g., detecting hallucinations across mutually exclusive answers). However, standard metrics like entropy fail when applied to behavioral evaluation using psychometric rating scales. This is because entropy treats all tokens as unordered, categorical variables, ignoring the ordinality of a Likert scale. Because standard uncertainty metrics are indifferent to this ordinality, we lack the mathematical foundation to utilize exact token probability mass functions (PMFs) for behavioral evaluation. As demonstrated in 3, our framework resolved this impasse by formulating a measure of consensus specifically designed for ordinal variables.

Geopolitical bias and ethnocentrism in LLMs As LLMs are deployed globally, there is growing concern regarding their cultural neutrality. A robust body of literature documents that LLMs exhibit systemic geopolitical and geographical biases (Shwartz, 2022; Bhatia and Shwartz, 2023; Bhatia et al., 2024; Naous and Xu, 2025; Kruspe, 2024). Furthermore, models frequently struggle to bridge culture gaps (Tanwar et al., 2025; Cecilia Liu et al., 2024; Saha et al., 2025; Liu et al., 2026). To systematically quantify these cultural alignments, researchers frequently turn to global sociological instruments, such as the World Values Survey (Zhao et al., 2024). Yet, these evaluations suffer from the exact generative brittleness highlighted above. Kabir et al. (2025) noted that evaluating cultural alignment in LLMs using standard multiple-choice surveys leads to wildly inconsistent outputs under minor perturbations, such as choice reordering.

To bypass this brittleness, research has advocated moving toward unconstrained, open-ended generative evaluations. However, relying on openended generation introduces a new set of methodological bottlenecks. Specifically, it is notoriously difficult to reliably parse distinct categorical or ordinal scores from free-text outputs. Open-ended generation is highly susceptible to "fence-sitting", where models generate evasive disclaimers (e.g., "As an AI, I do not have a country...") rather than revealing their latent behavioral stance (Bhatia et al., 2025; Liu et al., 2025). Dealing with these nuances typically requires utilizing another LLM as a judge to evaluate the text, which merely injects a secondary layer of bias into the evaluation pipeline.

## 3 Methodology

Figure 1 illustrates our framework. We formulate LLM evaluation as a fully crossed factorial experiment to resolve the design gap. Within this design, we treat the LLM as a generator of exact Probability Mass Functions (PMFs) to resolve the measurement gap and bypass generative sampling noise. Finally, we process exact PMFs through an analytical pipeline, which provides the foundation necessary for evaluating ordinal psychometric data and isolating causal behavioral shifts.

## 3.1 Experiment design

Rather than probing models with large datasets of ad-hoc prompt variations, we formalize LLM behavioral evaluation as a fully crossed fixed-effects experiment. We define our experiment through a set of independent factors (e.g., the MODEL being tested, and the TARGET COUNTRY in the prompt; see Fig. 7). Each factor contains distinct levels (e.g., four specific countries). A single experimental condition, denoted as λ, represents one specific combination of these factors (e.g., MODEL = Llama-3.3, TARGET = Canada). Testing every possible combination of λ forms our complete ex-<sup>perimental</sup> <sup>design</sup> <sup>space</sup> D<sup>.</sup> <sup>By</sup> <sup>evaluating</sup> <sup>across</sup> the entirety of , we can mathematically isolate the effect of one variable while controlling for the others. Under each condition λ, the model evaluates a psychometric instrument consisting of K individual questions (items). For each item, the model’s response is bounded to an ordinal Likert scale  ranging from $y _ { \mathrm { m i n } }$ to y<sub>max</sub> (e.g., digits 1 through 7).

![](images/12053b31b30c7722798d235a408ee5b23e006f929907d36c5503138052dc0ce2.jpg)  
Figure 1: High-level exact-PMF framework: Fully crossed experiment design token probability data collection three-layer analysis (Constraint, Consensus, Construct).

A central property of this design is context isolation. The model’s internal state is effectively reset between items. This is an advantage that LLM behavioral analysis has over human behavioral analysis. With humans, there is a within-subjects carryover effect that contaminates subsequent responses of human participants, whereas statelessness of LLMs does not suffer from this limitation. For a fixed experimental condition λ, the predicted response $Y _ { k , \lambda }$ for item k depends solely on the specific isolated prompt $x _ { k , \lambda }$ . Following Wadi and Fredette (2025), we treat the items as conditionally independent given λ, meaning the joint probability mass function over the scale factorizes exactly: $\begin{array} { r } { P ( \mathbf { Y } _ { \lambda } = \mathbf { y } \mid \boldsymbol { \lambda } ) = \prod _ { k = 1 } ^ { K } P ( Y _ { k , \lambda } = y _ { k } \mid \boldsymbol { \lambda } ) } \end{array}$

## 3.2 Constraint

Before analyzing behavior, we must ensure the model adheres to the psychometric instrument. LLMs generate predictions over a large vocabulary ( ), but a Likert scale only permits specific ordinal tokens (e.g., digits 1 through 7).

For a given prompt x, let $P _ { \mathrm { r a w } } ( t \mid x )$ denote the model’s next-token probability mass function over the full vocabulary , where $t \in \nu$ is a candidate token. Let $\mathcal { V } _ { \mathrm { v a l } } \subset \mathcal { V }$ be the valid token set with a surjective mapping $\phi : \mathcal { V } _ { \mathrm { v a l } } \to \mathcal { V }$ associating valid tokens to ordinal values. To account for tokenizer idiosyncrasies (e.g., BPE leading spaces like " 7" versus "7"), $\mathcal { V } _ { \mathrm { v a l } }$ includes all valid digit surface forms, but strictly excludes semantic equivalents (e.g., "seven") to penalize syntactic constraint failures. We define the model’s failure rate as $\begin{array} { r } { 1 - \sum _ { t \in \mathcal { V } _ { \mathrm { v a l } } } P _ { \mathrm { r a w } } ( t \mid x _ { k , \lambda } ) } \end{array}$ , which quantifies how often the model fence-sits (i.e., refuses to answer) or hallucinates out of the constraint. For all subsequent behavioral analyses, we utilize the renormalized conditional distribution on $\mathcal { V } _ { \mathrm { v a l } }$ . Because multiple tokens can map to the same digit, we sum over the pre-image to yield an exact ordinal item PMF: $P ( Y _ { k , \lambda } = y \mid$ $\begin{array} { r } { \pmb { \lambda } ) = \sum _ { t \in \phi ^ { - 1 } ( y ) } P ( t  { | } x _ { k , \lambda } , t \in \mathcal { V } _ { \mathrm { v a l } } ) } \end{array}$ This follows the convention in human behavioral experiments, which excludes human responses that do not adhere to instructions (Cronbach, 1950).

## 3.3 Consensus

Quantifying the expected value (mean) of a Likert response is necessary but insufficient, as identical means can mask fundamentally different behavioral regimes (e.g., polarizing mass on {1, 7} vs.

stable mass on {4, 4}).

Standard uncertainty metrics like Shannon entropy over valid tokens fail here because they are invariant to ordinal distance. To address this, we compute a multivariate generalization of Consensus (Tastle and Wierman, 2007) directly on the model’s PMF. This metric explicitly scales probabilities by their distance from the mean. Let $\pmb { \mu }$ be the itemwise mean vector and $d _ { \operatorname* { m a x } } = \sqrt { K } \left( y _ { \operatorname* { m a x } } - y _ { \operatorname* { m i n } } \right)$ be the maximum diagonal distance on the Likert scale. The multivariate Consensus (Cns) is defined $\mathrm { a s } ^ { 2 }$

$$
\mathrm { C n s } ( \mathbf { Y } _ { \boldsymbol { \lambda } } ) = 1 + \sum _ { \mathbf { y } \in \mathcal { V } ^ { K } } P ( \mathbf { y } ) \log _ { 2 } \left( 1 - \frac { \| \mathbf { y } - \pmb \mu \| _ { 2 } } { d _ { \operatorname* { m a x } } } \right) .\tag{1}
$$

Unlike entropy, this metric correctly quantifies the model’s internal consistency. It severely penalizes polarization at the extremes and rewards concentrated probability mass, allowing us to accurately measure the model’s opinion certainty (Table 3). It also resolves the issue of collapsing PMFs from the convolution operator (Fig. 6).

## 3.4 Construct

A common approach to analyzing LLM behavior involves comparing descriptive statistics, such as average response scores across different prompt manipulations. While this practice is highly informative for observing general trends, analyzing raw point estimates inherently obscures information about the underlying variance. Consequently, it becomes difficult to ascertain whether the observed difference between groups is statistically significant, or whether the magnitude of the difference (effect size) is meaningful in practice. Moreover, a key component of controlled experiments is understanding interactions between factors. Two factors can interact to amplify or decrease their individual effects (Wadi et al., 2026).

To overcome these limitations, we adapt the Analysis of Variance (ANOVA) framework for LLMs. A statistically grounded factorial framework allows us to systematically answer three critical questions: (1) whether the behavioral shift between conditions is statistically significant, (2) isolated from confounding variables, what the exact effect size and magnitude of that shift is, and (3) how experimental factors interact to jointly shape the model’s behavior.

In standard psychometrics, a latent trait, such as ethnocentrism, cannot be reliably measured by a single question. Rather, it is quantified using an instrument containing a total of K individual Likert items. Let $Y _ { k , \lambda }$ represent the model’s ordinal response to item k under a specific experimental condition λ (e.g., a specific target country and LLM). The composite construct score $S _ { \lambda }$ is defined as the sum of these individual item responses $\begin{array} { r } { ( S _ { \lambda } = \sum _ { k = 1 } ^ { K } Y _ { k , \lambda } ) } \end{array}$ . Because our framework operates on exact probability distributions rather than single scalar outputs, we evaluate the full probability mass function (PMF) of this composite score.

Given the conditional independence of items (Sec. A.2), according to standard probability theory, the distribution of a sum of independent variables is computed via discrete convolution (denoted by the operator ⊛). Therefore, the exact probability distribution of the total construct score $P _ { S _ { \lambda } }$ is derived by convolving the individual item PMFs: $P _ { S _ { \lambda } } = P _ { Y _ { 1 , \lambda } } \circledast \cdots \circledast P _ { Y _ { K , \lambda } }$ . Unlike standard evaluation methods that compute descriptive statistics over a handful of noisy, sampled text generations, the convolution approach analytically derives the exact probability distribution of the final construct score $( P _ { S _ { \lambda } } )$ . This propagates all aleatoric uncertainty from the token level up to the final construct level.

Next, we need to systematically decompose LLM responses $( P _ { S _ { \lambda } } )$ into distinct, interpretable components. Rather than evaluating each experimental condition (λ) as a whole, we decompose the model’s overall construct distribution into three parts: the model’s baseline behavior, the isolated Main Effects (e.g., how much changing the Target Country specifically shifts the behavior), and the Interaction Effects (e.g., how the Target Country interacts with a specific LLM). We formalize this using a distributional Hoeffding decomposition. Let $S _ { 0 }$ represent the grand-mixture baseline across all conditions, $E _ { c } ( \lambda _ { c } )$ represent the Main Effect distribution for factor c at level $\lambda _ { c } ,$ and $E _ { U } ( \lambda _ { U } )$ represent the higher-order Interaction Effect distribution for a subset of factors $U$ . This decomposition satisfies the following identity in expectation:

$$
\mathbb { E } [ S _ { \lambda } ] = \underbrace { \mathbb { E } [ S _ { 0 } ] } _ { \mathrm { G r a n d M e a n } } + \sum _ { c \in \mathcal { C } } \underbrace { \mathbb { E } [ E _ { c } ( \lambda _ { c } ) ] } _ { \mathrm { M a i n F f f e c t s } } + \sum _ { U \subseteq \mathcal { C } } \underbrace { \mathbb { E } [ E _ { U } ( \lambda _ { U } ) ] } _ { \mathrm { I } \mathrm { I } \mathrm { I } \mathrm { e } \mathrm { r a c t i o n s } } ,\tag{2}
$$

This exact-PMF provides the guarantee that the expected values of the derived probability distributions $( \mathbb { E } [ E _ { c } ( \lambda _ { c } ) ]$ and $\mathbb { E } [ E _ { U } ( \lambda _ { U } ) ] )$ match the proven fixed-effects parameters calculated in traditional human ANOVA experiments. We formalize this alignment with the following theorem (Proof in App. A.9):

Theorem 3.1. For any fully crossed experimental design, the expectations of the isolated effect distributions $( E _ { c } , E _ { U } )$ recover the classical unique Hoeffding decomposition.

In practical terms, Theorem 3.1 guarantees that when we measure a higher-order behavior, such as a "country-of-origin bias" (an interaction effect), we have mathematically stripped away the model’s baseline ethnocentrism (main effect). This ensures researchers do not double-count biases, achieving the exact same control over confounding variables used by human behavioral scientists.

Finally, to translate these effect distributions into interpretable behavioral insights, we extract the following summary metrics for our case study, reported for any given effect distribution E:

• Mean Shift $( \mathbb { E } [ E ] ) \colon$ : The expected value of the distribution, representing the actual magnitude (effect size) of the isolated behavioral effect.

• Predictive Dispersion (SD(E)): The standard deviation of the effect, capturing the model’s aleatoric uncertainty regarding the shift.

• Discrete Probability of Direction (dPD): Defined as max $\cdot ( \mathbb { P } ( E > 0 ) , \mathbb { P } ( E < 0 ) )$ , this metric quantifies the extent to which the behavioral shift exhibits a consistent directionality (i.e., strictly positive or strictly negative), independent of the effect’s magnitude (App. A.4.8).

• Signal-to-Noise Ratio (SNR): Calculated as $| \mathbb { E } [ E ] | / \mathrm { S D } ( E )$ , SNR quantifies the statistical reliability of the effect, indicating whether the observed behavioral shift is a robust, distinct phenomenon or merely an artifact of the model’s internal uncertainty.

With these metrics, we now apply this methodology to our main sociotechnical concern. We test whether LLMs exhibit systematically biased ethnocentric behaviors based on their country of origin. See App. A.4.12 for details of each metric.

## 4 Experiment

## 4.1 Design

We test our framework using the Consumer Ethnocentrism Tendencies Scale (CETSCALE; Shimp and Sharma, 1987). This is a widely validated K=17-item Likert instrument with a response space of $\mathcal { V } = \{ 1 , \ldots , 7 \}$ . While a 17-item survey is much smaller than typical NLP benchmark datasets, it constitutes a complete, rigorously tested measurement tool in the behavioral sciences. Using this exact, unmodified scale ensures that we are reliably measuring the intended bias without diluting the test.

We employ a symmetrical, fully crossed $5 \ \times \ 4$ fixed-effects design across factors MODEL, TARGET . The model levels include open-weights models representing four distinct countries-of-origin: <sub>MODEL</sub> = Llama 3.3 70B and Gemma 3 27B (USA); Qwen3 Next 80B (China); Aya Expanse 32B (Canada); and Ministral 14B (France) . The Target Country levels mirror these origins: $\begin{array} { r l } { \mathcal { L } _ { \mathrm { T A R G E T } } } & { { } = } \end{array}$ USA, China, Canada, France . This symmetrical $5 \times 4$ design is critical as it allows our framework to mathematically isolate (i) the global main effects of MODEL and TARGET, and (ii) the precise MODEL  TARGET interaction effects (e.g., asking a Chinese-developed model to express its ethnocentrism towards USA), which represent the actual country-of-origin bias.

All conditions share an identical stimulus structure. A fixed system message strictly enforces the response schema ("...Respond with ONLY the digit."). Each of the 17 CETSCALE items acts as a user-message template $( T _ { k } )$ containing a target country placeholder (e.g., {People}, {Adj}). The TARGET factor deterministically instantiates these placeholders via a strict lexicon (Figure 7). Importantly, prompt wording is treated as a fixed experimental control rather than tuned per model, with the aim to mirror stimulus control in human behavioral experiments.

For each (MODEL, TARGET, $k ) \ \in \ \mathcal { L } _ { \mathrm { M o D E L } } \ \times$ $\mathcal { L } _ { \mathrm { { T A R G E T } } } \times \{ 1 , \dots , 1 7 \}$ , we perform a single forward pass and record the exact next-token PMF $( P _ { \mathrm { r a w } } ( t \ \mid \ x _ { k , \lambda } ) )$ over the full vocabulary.<sup>3</sup> We define $\mathcal { V } _ { \mathrm { v a l } }$ as the specific valid digit tokens $\{ 1 , \ldots , 7 \}$ for each model’s respective tokenizer. In total, this yields $5 \times 4 \times 1 7 = 3 4 0$ exact itemlevel predictive distributions. These PMFs are directly passed through the Constraint Layer to measure formatting failures, the Consensus Layer to measure ordinal certainty, and the Construct Layer to isolate distributional Hoeffding effect sizes.

## 4.2 Results

Constraint. As shown in Figure 2, Gemma 3, Llama 3.3, and Qwen3 demonstrate near-perfect adherence to the task constraints, concentrating their probability mass almost entirely on valid ordinal tokens $( M \ < \ 0 . 0 0 1 , \ \mathrm { S D } \ < \ 0 . 0 0 1 )$ . In contrast, Ministral exhibits a small but consistent failure to constrain its mass to the requested digits $( M \ : = \ : 0 . 0 0 2 , \mathrm { S D \ : = \ : 0 . 0 0 1 ) }$ . More notably, Aya Expanse displays the highest aggregate failure rate $( M \ : = \ : 0 . 0 0 6 )$ driven by large variance $\mathrm { ( S D = 0 . 0 4 9 ) }$ . This indicates that utilizing Aya or Ministral in automated agentic settings could lead to sporadic breakdowns due to their probabilistic inability to adhere strictly to psychometric constraints.

![](images/580da1ce7a88ffd0861f9c6b684b60f82ef6b34ff8ee5453337790a018d6f1e5.jpg)  
Figure 2: Failure Rate across the evaluated LLMs.

Consensus. Across all target countries, four models show exceptionally high internal agreement regarding the instrument, maintaining Cns > 0.88 (Figure 3). Ministral is a major outlier, exhibiting substantially lower consensus across all targets $( \mathrm { C n s } \approx 0 . 6 5  – 0 . 6 7 )$ . This shows that, even after enforcing valid digits, Ministral’s item-level PMFs frequently place large probability mass on conflicting ends of the Likert scales. This stark contrast highlights the critical necessity of an ordinal-aware dispersion metric. Standard entropy would fail to capture this specific behavioral dissonance (Table 5).

![](images/5b6c93417f011c0df8cee76e92d41a2e8f8f10e65da7d3e5e475b56369c029b4.jpg)  
Figure 3: Multivariate Consensus (Cns) for the CETSCALE instrument. Lower values indicate severe behavioral polarization.

Construct. To contextualize the LLMs’ response to the CETSCALE, we compare each model’s predictive distribution for TARGET=USA against the human sample means and standard deviations reported in the original CETSCALE validation studies (Shimp and Sharma, 1987; Table 1; Figure 8)<sup>4</sup>. Several LLMs exhibit expected composite scores comparable to, or explicitly exceeding, those observed in the most ethnocentric human populations. See the full distributions in Figure 10.

Table 1: Comparison of CETSCALE scores between historical human populations (Shimp and Sharma, 1987) and LLMs evaluated in this study (restricted to the TAR-GET=USA condition). Higher composite score indicates higher ethnocentrism.
<table><tr><td>SUBJECT</td><td>SAMPLE / MODEL</td><td>E[S]</td><td>SD[S]</td></tr><tr><td rowspan="5">HUMAN</td><td>DETROIT (USA)</td><td>68.58</td><td>25.96</td></tr><tr><td>CAROLINAS (USA)</td><td>61.28</td><td>24.41</td></tr><tr><td>DENVER (USA)</td><td>57.84</td><td>26.10</td></tr><tr><td>LOS ANGELES (USA)</td><td>56.62</td><td>26.37</td></tr><tr><td>STUDENTS (PRE) STUDENTS (POST)</td><td>51.92 53.39</td><td>16.37 16.52</td></tr><tr><td rowspan="5">LLM</td><td>AYA EXPANSE 32B</td><td>89.11</td><td>1.32</td></tr><tr><td>LLAMA 3.3 70B</td><td>71.99</td><td>0.95</td></tr><tr><td>MINISTRAL 14B</td><td>70.20</td><td>5.25</td></tr><tr><td>GEMMA327B</td><td>60.32</td><td>0.94</td></tr><tr><td>QWEN3 NEXT 80B</td><td>53.85</td><td>1.17</td></tr></table>

Main Effects. Extracting precise main effect distributions (Table 2), reveals large and systematic behavioral divides. $\mathbf { A Y A }  – 3 2 \mathbf { B }$ exhibits a severe positive deviation from the grand mean $( \mathbb { E } = 2 1 . 1 9 ,$ $\mathrm { S N R } = 1 . 6 3 , \mathrm { d P D } > 0 . 9 9 )$ , indicating a pervasive ethnocentrism that far exceeds human baselines. Conversely, QWEN3-80B demonstrates a highly robust negative main effect (E = 14.63, $\mathrm { S N R } = 1 . 2 0 , \mathrm { d P D } = 0 . 9 3 )$ , resulting in expected scores similar to liberal human populations.

![](images/cbc1ffd7c1de1b4cf435229fa50dffb7e05bd4c93b4e8f5f36719bed4c1301c0.jpg)

(a) Aya Expanse 32B  
![](images/941cec3f6961ecf12f9568811a745b29e953c6b9c35eeffeaf9b98cf808f8043.jpg)  
(b) Qwen3 Next 80B  
Figure 4: Distribution of CETSCALE for Aya Expanse 32B (high) vs. Qwen3 Next 80B (low) ethnocentrism and Target Country.

Analyzing the main effects for the Target Countries (Table 2; and the pairwise contrasts in Table 7; Figure 9) reveals that the most statistically robust biases involve China. Across all models, there is a systemic decrease in ethnocentrism when the target is China $( \mathbb { E } = - 6 . 4 6 , \mathrm { S N R } = 1 . 0 4 )$ , contrasting sharply with the relative favoritism shown toward North American targets (e.g., USA-China contrast $\mathbb { E } = 9 . 3 0 , \mathrm { S N R } = 1 . 3 4 )$

Table 2: Summaries of main effects.
<table><tr><td>PARAMETER</td><td>E</td><td>SD</td><td>SNR</td><td>dPD</td></tr><tr><td>µ∅</td><td>66.25</td><td>14.21</td><td></td><td></td></tr><tr><td>AYA EXPANSE 32B</td><td>21.19</td><td>13.01</td><td>1.63</td><td>0.99</td></tr><tr><td>GEMMA 3 27B</td><td>-11.99</td><td>13.15</td><td>0.91</td><td>0.77</td></tr><tr><td>LLAMA 3.3 70B</td><td>0.31</td><td>12.79</td><td>0.02</td><td>0.55</td></tr><tr><td>MINISTRAL 14B</td><td>5.11</td><td>9.59</td><td>0.53</td><td>0.62</td></tr><tr><td>QWEN3 NEXT 80B</td><td>-14.63</td><td>12.17</td><td>1.20</td><td>0.93</td></tr><tr><td>USA</td><td>2.84</td><td>5.43</td><td>0.52</td><td>0.55</td></tr><tr><td>CANADA</td><td>4.62</td><td>5.72</td><td>0.81</td><td>0.90</td></tr><tr><td>CHINA</td><td>-6.46</td><td>6.19</td><td>1.04</td><td>0.95</td></tr><tr><td>FRANCE</td><td>-1.00</td><td>4.72</td><td>0.21</td><td>0.54</td></tr></table>

Interaction Effects and Country-of-Origin Bias. We define country-of-origin bias as a positive interaction between a model and its own country of origin, measured as an isolated deviation from the grand-mean baseline. The interaction distributions $( E _ { U } )$ show that several models exhibit structured, non-additive interaction patterns (Figure 5; Table 6), though the strength and directionality of these effects vary considerably across models. The clearest regional pattern appears for GEMMA 3- 27B and LLAMA 3.3-70B (both U.S.-developed), which show positive interactions toward the North American bloc (USA and Canada) alongside a pronounced negative interaction with China. Both models exhibit positive own-country (USA) interactions (+3.21 and +2.59), consistent with ingroup favoritism, although these particular ingroup cells are directionally moderate (dPD = 0.63 and 0.58). We emphasize that clean own-country favoritism is not universal across all five models. AYA-32B (Canada, 3.77) and QWEN3 NEXT-80B (China, 0.20) do not exhibit a positive own-country interaction, and MINISTRAL-14B shows a positive but modest France interaction (+2.45) that is smaller than its interaction with China. This showcases how the PMF ANOVA framework isolates each interaction from the model’s baseline behavior and main effects and allows researchers to empirically test whether a construct such as country-of-origin bias is present, absent, or reversed for a given model.

![](images/7342ac4025ec4f33f452078b59e6ec43a8647329a17c7ffa4016af49f609335e.jpg)  
Figure 5: Interactions of Model and Target Country.

Aggregate vs. Factorial Benchmarks. To empirically demonstrate the danger of conflating causal mechanisms, we compared factorial interactions against an aggregate benchmark (App. 9). If a researcher were to evaluate Target Country bias by taking an aggregate average, they would conclude there is a universal negative bias against France $( \mathbb { E } ~ = ~ - 1 . 0 0 )$ and China $( \mathbb { E } ~ = ~ - 6 . 4 6 )$ . However, factorial decomposition shows this aggregate view is directionally incorrect for specific models: Ministral-14B actually exhibits positive ingroup favoritism toward France (+2.45). Similarly, while the aggregate benchmark suggests a universal penalization of China, both Aya-32B (+5.35) and Ministral-14B (+4.54) exhibit strong positive interactions.

## 4.3 The Cost of Generative Sampling

To quantify the cost of the standard NLP Monte Carlo sampling paradigm, we benchmarked a simulated sampling approach against our exact distributions (acting as ground truth; App. A.8). Standard text sampling introduces severe, previously unquantified reliability issues in behavioral evaluation.

First, sampling noise degrades the detection of subtle biases. Because the standard error decays only as $1 / \sqrt { N }$ , a standard sampling budget of $N = 1 0 0$ still yields a standard error of up to 0.52 composite-score points for low-consensus models (App. A.8.1). To determine how often this noise leads to false conclusions, we calculated the flip probability $( P _ { \mathrm { f l i p } } ) { \mathrm { : } }$ the chance that finite sampling recovers an effect with the wrong sign (App. A.9). At $N = 1 0$ , standard sampling flips the sign of small effects $( | E | < 1 )$ 18% of the time, and even at $N = 1 0 0 .$ , it flips them 6.3% of the time. The exact-PMF framework achieves the $N  \infty$ limit in a single forward pass, ensuring no interaction effect is masked or reversed by sampling variance. Second, non-default decoding (e.g., temperature scaling) introduces systematic biases that no amount of sampling can fix. Our analysis (App. A.8.2) shows that for models with low internal consensus (e.g., Ministral), modifying temperature artificially shifts the latent construct score by up to 2.79 points.

## 4.4 Prompt Sensitivity

A common critique in NLP behavioral evaluation is that model outputs are highly sensitive to prompt phrasing. A key strength of casting evaluation as a fully crossed factorial design is its extensibility. Any potential confounder can be introduced as an additional factor and isolated via the distributional ANOVA.

To demonstrate this, we promoted the administration (system) prompt to a first-class experimental factor $( \lambda _ { \mathrm { P r o m p t } } )$ . As detailed in App. A.10, we tested five additional framing variants (ranging from a market-research persona to minimal formatting; Table 13) across three models and four target countries, yielding 1,224 exact item-level PMFs.

The full ANOVA decomposition (Tables 14 and 15) show three key findings. First, the core countryof-origin signals are robust. The TARGET main effects reproduce the directional pattern of the main study under statistical control for framing. Second, prompt framing acts as a causal main effect of its own, shifting the latent construct score by up to 13.5 points depending on the phrasing used. Third, the framework formally quantifies model-specific prompt sensitivity via the MODEL PROMPT interaction. We isolate that while Gemma 3 exhibits low sensitivity to wording changes, both Aya and Ministral interact strongly with specific framings.

## 5 Conclusion

In this work, we introduced a new paradigm for the behavioral evaluation of LLMs, shifting the focus from large-scale, unstructured benchmark datasets to controlled factorial experiments. By bridging human psychometrics with LLM mechanics, we addressed critical gaps in experimental design, measurement, and analysis. By operating directly on exact, token-level Probability Mass Functions (PMFs) without generative sampling noise, we formulated an analytical pipeline, featuring a multivariate ordinal consensus metric and a distributional ANOVA. This pipeline allows researchers to mathematically isolate causal behavioral shifts and factor interactions without artificially inflating variance.

We validated this framework with a case study on consumer ethnocentrism. We isolated systemic country-of-origin biases that would otherwise be obscured by confounding variables in standard observational datasets. Ultimately, by prioritizing causal experimental control and exact probability distributions over text sampling, this generalpurpose framework provides a statistically rigorous foundation for future research into LLM alignment, values, and latent behavioral traits.

## Limitations

While our mathematical framework is designed to be applied to any behavioral evaluation, our empirical experiments focus exclusively on consumer ethnocentrism. We deliberately chose to analyze a single construct in depth to clearly illustrate how the distributional ANOVA operates under the hood. Now that the methodology is established, applying this exact-PMF framework to a much wider array of psychological traits, moral values, and geopolitical biases is an exciting direction for future work. Therefore, the specific findings in this paper should not be interpreted as comprehensive claims about the general political ideology or cultural alignment of the tested models.

Furthermore, all experiments were conducted in English. Whether the observed systemic biases hold when models are queried in other languages or scripts remains an open empirical question. Additionally, as these are snapshot evaluations, our findings reflect the models’ latent states at the time of testing and may not generalize to future weight updates or subsequent model versions.

A common critique in NLP evaluation concerns model sensitivity to prompt variations. We adopt the strict position from behavioral psychometrics that the prompt template is a standardized measurement instrument, not a tunable hyperparameter. In human surveying, rewording a validated item constitutes a fundamental change to the construct being measured, rather than a robustness failure of the experiment. We stress, however, that our framework treats any experimental variable as a crossed factor, so prompt phrasing is itself a first-class factor λ<sub>Prompt</sub> that the design directly supports. Introducing a set of semantically-equivalent paraphrases as levels of λ<sub>Prompt</sub> requires no change to the underlying theory. The Prompt main effect and the Model Prompt and Target  Prompt interactions are recovered by the identical distributional-ANOVA method. In this view, prompt sensitivity becomes a measurable, isolable effect rather than an uncontrolled confound.

Methodologically, our framework relies on exact next-token probability distributions and is explicitly designed for constrained, single-token response paradigms (e.g., Likert scales). While major proprietary providers (e.g., OpenAI, Anthropic, Google) currently expose log-probabilities via their APIs, providers that entirely obscure their next-token probabilities fall outside the scope of this approach.

Furthermore, as LLMs increasingly utilize multitoken reasoning protocols (e.g., Chain-of-Thought) prior to generating a final response, extending our Consensus and Construct statistics to integrate over latent multi-token trajectory spaces remains a complex, open challenge for future work.

## Ethical Considerations

This framework is intended as a diagnostic tool for model evaluation and behavioral research. Scores on any single psychometric construct should not be interpreted as a comprehensive safety certification, nor should findings about specific models be generalized to the organizations or cultures associated with their development. We caution that comparison to human benchmarks is provided for calibration purposes only and should not be used to normalize or excuse systematic bias in language models. Finally, as with any measurement framework, there is a risk that optimizing directly against these metrics could produce models that score favorably without genuinely reducing the underlying biases the instruments are designed to detect.

## Acknowledgements

The authors acknowledge the use of AI writing assistants for proofreading and improving the overall clarity of the writing.

## References

Katherine A Batterton and Kimberly N Hale. 2017. The likert scale what it is and how to use it. Phalanx, 50(2):32–39.

Raphaël Bentegeac, Bastien Le Guellec, Grégory Kuchcinski, Philippe Amouyel, and Aghiles Hamroun. 2025. Token probabilities to mitigate large language models overconfidence in answering medical questions: quantitative study. Journal ofmedical Internet research, 27:e64348.

Mehar Bhatia, Shravan Nayak, Gaurav Kamath, Marius Mosbach, Vered Shwartz, Siva Reddy, and 1 others. 2025. Value drifts: Tracing value alignment during llm post-training. arXiv preprint arXiv:2510.26707.

Mehar Bhatia, Sahithya Ravi, Aditya Chinchure, EunJeong Hwang, and Vered Shwartz. 2024. From local concepts to universals: Evaluating the multicultural understanding of vision-language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 6763–6782, Miami, Florida, USA. Association for Computational Linguistics.

Mehar Bhatia and Vered Shwartz. 2023. GD-COMET: A geo-diverse commonsense inference model. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7993–8001, Singapore. Association for Computational Linguistics.

Rishi Bommasani, Percy Liang, and Tony Lee. 2023. Holistic evaluation of language models. Annals ofthe New York Academy ofSciences, 1525(1):140–146.

Chen Cecilia Liu, Fajri Koto, Timothy Baldwin, and Iryna Gurevych. 2024. Are multilingual LLMs culturally-diverse reasoners? an investigation into multicultural proverbs and sayings. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2016–2039, Mexico City, Mexico. Association for Computational Linguistics.

Jacob Cohen. 2013. Statistical power analysisfor the behavioral sciences. routledge.

Lee J Cronbach. 1950. Further evidence on response sets and test design. Educational and psychological measurement, 10(1):3–31.

Nicholas Deas and Kathleen McKeown. 2025. Artificial impressions: Evaluating large language model behavior through the lens of trait impressions. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 19418–19444, Suzhou, China. Association for Computational Linguistics.

Esin Durmus, Karina Nguyen, Thomas Liao, Nicholas Schiefer, Amanda Askell, Anton Bakhtin, Carol Chen, Zac Hatfield-Dodds, Danny Hernandez, Nicholas Joseph, Liane Lovitt, Sam McCandlish, Orowa Sikder, Alex Tamkin, Janel Thamkul, Jared Kaplan, Jack Clark, and Deep Ganguli. 2024. Towards measuring the representation of subjective global opinions in language models. In First Conference on Language Modeling.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630.

Jessica Kay Flake and Eiko I Fried. 2020. Measurement schmeasurement: Questionable measurement practices and how to avoid them. Advances in methods and practices in psychological science, 3(4):456– 465.

Chadi Helwe, Oana Balalau, and Davide Ceolin. 2025. Navigating the political compass: Evaluating multilingual LLMs across languages and nationalities. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 17179–17204, Vienna, Austria. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt.

2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Hang Jiang, Xiajie Zhang, Xubo Cao, Cynthia Breazeal, Deb Roy, and Jad Kabbara. 2024. PersonaLLM: Investigating the ability of large language models to express personality traits. In Findings ofthe Associationfor Computational Linguistics: NAACL 2024, pages 3605–3627, Mexico City, Mexico. Association for Computational Linguistics.

Jana Jung, Marlene Lutz, Indira Sen, and Markus Strohmaier. 2026. Do psychometric tests work for large language models? evaluation of tests on sexism, racism, and morality. In Proceedings of the 19th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8143–8173, Rabat, Morocco. Association for Computational Linguistics.

Mohsinul Kabir, Ajwad Abrar, and Sophia Ananiadou. 2025. Break the checkbox: Challenging closed-style evaluations of cultural alignment in LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 24–51, Suzhou, China. Association for Computational Linguistics.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, and 1 others. 2022. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221.

Anna Kruspe. 2024. Musical ethnocentrism in large language models. In Proceedings ofthe 3rd Workshop on NLPfor Music and Audio (NLP4MusA), pages 62– 68, Oakland, USA. Association for Computational Lingustics.

Tatsuki Kuribayashi, Yohei Oseki, and Timothy Baldwin. 2024. Psychometric predictive power of large language models. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 1983–2005, Mexico City, Mexico. Association for Computational Linguistics.

Xingxuan Li, Yutong Li, Lin Qiu, Shafiq Joty, and Lidong Bing. 2024. Evaluating psychological safety of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1826–1843, Miami, Florida, USA. Association for Computational Linguistics.

Chen Cecilia Liu, Hiba Arnaout, Nils Kovaciˇ c, Dana´ Atzil-Slonim, and Iryna Gurevych. 2026. Tailored emotional LLM-supporter: Enhancing cultural sensitivity. In Proceedings ofthe 19th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 535–574, Rabat, Morocco. Association for Computational Linguistics.

Zhuozhuo Joy Liu, Farhan Samir, Mehar Bhatia, Laura K. Nelson, and Vered Shwartz. 2025. Is

it bad to work all the time? cross-cultural evaluation of social norm biases in gpt-4. Preprint, arXiv:2505.18322.

Dominique Makowski, Mattan S. Ben-Shachar, and Daniel Lüdecke. 2019. bayestestr: Describing effects and their uncertainty, existence and significance within the bayesian framework. Journal of Open Source Software, 4(40):1541.

Tarek Naous and Wei Xu. 2025. On the origin of cultural biases in language models: From pre-training data to linguistic phenomena. In Proceedings of the 2025 Conference ofthe Nations ofthe Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6423–6443, Albuquerque, New Mexico. Association for Computational Linguistics.

Gian-Carlo Rota. 1964. On the foundations of combinatorial theory: I. theory of möbius functions. In Classic Papers in Combinatorics, pages 332–360. Springer.

Sougata Saha, Saurabh Kumar Pandey, Harshit Gupta, and Monojit Choudhury. 2025. Reading between the lines: Can LLMs identify cross-cultural communication gaps? In Proceedings of the 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8043–8067, Albuquerque, New Mexico. Association for Computational Linguistics.

Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. 2023. Whose opinions do language models reflect? In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 29971–30004. PMLR.

Mahesh N Shankarmahesh. 2006. Consumer ethnocentrism: an integrative review of its antecedents and consequences. International marketing review, 23(2):146–172.

Terence A Shimp and Subhash Sharma. 1987. Consumer ethnocentrism: Construction and validation of the cetscale. Journal of marketing research, 24(3):280–289.

Bangzhao Shu, Lechen Zhang, Minje Choi, Lavinia Dunagan, Lajanugen Logeswaran, Moontae Lee, Dallas Card, and David Jurgens. 2024. You don’t need a personality test to know these models are unreliable: Assessing the reliability of large language models on psychometric instruments. In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5263–5281, Mexico City, Mexico. Association for Computational Linguistics.

Vered Shwartz. 2022. Good night at 4 pm?! time expressions in different cultures. In Findings of the Associationfor Computational Linguistics: ACL 2022, pages 2842–2853, Dublin, Ireland. Association for Computational Linguistics.

Eshaan Tanwar, Anwoy Chatterjee, Michael Saxon, Alon Albalak, William Yang Wang, and Tanmoy Chakraborty. 2025. Do you know about my nation? investigating multilingual language models’ cultural literacy through factual knowledge. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 14956–14979, Suzhou, China. Association for Computational Linguistics.

William J Tastle and Mark J Wierman. 2007. Consensus and dissention: A measure of ordinal dispersion. International Journal ofApproximate Reasoning, 45(3):531–545.

Cédric Villani and 1 others. 2008. Optimal transport: old and new, volume 338. Springer.

Davood Wadi and Marc Fredette. 2025. A monte-carlo sampling framework for reliable evaluation of large language models using behavioral analysis. In The 2025 Conference on Empirical Methods in Natural Language Processing.

Davood Wadi, Renaud Legoux, Marc Fredette, and Sylvain Sénécal. 2026. The interplay of altruism and financial incentives: Maximizing online reviews through effective messaging. Journal ofElectronic Commerce Research, 27(2).

Davood Wadi and Yu Ma. 2026. Shopping by algorithm: How agentic ai deploys human heuristics as a surrogate consumer.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, and 1 others. 2024. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345.

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, and 1 others. 2025. The rise and potential of large language model based agents: A survey. Science China Information Sciences, 68(2):121101.

Wenlong Zhao, Debanjan Mondal, Niket Tandon, Danica Dillion, Kurt Gray, and Yuling Gu. 2024. World-ValuesBench: A large-scale benchmark dataset for multi-cultural value awareness of language models. In Proceedings ofthe 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 17696–17706, Torino, Italia. ELRA and ICCL.

## A Appendix

## A.1 Experiment Design

We restate the experimental design and notation used throughout the paper in a self-contained format here and provide the proofs.

We adopt a generalized fully crossed factorial experimental design. All independent variables, including language model identity, are treated as factors embedded within a unified design space.

Let ${ \mathcal { C } } = \{ 1 , \ldots , C \}$ denote the index set of experimental factors. Each factor $c \in { \mathcal { C } }$ admits a finite set of discrete levels, with cardinality $\Lambda _ { c } \in \mathbb { N }$ We denote the level selected for factor c by

$$
\lambda _ { c } \in \{ 1 , \ldots , \Lambda _ { c } \} .
$$

The complete experimental design is given by the Cartesian product of the level sets of all factors,

$$
{ \mathcal { D } } = \prod _ { c \in { \mathcal { C } } } \{ 1 , \dots \cdot , \Lambda _ { c } \} .\tag{3}
$$

For any subset of factors $U \subseteq { \mathcal { C } } ,$ , we denote $\pmb { \lambda } _ { U } : = ( \lambda _ { c } ) _ { c \in U }$ the vector of factor levels indexed by U. The vector of the levels for the complete factor set is hence denoted as $\lambda _ { \mathcal { C } }$ (abbreviated as λ).

The dependent variable is measured through a psychometric instrument consisting of K items. Each item is designed to elicit a response on a shared ordinal scale formalized as an ordered set $( \mathscr { V } , \preceq )$

$$
\mathcal { Y } = \{ y _ { \operatorname* { m i n } } , \dots , y _ { \operatorname* { m a x } } \} .\tag{4}
$$

Each prompt instructs the model to respond with a single digit only (one token) indicating the ordinal category, , with no additional text. We therefore treat the model’s response as the next-token distribution.

The joint response from all items, K, corresponds to a vector

$$
\mathbf { y } = ( y _ { 1 } , \dots , y _ { K } ) \in y ^ { K } .\tag{5}
$$

In behavioral experiments only a subset of the complete output space corresponds to valid responses. Let  denote the model’s full token vocabulary, and let $\nu _ { \mathrm { v a l } } \subset \nu$ denote the set of valid tokens for the task with the same cardinality as $\mathcal { V } ,$ $| \mathcal { V } _ { \mathrm { v a l } } | = | \mathcal { V } |$ . Let $\textstyle P _ { \mathrm { r a w } } ( t \mid x )$ denote the model’s next-token probability distribution given a prompt x, defined over the full vocabulary . We define the valid probability mass as

$$
P _ { \mathrm { v a l } } ( x ) = \sum _ { t \in \mathcal { V } _ { \mathrm { v a l } } } P _ { \mathrm { r a w } } ( t \mid x ) .\tag{6}
$$

To interface with the language model, we define a surjective mapping

$$
\phi : \mathcal { V } _ { \mathrm { v a l } } \to \mathcal { V }\tag{7}
$$

that accounts for tokenizer-specific variations of digits (e.g., leading spaces) and associates a specific subset of valid vocabulary tokens $\nu _ { \mathrm { v a l } } \subset \nu$ with their numeric values, . This mapping transforms the token distribution into a probability mass function over ordinal integers, allowing us to use ordinal dispersion measures to calculate the internal consistency of the model’s predictive distribution.

## A.2 Conditional Independence via Context Isolation

A central property of this design is that the model’s internal state is reset between items. For a fixed experimental condition λ, the response $Y _ { k , \lambda }$ for item k depends solely on the specific prompt $x _ { k , \lambda }$ and the model parameters. Following Wadi and Fredette (2025), we treat the items as conditionally independent given λ. Formally, for any realization vector $\mathbf { y } \in \mathcal { V } ^ { K }$ , the joint probability mass function factorizes as

$$
P ( \mathbf { Y } _ { \lambda } = \mathbf { y } \mid \lambda ) = \prod _ { k = 1 } ^ { K } P ( Y _ { k , \lambda } = y _ { k } \mid \lambda ) .\tag{8}
$$

This assumption justifies the use of product measures for joint distributions (Appendix A.3) and convolution for composite sums (Appendix A.4).

## A.3 The Consensus Layer

Having confirmed the model’s focus on valid outputs, we now superimpose the numeric, ordinal structure of the task onto the analysis. We require a measure that quantifies whether the model’s attitude towards the numeric scale is behaviorally consistent. Consider a single item with a 7-point Likert scale $( \mathcal { V } = \{ 1 , \dots , 7 \} )$ . A distribution assigning equal mass to opposing endpoints with the following probability mass functions (PMFs)

$$
P ( Y = 1 ) = 0 . 5 , \quad P ( Y = 7 ) = 0 . 5 ,
$$

exhibits dissension, as the model simultaneously expresses diametrically opposing views. In contrast, a distribution concentrated on adjacent scale

Table 3: Visualization of LLM Consensus (Cns) and Dissension (Dsn) cases. Each histogram depicts a characteristic output distribution on a 1–7 scale.  
![](images/579e6e6496881f4aef1cda0ba9a1c5671f8add9e1f68f41a08e6b539e5cfd0de.jpg)

points,

$$
P ( Y = 5 ) = 0 . 5 , \quad P ( Y = 6 ) = 0 . 5 ,
$$

is logically consistent, reflecting only local uncertainty regarding the precise intensity of the attitude.

To obtain a measure of consensus from a sample of individuals, Tastle and Wierman (2007) introduced Consensus (Cns) as

$$
\operatorname { C n s } ( Y ) = 1 + \sum _ { y \in \mathcal { Y } } \left[ P ( Y = y ) \times \log _ { 2 } \left( 1 - \frac { | y - \mu _ { Y } | } { d _ { \operatorname* { m a x } } } \right) \right]\tag{9}
$$

where $\begin{array} { r } { \mu _ { Y } = \sum _ { y \in \mathcal { Y } } { y P ( Y = y ) } } \end{array}$ is the expected value of the ordinal distribution and $d _ { \mathrm { m a x } } \ =$ $y _ { \mathrm { m a x } } - y _ { \mathrm { m i n } }$ . Table 3 illustrates Consensus on different PMFs. The issue with using entropy to measure consensus is shown in the 4th $( P ( 3 ) = 0 . 5 $ and $P ( 5 ) = 0 . 5 )$ and 5th row $( P ( 1 ) = 0 . 5 $ and $P ( 7 ) = 0 . 5 )$ , where the latter shows lower levels of consensus, but entropy gives the same value $( E n t r o p y = 1 . 0 0 )$

Whereas the original Consensus statistic is defined over a sample of aggregated scalar responses, our setting differs in two fundamental ways. First, with language models as subjects, we have direct access to the PMF of the data generation process rather than sampled responses. Second, behavioral instruments such as Likert scales typically consist of multiple items, each with varying levels of Consensus and Dissension. Calculating Consensus over aggregated responses disregards the inter-item Dissension.

![](images/b222849032fc0160b6fbbcf6bcd5604389236425c9efacac0503560171f2773f.jpg)  
Figure 6: The failure of aggregate Consensus in capturing between-item dissension. Distinct response patterns in a two-item Likert space (e.g., (4, 4) vs. (1, 7)) collapse to the same scalar score after aggregation. The original Consensus measure operates on this projection and cannot detect between-item dissension.

Standard aggregation procedures, such as summation or averaging, project this space onto a single scalar dimension. Figure 6 illustrates this effect for a two-item, 7-point Likert scale. Response patterns such as (4, 4), (1, 7), and (7, 1) all yield the same aggregate sum, causing the inter-item Dissension to be lost. However, these points occupy very different locations in the original two-dimensional response space, with (1, 7) and (7, 1) representing maximal disagreement between items.

Thus, we derive the Multivariate Consensus (Consensus from now on) across all items on a common ordinal scale. For each item $k \in \{ 1 , \ldots , K \}$ we utilize the mapping $\phi \ ( \mathrm { E q . \ 7 } )$ to project the renormalized token distribution onto the ordinal scale (Eq. 5). We define the item-level ordinal random variable $Y _ { k }$ with probability mass function

$$
P ( Y _ { k } = y ) = P \left( t = \phi ^ { - 1 } ( y ) \mid x _ { k } , t \in \mathcal { V } _ { \mathrm { v a l } } \right)\tag{10}
$$

where $x _ { k }$ is the prompt for item k. This step formally converts the model’s token prediction into a numeric behavioral response. We compute the expected response for each item, forming the centroid of the joint distribution

$$
\pmb { \mu } = ( \mu _ { 1 } , \ldots , \mu _ { K } ) ,\tag{11}
$$

where $\begin{array} { r } { \mu _ { k } = \sum _ { y \in \mathcal { Y } } y P ( Y _ { k } = y ) } \end{array}$ . The centroid, $\textstyle \mu ,$ represents the model’s expected position on each Likert item in the K-dimensional space. We define the maximum possible distance between any response vector and the centroid as the diagonal of the hypercube defined by the scale extrema

$$
d _ { \operatorname* { m a x } } = \sqrt { K \left( y _ { \operatorname* { m a x } } - y _ { \operatorname* { m i n } } \right) ^ { 2 } } .\tag{12}
$$

This normalizes distances to the unit interval and ensures comparability across instruments with different numbers of items. The Multivariate Consensus is defined as the expected logarithmic agreement of the joint distribution with respect to the centroid (Eq. 1).

Intuitively, this measure assigns higher consensus when the probability mass is concentrated near the centroid across all items, and lower consensus when mass is dispersed or polarized. Unlike entropy-based measures, Consensus is sensitive to ordinal distances. When $K = 1$ , the multivariate formulation reduces to the original Consensus (Tastle and Wierman, 2007), with the key distinction that we operate directly on the true PMF rather than on sampled responses. Multivariate Dissension (Dsn) is defined as the complement of Consensus, representing the degree of polarization or dispersion away from the mean.

$$
\operatorname { D s n } ( \mathbf { Y } ) = 1 - \operatorname { C n s } ( \mathbf { Y } )\tag{13}
$$

## A.4 The Construct Layer

Our goal is to quantify differences in the dependent variable across experimental conditions and models, while fully propagating uncertainty induced by the model’s generative process. We formalize inference over latent scale scores using an analytical fixed-effects ANOVA/Hoeffding decomposition. For a specific condition vector λ, the composite scale score, $S _ { \lambda }$ , is the aggregation of item-level random variable responses

$$
S _ { \lambda } = \sum _ { k = 1 } ^ { K } Y _ { k , \lambda } .\tag{14}
$$

Let $P _ { Y _ { k , \lambda } }$ denote the PMF of item k under condition λ. Given independence (Sec. A.2), the distribution of the composite score is obtained via discrete convolution

$$
P _ { S _ { \lambda } } = P _ { Y _ { 1 , \lambda } } \circledast P _ { Y _ { 2 , \lambda } } \circledast \dots \circledast P _ { Y _ { K , \lambda } } .\tag{15}
$$

## A.4.1 Factorial Decomposition Over Distributions

We formalize an analytical ANOVA/Hoeffding decomposition of the composite scale score. Our framework constructs a unique family of centered effect distributions $\{ E _ { U } \} _ { U \subseteq { \mathcal { C } } }$ , representing main effects and interactions, such that the total expected score satisfies the additive identity

$$
\mathbb { E } [ S _ { \lambda } ] = \underbrace { \mathbb { E } [ S _ { 0 } ] } _ { \mathrm { G r a n d M e a n } } + \sum _ { c \in \mathcal { C } } \underbrace { \mathbb { E } [ E _ { c } ( \lambda _ { c } ) ] } _ { \mathrm { M a i n F f f e c t s } } + \sum _ { U \subseteq \mathcal { C } } \underbrace { \mathbb { E } [ E _ { U } ( \lambda _ { U } ) ] } _ { \mathrm { I } \mathrm { I } \mathrm { I } \mathrm { e } \mathrm { r a c t i o n s } } ,\tag{16}
$$

while retaining the full probability mass function for every term in the summation.

For each experimental condition $\lambda \in { \mathcal { D } }$ , the composite scale score $S _ { \lambda }$ is a discrete random variable with probability mass function $P _ { S _ { \lambda } }$ (Eq. 14). Rather than reducing $S _ { \lambda }$ to a point estimate, we treat the mapping

$$
\lambda \mapsto P _ { S _ { \lambda } }\tag{17}
$$

as the primitive object of analysis. This allows all uncertainty induced by the model’s generative process to be propagated exactly through subsequent analyses. Given the fully crossed factorial design, all factor combinations are symmetric. Therefore, the design space  takes on a uniform probability distribution π, assigning equal weight to each experimental condition.

Assumption A.1 (Fully Crossed Fixed-Effects Design). The experimental design space $\begin{array} { r l } { \mathcal { D } } & { { } = } \end{array}$ $\Pi _ { c \in { \mathcal { C } } } \{ 1 , \ldots , \Lambda _ { c } \}$ is finite and fully crossed. All expectations over  are taken with respect to the uniform measure.

Proposition A.2 (Linearity of Expectation under Mixture Marginalization). Let  be a finite factorial design with probability measure π, and let ${ P _ { \bar { S } } } ~ = ~ { \mathbb E } _ { \lambda \sim \pi } [ { P _ { S } } _ { \lambda } ]$ be the mixture distribution. Then,

$$
\mathbb { E } [ \bar { S } ] = \mathbb { E } _ { \lambda \sim \pi } [ \mathbb { E } [ S _ { \lambda } ] ] .\tag{18}
$$

The proof follows immediately from the linearity of finite sums and the definition of expectation.

Corollary A.3 (Expectation of Factorial Marginal Distributions). Let $S _ { U } ( \lambda _ { U } )$ denote a factorial marginal distribution defined by marginalizing $P _ { S _ { \lambda } }$ over $\lambda _ { - U }$ under the uniform measure. Then

$$
\begin{array} { r } { \mathbb { E } [ S _ { U } ( \lambda _ { U } ) ] = \mathbb { E } _ { \lambda _ { - U } } [ \mathbb { E } [ S _ { \lambda } ] ] . } \end{array}
$$

The proof follows directly from Prop. A.2 by taking π to be the uniform distribution over λ<sub>−</sub> $- U$

## A.4.2 Grand Mixture Distribution

To establish a baseline for factorial effects, we construct the distributional equivalent of the classical ANOVA Grand Mean. We compute this as the Grand Mixture Distribution $( P _ { S _ { 0 } } )$ , obtained by averaging the composite score distributions uniformly across the entire design space

$$
P _ { S _ { 0 } } = \mathbb { E } _ { \lambda \sim \pi } [ P _ { S _ { \lambda } } ] = \frac { 1 } { | \mathcal { D } | } \sum _ { \lambda \in \mathcal { D } } P _ { S _ { \lambda } } .\tag{19}
$$

The random variable $S _ { 0 }$ encapsulates the global background distribution of the construct, encompassing all sources of variation in the experiment.

## A.4.3 Marginal Distributions

ANOVA Marginal Mean averages responses over complementary factors to isolate a specific level. We extend it by computing the Marginal Slice Distribution $( P _ { M _ { c } ( \lambda _ { c } ) } )$ . This is obtained by taking the uniform mixture of composite score distributions over all factors other than c,

$$
\begin{array} { l } { { \displaystyle P _ { M _ { c } ( \lambda _ { c } ) } = \mathbb { E } _ { \lambda _ { - c } } \Big [ P _ { S _ { ( \lambda _ { c } , \lambda _ { - c } ) } } \Big ] } } \\ { { \displaystyle \qquad = \frac { 1 } { | \mathscr { D } _ { - c } | } \sum _ { \lambda _ { - c } \in \mathscr { D } _ { - c } } P _ { S _ { ( \lambda _ { c } , \lambda _ { - c } ) } } . } } \end{array}\tag{20}
$$

The random variable $M _ { c } ( \lambda _ { c } )$ is the predictive distribution of the composite score when factor c is fixed at level $\lambda _ { c } ,$ averaging uniformly over all remaining factors. At the expectation level (Prop. A.2), E $\left[ M _ { c } ( \lambda _ { c } ) \right]$ ] equals the classical marginal mean.

Corollary A.4 (Grand mixture as an average of marginal slices). Under Assumption A.1, for any factor c,

$$
P _ { S _ { 0 } } = \frac { 1 } { \Lambda _ { c } } \sum _ { \lambda _ { c } = 1 } ^ { \Lambda _ { c } } P _ { M _ { c } ( \lambda _ { c } ) } .\tag{21}
$$

Proof. Fix $c \in { \mathcal { C } }$ . The grand mixture distribution is

$$
P _ { S _ { 0 } } = { \frac { 1 } { | { \mathcal { D } } | } } \sum _ { \lambda \in { \mathcal { D } } } P _ { S _ { \lambda } } .\tag{22}
$$

Under Assumption A.1, the design space factors as $\begin{array} { r c l } { \mathcal { D } } & { = } & { \{ 1 , \dots , \Lambda _ { c } \} \times \mathcal { D } _ { - c } , } \end{array}$ , where $\mathcal { D } _ { - c } : =$ $\Pi _ { d \in { \mathcal { C } } \backslash \{ c \} } \{ 1 , \dots , \Lambda _ { d } \}$ . Hence each $\lambda \ \in \ { \mathcal { D } }$ can be written uniquely as $\pmb { \lambda } = ( \lambda _ { c } , \lambda _ { - c } )$ with $\lambda _ { c } \in$ $\{ 1 , \ldots , \Lambda _ { c } \}$ and $\lambda _ { - c } \in \mathcal { D } _ { - c }$ . We may rewrite the sum in Eq. (22) as a double sum

$$
P _ { S _ { 0 } } = \frac { 1 } { | \mathcal { D } | } \sum _ { \lambda _ { c } = 1 } ^ { \Lambda _ { c } } \sum _ { \lambda _ { - c } \in \mathcal { D } _ { - c } } P _ { S _ { ( \lambda _ { c } , \lambda _ { - c } ) } } .\tag{23}
$$

Because the design is fully crossed, $\begin{array} { r l } { | \mathcal { D } | } & { { } = } \end{array}$ $\Lambda _ { c } \left| \mathcal { D } _ { - c } \right|$ . Substituting this into Eq. (23) gives

$$
P _ { S _ { 0 } } = \frac { 1 } { \Lambda _ { c } \left| \mathcal { D } _ { - c } \right| } \sum _ { \lambda _ { c } = 1 } ^ { \Lambda _ { c } } \sum _ { \lambda _ { - c } \in \mathcal { D } _ { - c } } P _ { S _ { ( \lambda _ { c } , \lambda _ { - c } ) } }\tag{24}
$$

$$
= \frac { 1 } { \Lambda _ { c } } \sum _ { \lambda _ { c } = 1 } ^ { \Lambda _ { c } } \left( \frac { 1 } { | \mathscr { D } _ { - c } | } \sum _ { \lambda _ { - c } \in \mathscr { D } _ { - c } } P _ { S _ { ( \lambda _ { c } } , \lambda _ { - c } ) } \right) .\tag{25}
$$

The expression in parentheses in Eq. (25) is exactly the marginal slice distribution $P _ { M _ { c } ( \lambda _ { c } ) }$ from Eq. (20). Therefore,

$$
P _ { S _ { 0 } } = \frac { 1 } { \Lambda _ { c } } \sum _ { \lambda _ { c } = 1 } ^ { \Lambda _ { c } } P _ { M _ { c } ( \lambda _ { c } ) } ,
$$

which proves Eq. (21).

## A.4.4 Expectation-Level Decomposition

Our primary estimands in the Construct Layer are predictive distributions (PMFs) and contrast distribution constructed via mixtures and paired subtraction. Because such contrasts depend on a chosen coupling, they do not, in general, admit an additive decomposition at the level of random variables. However, our distributional framework connects to classical fixed-effects ANOVA at the expectation level (Eq. 16). Let $\mu ( \lambda )$ be the deterministic mean surface over the factorial design

$$
\mu ( \lambda ) : = \mathbb { E } [ S _ { \lambda } ] , \qquad \lambda \in \mathcal { D } .\tag{26}
$$

Here, we state the Hoeffding/ANOVA decomposition of $\mu$ on the finite product space . We provide the unique collection of expectation-level effect components $\{ \mu _ { U } \} _ { U \subseteq { \mathcal C } }$ that our distributional effect estimands target in expectation.

Theorem A.5 (Expectation-Level ANOVA/Hoeffding Decomposition). Assume a finite fully crossed design  endowed with the uniform product measure (Assumption A.1). There exists a unique collection offunctions $\{ \mu _ { U } \} _ { U \subseteq { \mathcal C } }$ such that

$$
\mu ( \lambda ) = \sum _ { U \subseteq { \mathcal { C } } } \mu _ { U } ( \lambda _ { U } ) ,\tag{27}
$$

and each component satisfies the identifiability (sum-to-zero) constraints

$$
\sum _ { \lambda _ { c } = 1 } ^ { \Lambda _ { c } } \mu _ { U } \big ( \lambda _ { U } \big ) = 0 \qquad f o r a l l U \subseteq \mathscr { C } a n d a l l c \in U .\tag{28}
$$

Proof. We work on the finite product space $\mathcal { D }$ with the uniform product measure implied by Assumption A.1.

For any function $f : \mathcal { D }  \mathbb { I }$ R and any subset $U \subseteq$ , define the uniform marginalization operator

$$
( T _ { U } f ) ( \lambda _ { U } ) : = \mathbb { E } _ { \lambda _ { - U } } [ f ( \lambda _ { U } , \lambda _ { - U } ) ]\tag{29}
$$

$$
= \frac { 1 } { | \mathscr { D } _ { - U } | } \sum _ { \substack { \lambda _ { - U } \in \mathscr { D } _ { - U } } } f ( \lambda _ { U } , \lambda _ { - U } ) ,\tag{30}
$$

where $\begin{array} { r } { \mathcal { D } _ { - U } : = \prod _ { c \in \mathcal { C } \setminus U } \{ 1 , \dots , \Lambda _ { c } \} } \end{array}$ . By construction, $T _ { U } f$ depends only on $\lambda _ { U }$ , and $T c f = f$

Define components recursively in increasing or-<sup>der</sup> <sup>of</sup> |<sup>U</sup>|

$$
\mu _ { \emptyset } : = ( T _ { \emptyset } \mu ) \qquad \mathrm { ( a c o n s t a n t ) } ,\tag{31}
$$

$$
\mu _ { U } ( \pmb { \lambda } _ { U } ) : = ( T _ { U } \mu ) ( \pmb { \lambda } _ { U } ) - \sum _ { V \subsetneq U } \mu _ { V } ( \pmb { \lambda } _ { V } ) ,\tag{32}
$$

$$
U \neq \varnothing .
$$

This is well-defined because the right-hand side for µ<sub>U</sub> only involves $\mu { _ { V } }$ with strictly smaller index sets $V \subsetneq U$ , which have already been defined.

From Eq. (31) we immediately obtain, for every $U \subseteq { \mathcal { C } } ,$

$$
( T _ { U } \mu ) ( \lambda _ { U } ) = \sum _ { V \subseteq U } \mu _ { V } ( \lambda _ { V } ) .\tag{33}
$$

For $U = \emptyset$ this holds by definition. For $U \neq \emptyset$ it is a simple rearrangement of (31). Setting $U = { \mathcal { C } }$ and using $T _ { \mathit { C } } \mu = \mu$ yields

$$
\mu ( \lambda ) = \sum _ { V \subseteq { \mathcal C } } \mu _ { V } ( \lambda _ { V } ) ,
$$

which is Eq. (27).

Identifiability. Fix $U \subseteq { \mathcal { C } }$ and $c \in U$ . Uniformly averaging $\mu _ { U }$ over $\lambda _ { c }$ gives

$$
\mathbb { E } _ { \lambda _ { c } } [ \mu _ { U } ( \lambda _ { U } ) ] = \mathbb { E } _ { \lambda _ { c } } [ ( T _ { U } \mu ) ( \lambda _ { U } ) ] - \sum _ { V \subseteq U } \mathbb { E } _ { \lambda _ { c } } [ \mu _ { V } ( \lambda _ { V } ) ] .\tag{34}
$$

Because the measure is a product measure, averaging $T _ { U } \mu$ over $\lambda _ { c }$ removes coordinate c from the conditioning,

$$
\operatorname { \mathbb { E } } _ { \lambda _ { c } } [ ( T _ { U } \mu ) ( \lambda _ { U } ) ] = ( T _ { U \setminus \{ c \} } \mu ) ( \lambda _ { U \setminus \{ c \} } ) .
$$

For $V \subsetneq U , { \mathrm { i f } } c \in V$ then $\mathbb { E } _ { \lambda _ { c } } [ \mu _ { V } ] = 0$ by induction on $| V | . \operatorname { I f } c \notin V$ then $\mu { _ { V } }$ does not depend on $\lambda _ { c }$ and the average leaves it unchanged. Hence,

$$
\mathbb { E } _ { \lambda _ { c } } [ \mu _ { U } ( \lambda _ { U } ) ] = ( T _ { U \setminus \{ c \} } \mu ) ( \lambda _ { U \setminus \{ c \} } ) - \sum _ { V \subseteq U \setminus \{ c \} } \mu _ { V } ( \lambda _ { V } ) .\tag{35}
$$

The right-hand side is zero by applying Eq. (33) with $U \setminus \{ c \}$ . Therefore $\mathbb { E } _ { \lambda _ { c } } [ \mu _ { U } ( \lambda _ { U } ) ] = 0$ , which under uniformity is equivalent to Eq. (28).

Uniqueness. Suppose $\{ \tilde { \mu } _ { U } \}$ is another collection satisfying Eq. (27) and Eq. (28). Define $h _ { U } : =$ $\tilde { \mu } { \boldsymbol { U } } - \mu { \boldsymbol { U } }$ . Then $\begin{array} { r } { \sum _ { U } h _ { U } ( \lambda _ { U } ) = 0 } \end{array}$ for all λ and each h satisfies the same sum-to-zero constraints. Apply T<sub>U</sub> to $\textstyle \sum _ { W } h _ { W } = 0$ , where W is any subset of . If $W \subseteq U$ , then $T _ { U } h _ { W } = h _ { W }$ . If $W \not \subseteq U$ pick $c \in W \setminus U$ . Marginalizing over $- U$ averages over $\lambda _ { c } ,$ and the sum-to-zero constraint implies $T _ { U } h _ { W } = 0$ . Thus,

$$
0 = T _ { U } \left[ \sum _ { W \subseteq \mathcal { C } } h _ { W } \right] = \sum _ { W \subseteq U } h _ { W } \qquad \mathrm { f o r ~ a l l } \ U \subseteq \mathcal { C } .
$$

Taking $U = \emptyset$ gives $h _ { \emptyset } = 0$ . Proceed by induction on $| U |$ . Assuming $h _ { V } ~ = ~ 0$ for all $V \subsetneq U$ , the identity above implies $h _ { U } = 0$ . Hence $h _ { U } \equiv 0$ for all U, proving uniqueness. □

Corollary A.6 (Möbius inversion form). The components in Theorem A.5 admit the explicit inclusion– exclusionform

$$
\mu _ { U } ( \mathbf { \lambda } _ { U } ) = \sum _ { V \subseteq U } ( - 1 ) ^ { | U | - | V | } ( T _ { V } \mu ) ( \mathbf { \lambda } _ { V } ) , \qquad U \subseteq { \mathcal { C } } .\tag{36}
$$

Proof. Equation (33) states that for each fixed $U ,$ $\begin{array} { r } { ( T _ { U } \mu ) ~ = ~ \sum _ { V \subseteq U } \mu _ { V } } \end{array}$ . This is exactly the zetatransform relation on the Boolean lattice $( 2 ^ { \mathcal { C } } , \subseteq )$ Applying Möbius inversion (Rota 1964) yields Eq. (36). □

Remark A.7. By Prop. A.2 and the definition of the global marginal slice distribution $M _ { U } ( \lambda _ { U } )$ (Eq. (60)),

$$
\mathbb { E } [ M _ { U } ( \lambda _ { U } ) ] = ( T _ { U } \mu ) ( \lambda _ { U } ) .
$$

Thus, $T _ { U } \mu$ can be interpreted as the classical marginal mean surface, and $\mu _ { U }$ in Eq. (36) are the associated ANOVA/Hoeffding effect components targeted by our distributional effect estimands in expectation.

## A.4.5 Main Effects Distribution

Classical ANOVA main effects quantify centered deviations of a factor-level marginal mean from the grand mean. In our distributional setting, we seek an effect distribution whose expectation recovers the ANOVA main effect (Eq. 16). There are various choices for computing the difference distribution. One option is to compute the convolution-based subtraction $M _ { c } ( \lambda _ { c } ) - S _ { 0 }$ . The main problem with this derivation is that it treats the slice distribution of the factor, $M _ { c } ( \lambda _ { c } )$ , and the grand mixture, $S _ { 0 } .$ , as independent. This is an incorrect assumption because $S _ { 0 }$ and $M _ { c } ( \lambda _ { c } )$ are clearly correlated $( M _ { c } ( \lambda _ { c } )$ is a component of the mixture $S _ { 0 } )$

Moreover, convolution of two random variables adds their variance together. This inflates the variance associated with the main effect of factor c. Specifically, the convolution-based subtraction $M _ { c } ( \lambda _ { c } ) - S _ { 0 }$ corresponds to a between-subjects contrast (difference of two independent draws), which can yield substantial variability even when the two distributions are identical.

We therefore derive the main-effect distributions by minimizing the variance of the paired difference between the specific level and the baseline. This approach isolates the systematic shift attributable to the factor while removing the "noise" common to both distributions. To achieve this, we employ a comonotone coupling (i.e., 1D monotone optimal transport), which maximizes the covariance between paired draws (Villani et al., 2008).

The pairing has two components. We have a blocked construction that respects the factorial design by holding the complementary factors $\lambda _ { - c }$ fixed. Within each block, we have a canonical comonotone coupling, which minimizes the expected squared difference between paired draws among all couplings with the same marginals.

Definition A.8 (Paired difference of two univariate discrete distributions). Let X and Y be univariate discrete random variables with finite support and CDFs $F _ { X }$ and $F _ { Y }$ . Let $U \sim \mathrm { U n i f } ( 0 , 1 )$ and define the (left-continuous) generalized inverse CDF

$$
F ^ { - 1 } ( u ) : = \operatorname* { i n f } \{ z \in \mathbb { R } : F ( z ) \geq u \} .
$$

The comonotone coupling is given by $X ^ { \uparrow } : =$ $F _ { X } ^ { - 1 } ( U )$ and $Y ^ { \uparrow } : = F _ { Y } ^ { - 1 } ( U )$ . We define the paired difference random variable as

$$
X \ominus Y : = X ^ { \uparrow } - Y ^ { \uparrow } .\tag{37}
$$

This construction preserves marginals $( X ^ { \uparrow } \ { \overset { d } { = } } \ X$ $Y ^ { \uparrow } \overset { d } { = } Y )$ , hence E $[ X \ominus Y ] = \mathbb { E } [ X ] - \mathbb { E } [ Y ]$

Let supp $( X ) = \{ x _ { 1 } < \cdots < x _ { m } \}$ with masses $p _ { i } = P ( X = x _ { i } )$ and define $\begin{array} { r } { P _ { i } = \sum _ { r = 1 } ^ { i } p _ { r } } \end{array}$ with $P _ { 0 } = 0$ . Similarly, let $\operatorname { s u p p } ( Y ) = \{ y _ { 1 } < \cdots <$ $y _ { n } \}$ with masses $q _ { j }$ and $\begin{array} { r } { Q _ { j } = \sum _ { s = 1 } ^ { j } q _ { s } } \end{array}$ with $Q _ { 0 } =$ 0. The comonotone coupling assigns joint mass

$$
w _ { i j } = \mathrm { m a x } \Big \{ 0 , \mathrm { m i n } ( P _ { i } , Q _ { j } ) - \mathrm { m a x } ( P _ { i - 1 } , Q _ { j - 1 } ) \Big \} ,
$$

$$
i = 1 , \ldots , m , \ j = 1 , \ldots , n .\tag{38}
$$

(39)

The paired-difference PMF is the pushforward under $( x , y ) \mapsto x - y ,$

$$
P _ { X \ominus Y } ( z ) = \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } w _ { i j } \parallel \{ z = x _ { i } - y _ { j } \} .\tag{40}
$$

All terms are finite and are computed deterministically from the two input PMFs.

To respect the factorial dependence structure $( \mathrm { i } . \mathrm { e } . , S _ { 0 }$ is constructed from the same family $\{ S _ { \lambda } \}$ as $M _ { c } ( \lambda _ { c } ) )$ , we form a grand-mean baseline within each block $\lambda _ { - c }$ by averaging only over the levels of factor c

$$
P _ { S _ { 0 | \lambda _ { - c } } } : = \frac { 1 } { \Lambda _ { c } } \sum _ { \lambda _ { c } ^ { \prime } = 1 } ^ { \Lambda _ { c } } P _ { S _ { ( \lambda _ { c } ^ { \prime } , \lambda _ { - c } ) } } .\tag{41}
$$

Mixing these blocked baselines uniformly over $\lambda _ { - c }$ recovers the original grand mixture distribution $\begin{array} { r } { \frac { 1 } { | \mathcal { D } _ { - c } | } \sum _ { \lambda _ { - c } } P _ { S _ { 0 | \lambda _ { - c } } } = P _ { S _ { 0 } } } \end{array}$

The (centered) main-effect distribution for factor c at level $\lambda _ { c }$ is the uniform mixture ofpaired withinblock contrasts

$$
E _ { c } ( \lambda _ { c } ) : = \mathbb { E } _ { \lambda _ { - c } } \bigl [ S _ { ( \lambda _ { c } , \lambda _ { - c } ) } \ \ominus \ S _ { 0 | \lambda _ { - c } } \bigr ] \ .\tag{42}
$$

At the PMF level this becomes

$$
P _ { E _ { c } ( \lambda _ { c } ) } = \frac { 1 } { | \mathscr { D } _ { - c } | } \sum _ { \lambda _ { - c } \in \mathscr { D } _ { - c } } P _ { S _ { ( \lambda _ { c } , \lambda _ { - c } ) } \ominus S _ { 0 | \lambda _ { - c } } } .\tag{43}
$$

Each paired-difference PMF in Eq. (43) is computed exactly via the mass-matching formula in Eq. (40). No estimation or approximation is used at any stage.

Corollary A.9 (Expectation-level centering and sum-to-zero). For eachfactor $c \in { \mathcal { C } }$ and level $\lambda _ { c } ,$

$$
\mathbb { E } [ E _ { c } ( \lambda _ { c } ) ] = \mathbb { E } [ M _ { c } ( \lambda _ { c } ) ] - \mathbb { E } [ S _ { 0 } ] .\tag{44}
$$

Moreover, the paired main effects satisfy the standard ANOVA sum-to-zero constraint in expectation

$$
\sum _ { \lambda _ { c } = 1 } ^ { \Lambda _ { c } } \mathbb { E } [ E _ { c } ( \lambda _ { c } ) ] = 0 .\tag{45}
$$

Proof. By definition (Eq. (43)), the main-effect random variable $E _ { c } ( \lambda _ { c } )$ is the uniform mixture over blocks $\lambda _ { - c } \in \mathcal { D } _ { - c }$ of the paired contrast $S _ { ( \lambda _ { c } , \lambda _ { - c } ) } \ominus S _ { 0 | \lambda _ { - c } }$ . Equivalently, we can represent the mixture hierarchically. Draw $\lambda _ { - c } ~ \sim$ $\mathrm { U n i f } ( \mathcal { D } _ { - c } )$ , then draw

$$
\left( S _ { ( \lambda _ { c } , \lambda _ { - c } ) } ^ { \uparrow } , \ S _ { 0 | \lambda _ { - c } } ^ { \uparrow } \right)
$$

from the comonotone coupling.

Set $E _ { c } ( \lambda _ { c } ) = S _ { ( \lambda _ { c } , \lambda _ { - c } ) } ^ { \uparrow } - S _ { 0 | \lambda _ { - c } } ^ { \uparrow }$ . Because $\mathcal { D } _ { - c }$ is finite, we may apply iterated expectations.

$$
\mathbb { E } [ E _ { c } ( \lambda _ { c } ) ] = \frac { 1 } { | \mathcal { D } _ { - c } | } \sum _ { \lambda _ { - c } \in \mathcal { D } _ { - c } } \mathbb { E } \big [ S _ { ( \lambda _ { c } , \lambda _ { - c } ) } \ominus S _ { 0 | \lambda _ { - c } } \big ] .\tag{46}
$$

Fix a block λ<sub>−c</sub>. By Def. A.8,

$$
S _ { ( \lambda _ { c } , \lambda _ { - c } ) } \ominus S _ { 0 | { \bf \lambda } _ { - c } } = S _ { ( \lambda _ { c } , \lambda _ { - c } ) } ^ { \uparrow } - S _ { 0 | { \bf \lambda } _ { - c } } ^ { \uparrow } ,
$$

where $S _ { ( \lambda _ { c } , \lambda _ { - c } ) } ^ { \uparrow } \overset { d } { = } S _ { ( \lambda _ { c } , \lambda _ { - c } ) }$ and $S _ { 0 | \lambda _ { - c } } ^ { \uparrow } \stackrel { d } { = } S _ { 0 | \lambda _ { - c } } .$ Therefore the expectations match their marginals

$$
\mathbb { E } \Big [ S _ { ( \lambda _ { c } , \lambda _ { - c } ) } ^ { \uparrow } \Big ] = \mathbb { E } \big [ S _ { ( \lambda _ { c } , \lambda _ { - c } ) } \big ] ,\tag{47}
$$

$$
\mathbb { E } \Big [ S _ { 0 | \lambda _ { - c } } ^ { \uparrow } \Big ] = \mathbb { E } \big [ S _ { 0 | \lambda _ { - c } } \big ] .\tag{48}
$$

Using linearity of expectation, which does not require independence,

$$
\begin{array} { r } { \mathbb { E } \left[ S _ { ( \lambda _ { c } , \lambda _ { - c } ) } \ominus S _ { 0 | \pmb { \lambda _ { - c } } } \right] = \mathbb { E } \left[ S _ { ( \lambda _ { c } , \pmb { \lambda _ { - c } } ) } \right] - \mathbb { E } \left[ S _ { 0 | \pmb { \lambda _ { - c } } } \right] . } \end{array}\tag{49}
$$

Eq. (49) holds for any coupling with the given marginals. The comonotone coupling is chosen to minimize dispersion, but preserves the mean.

Substitute Eq. (49) into Eq. (46):

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ E _ { c } ( \lambda _ { c } ) ] = \frac { 1 } { | \mathcal { D } _ { - c } | } \sum _ { \lambda _ { - c } \in \mathcal { D } _ { - c } } \left( \mathbb { E } \big [ S _ { ( \lambda _ { c } , \lambda _ { - c } ) } \big ] - \mathbb { E } \big [ S _ { 0 | \lambda _ { - c } } \big ] \right. } \\ { \displaystyle \qquad = \underbrace { \frac { 1 } { | \mathcal { D } _ { - c } | } \sum _ { \lambda _ { - c } \in \mathcal { D } _ { - c } } \mathbb { E } \big [ S _ { ( \lambda _ { c } , \lambda _ { - c } ) } \big ] } _ { ( \lambda ) } } \\ { \displaystyle \qquad - \underbrace { \frac { 1 } { | \mathcal { D } _ { - c } | } \sum _ { \lambda _ { - c } \in \mathcal { D } _ { - c } } \mathbb { E } \big [ S _ { 0 | \lambda _ { - c } } \big ] } _ { ( \mathtt { j } ) } . } \end{array}\tag{}
$$

For (⋆), by the definition of the marginal slice distribution $P _ { M _ { c } ( \lambda _ { c } ) }$ (Eq. (20)) and Prop. A.2,

$$
\mathbb { E } [ M _ { c } ( \lambda _ { c } ) ] = \frac { 1 } { | { \mathcal D } _ { - c } | } \sum _ { \lambda _ { - c } \in { \mathcal D } _ { - c } } \mathbb { E } \big [ S _ { ( \lambda _ { c } , \lambda _ { - c } ) } \big ] .
$$

For ( ), expand the blocked grand mean (Eq. (41)) and apply Theorem A.2,

$$
\frac { 1 } { | \mathscr { D } _ { - c } | } \sum _ { \lambda _ { - c } } \mathbb { E } \left[ S _ { 0 | \lambda _ { - c } } \right] = \frac { 1 } { | \mathscr { D } _ { - c } | } \sum _ { \lambda _ { - c } } \mathbb { E } \left[ \frac { 1 } { \Lambda _ { c } } \sum _ { \lambda _ { c } ^ { \prime } = 1 } ^ { \Lambda _ { c } } S _ { ( \lambda _ { c } ^ { \prime } , \lambda _ { - c } ) } \right]\tag{52}
$$

$$
= \frac { 1 } { | \mathscr { D } _ { - c } | } \sum _ { \lambda _ { - c } } \frac { 1 } { \Lambda _ { c } } \sum _ { \lambda _ { c } ^ { \prime } = 1 } ^ { \Lambda _ { c } } \mathbb { E } \big [ S _ { ( \lambda _ { c } ^ { \prime } , \lambda _ { - c } ) } \big ]\tag{53}
$$

$$
= { \frac { 1 } { | { \mathcal { D } } | } } \sum _ { \lambda \in { \mathcal { D } } } \operatorname { \mathbb { E } } [ S _ { \lambda } ] = \operatorname { \mathbb { E } } [ S _ { 0 } ] .\tag{54}
$$

We used $| \mathcal { D } | = \Lambda _ { c } | \mathcal { D } _ { - c } |$ and the definition of the grand mixture S<sub>0</sub> (Eq. (19)).

Substituting these two identifications into Eq. (51) yields

$$
\mathbb { E } [ E _ { c } ( \lambda _ { c } ) ] = \mathbb { E } [ M _ { c } ( \lambda _ { c } ) ] - \mathbb { E } [ S _ { 0 } ] ,
$$

which proves Eq. (44).

Sum Eq. (46) over $\lambda _ { c } \in \{ 1 , \dots , \Lambda _ { c } \}$

$$
\sum _ { \lambda _ { c } = 1 } ^ { \Lambda _ { c } } \mathbb { E } [ E _ { c } ( \lambda _ { c } ) ] = \frac { 1 } { | \mathscr { D } _ { - c } | } \sum _ { \lambda _ { - c } \in \mathscr { D } _ { - c } } \sum _ { \lambda _ { c } = 1 } ^ { \Lambda _ { c } }\tag{55}
$$

$$
\mathbb { E } \left[ S _ { ( \lambda _ { c } , \lambda _ { - c } ) } \ominus S _ { 0 | \lambda _ { - c } } \right] .\tag{56}
$$

Apply Eq. (49) inside the inner sum

$$
\sum _ { \lambda _ { c } = 1 } ^ { \Lambda _ { c } } \mathbb { E } \big [ S _ { ( \lambda _ { c } , \lambda _ { - c } ) } \ominus S _ { 0 | \lambda _ { - c } } \big ] =\tag{57}
$$

$$
\sum _ { \lambda _ { c } = 1 } ^ { \Lambda _ { c } } \left( \mathbb { E } [ S _ { ( \lambda _ { c } , \lambda _ { - c } ) } ] - \mathbb { E } [ S _ { 0 | \lambda _ { - c } } ] \right) =\tag{58}
$$

$$
\sum _ { \lambda _ { c } = 1 } ^ { \Lambda _ { c } } \mathbb { E } \big [ S _ { ( \lambda _ { c } , \lambda _ { - c } ) } \big ] - \Lambda _ { c } \mathbb { E } \big [ S _ { 0 | \lambda _ { - c } } \big ] .\tag{59}
$$

But by definition of the blocked grand mean and linearity of expectation,

$$
\mathbb { E } [ S _ { 0 | \lambda _ { - c } } ] = \frac { 1 } { \Lambda _ { c } } \sum _ { \lambda _ { c } ^ { \prime } = 1 } ^ { \Lambda _ { c } } \mathbb { E } [ S _ { ( \lambda _ { c } ^ { \prime } , \lambda _ { - c } ) } ] ,
$$

so $\begin{array} { r } { \Lambda _ { c } \mathbb { E } \big [ S _ { 0 | \lambda _ { - c } } \big ] = \sum _ { \lambda _ { c } ^ { \prime } = 1 } ^ { \Lambda _ { c } } \mathbb { E } \big [ S _ { ( \lambda _ { c } ^ { \prime } , \lambda _ { - c } ) } \big ] } \end{array}$ . Hence the inner expression is identically zero for every $\lambda _ { - c } ,$ and therefore the outer average is also zero

$$
\sum _ { \lambda _ { c } = 1 } ^ { \Lambda _ { c } } \mathbb { E } [ E _ { c } ( \lambda _ { c } ) ] = 0 ,
$$

which proves Eq. (45).

## A.4.6 Interaction Effects Distribution

Whereas main effects quantify additive shifts attributable to individual factors, interaction effects quantify departures from additivity when multiple factors are fixed jointly. In classical fixed-effects ANOVA/Hoeffding decompositions, the U-way interaction component is obtained by Möbius inversion of marginal means on the Boolean lattice of factor subsets.

In our framework, since we are dealing directly with PMFs of the population, we define interaction effects using a paired, design-blocked contrast. We hold the complementary factors $\lambda _ { - U }$ fixed (blocking). Within each block, we couple all terms in the alternating sum via a shared latent quantile variable (comonotone coupling). This yields a paired distribution for the interaction contrast.

For any nonempty subset of factors $U \subseteq { \mathcal { C } }$ and level vector $\lambda _ { U }$ , define the U-way marginal slice distribution as the uniform mixture over all remaining factors

$$
P _ { M _ { U } ( \mathbf { \lambda } \mathbf { \lambda } \mathbf { \lambda } \mathbf { \lambda } \mathbf { \lambda } ) } = \frac { 1 } { | \mathcal { D } _ { - U } | } \sum _ { \substack { \lambda _ { - U } \in \mathcal { D } _ { - U } } } P _ { S _ { ( \lambda _ { U } , \lambda _ { - U } ) } . }\tag{60}
$$

We further set $M _ { \emptyset } : = S _ { 0 }$

Fix a nonempty subset of factors $U \subseteq { \mathcal { C } }$ and a realization of the complementary factors $\lambda _ { - U } \in$ $\mathcal { D } _ { - U }$ . For any $V \subseteq U$ , define the within-block design space

$$
{ \mathcal { D } } _ { U \backslash V } : = \prod _ { c \in U \backslash V } \{ 1 , \dots , \Lambda _ { c } \} .
$$

We derive the blocked marginal slice distribution as the uniform mixture over the remaining factors in $U \backslash V$ , holding $\lambda _ { - U }$ fixed

$$
P _ { M _ { V | \lambda _ { - U } } ( \lambda _ { V } ) }\tag{61}
$$

$$
: = \frac { 1 } { | { \mathcal D } _ { U \backslash V } | } \sum _ { \lambda _ { U \backslash V } \in { \mathcal D } _ { U \backslash V } } P _ { S _ { ( \lambda _ { V } , \lambda _ { U \backslash V } , \lambda _ { - U } ) } . } \qquad\tag{62}
$$

This unifies several cases,

$$
M _ { U | \lambda _ { - U } } ( \lambda _ { U } ) \equiv S _ { ( \lambda _ { U } , \lambda _ { - U } ) } ,\tag{63}
$$

$$
M _ { \emptyset | \lambda _ { - U } } = \frac { 1 } { | \mathscr { D } _ { U } | } \sum _ { \lambda _ { U } \in \mathscr { D } _ { U } } S _ { ( \lambda _ { U } , \lambda _ { - U } ) } ,\tag{64}
$$

where $M _ { \emptyset | \lambda _ { - U } }$ is the blocked grand mean (over factors $U )$ within the fixed complement $\lambda _ { - U }$

Averaging blocked slices over $\lambda _ { - U }$ recovers the global marginal slices from Eq. (60).

$$
P _ { M _ { V } ( \mathbf { \lambda } \mathbf { \lambda } \mathbf { \lambda } \mathbf { \lambda } _ { V } ) } = \frac { 1 } { | \mathcal { D } _ { - U } | } \sum _ { \substack { \lambda _ { - U } \in \mathcal { D } _ { - U } } } P _ { M _ { V } | \mathbf { \lambda } \mathbf { \lambda } _ { - U } } ( \mathbf { \lambda } _ { V } ) \cdot\tag{65}
$$

In particular, taking $\begin{array} { r l r l } { V } & { { } } & { = } & { { } } & { \emptyset } \end{array}$ yields $\begin{array} { r } { \frac { 1 } { | \mathcal { D } _ { - U } | } \sum _ { \lambda _ { - U } } P _ { M _ { \emptyset | \lambda _ { - U } } } = P _ { S _ { 0 } } } \end{array}$

To define a distribution for an alternating sum without imposing independence, we use a sharedquantile (comonotone) coupling.

Definition A.10 (Comonotone-coupled linear contrast). Let $\{ X _ { r } \} _ { r = 1 } ^ { R }$ be univariate discrete random variables with finite support and CDFs $\{ F _ { r } \} _ { r = 1 } ^ { R }$ Let $\begin{array} { r l r } { U ^ { * } } & { { } \sim } & { \mathrm { U n i f } ( 0 , 1 ) } \end{array}$ and define $X _ { r } ^ { \uparrow } : = F _ { r } ^ { - 1 } ( U ^ { * } )$ using the (left-continuous) generalized inverse from Def. A.8. For coefficients ${ \pmb { \alpha } } = ( \alpha _ { 1 } , \dots , \alpha _ { R } ) \in \mathbb { R } ^ { R }$ , define the comonotonecoupled linear contrast

$$
\mathcal { L } ^ { \uparrow } ( \pmb { \alpha } ; X _ { 1 } , \ldots , X _ { R } ) : = \sum _ { r = 1 } ^ { R } \alpha _ { r } X _ { r } ^ { \uparrow } .\tag{66}
$$

For $R = 2$ and $\pmb { \alpha } = ( 1 , - 1 )$ , this reduces to the paired difference $X _ { 1 } \ominus X _ { 2 }$ (Def. A.8).

Because each $X _ { r } ^ { \uparrow } = F _ { r } ^ { - 1 } ( U ^ { * } )$ is a step function of $U ^ { * }$ , the contrast in Eq. (66) is piecewise constant on a finite partition of [0, 1]. Let

$$
{ \mathcal B } : = \{ 0 , 1 \} \cup \bigcup _ { r = 1 } ^ { R } \left\{ F _ { r } ( x ) : x \in \operatorname { s u p p } ( X _ { r } ) \right\} ,
$$

and let $0 = u _ { 0 } < u _ { 1 } < \cdot \cdot \cdot < u _ { L } = 1$ be the sorted distinct elements of . For $\ell = 1 , \ldots , L ,$ , define

$$
z _ { \ell } : = \sum _ { r = 1 } ^ { R } \alpha _ { r } F _ { r } ^ { - 1 } ( u _ { \ell } ) .
$$

Then the PMF of $Z : = \mathcal L ^ { \uparrow } ( \pmb { \alpha } ; X _ { 1 } , \dots , X _ { R } )$ is

$$
P _ { Z } ( z ) = \sum _ { \ell = 1 } ^ { L } ( u _ { \ell } - u _ { \ell - 1 } ) \mathbb { 1 } \{ z = z _ { \ell } \} ,\tag{67}
$$

aggregating masses for identical $z _ { \ell }$ values. This computation is exact and deterministic.

Fix $U \subseteq { \mathcal { C } }$ with $| U | \ge 2$ . For a given level vector $\lambda _ { U }$ and block $\lambda _ { - U }$ , define the within-block interaction-effect random variable as the paired Möbius contrast over blocked slices

$$
H _ { U | \lambda _ { - U } } ( \lambda _ { U } ) : = \sum _ { V \subseteq U } ( - 1 ) ^ { | U | - | V | } \left( M _ { V | \lambda _ { - U } } ( \lambda _ { V } ) \right) ^ { \uparrow } ,\tag{68}
$$

where all terms are coupled comonotonically via the same $U ^ { * }$ in Def. A.10. The PMF of $H _ { U | \lambda _ { - U } } ( \lambda _ { U } )$ is computed exactly using Eq. (67) with $R = 2 ^ { | U | }$ and coefficients $\alpha _ { V } = ( - 1 ) ^ { | U | - | V | }$

Finally, we define the global U-way interaction effect distribution by mixing uniformly over complementary factors:

$$
P _ { E _ { U } ( \mathbf { \lambda } \mathbf { \lambda } \mathbf { \lambda } _ { U } ) } = \frac { 1 } { | \mathscr { D } _ { - U } | } \sum _ { \substack { \lambda _ { - U } \in \mathscr { D } _ { - U } } } P _ { H _ { U } | \mathbf { \lambda } _ { - U } } ( \mathbf { \lambda } _ { U } ) \cdot\tag{69}
$$

Two-way interactions. For $U = \{ a , b \}$ , within a fixed block $\lambda _ { - a b }$ the paired two-way interaction is

$$
H _ { \{ a , b \} | \lambda _ { - a b } } ( \lambda _ { a } , \lambda _ { b } ) = S _ { ( \lambda _ { a } , \lambda _ { b } , \lambda _ { - a b } ) } ^ { \uparrow }\tag{70}
$$

$$
- \ M _ { a | \lambda _ { - a b } } ^ { \uparrow } ( \lambda _ { a } )\tag{71}
$$

$$
- \ M _ { b | \lambda _ { - a b } } ^ { \uparrow } ( \lambda _ { b } )\tag{72}
$$

$$
+ M _ { \varnothing | \lambda _ { - a b } } ^ { \uparrow } ,\tag{73}
$$

and $P _ { E _ { \{ a , b \} } ( \lambda _ { a } , \lambda _ { b } ) }$ is the uniform mixture of these within-block PMFs over $\lambda _ { - a b }$

Corollary A.11 (Expectation-level ANOVA consistency and centering). Let $\mu ( \lambda ) = \operatorname { \mathbb { E } } [ S _ { \lambda } ]$ and let $\{ \mu _ { U } \} _ { U \subseteq { \mathcal C } }$ be the unique Hoeffding/ANOVA components from Theorem A.5. For any $U \subseteq { \mathcal { C } }$ with $| U | \geq 2$ and any $\lambda _ { U }$

$$
\mathbb { E } [ E _ { U } ( \lambda _ { U } ) ] = \mu _ { U } ( \lambda _ { U } ) .\tag{74}
$$

In particular, the interaction effects are centered in expectation. For any $c \in U$

$$
\sum _ { \lambda _ { c } = 1 } ^ { \Lambda _ { c } } \mathbb { E } [ E _ { U } ( \lambda _ { U } ) ] = 0 .\tag{75}
$$

Proof. Fix $\lambda _ { - U }$ . By construction, the comonotone coupling in Eq. (68) preserves the marginal distribution of each term $M _ { V | \lambda _ { - U } } ( \lambda _ { V } )$ , hence it does not alter expectations. Therefore, by linearity of expectation,

$$
\mathbb { E } \big [ H _ { U | \lambda _ { - U } } ( \lambda _ { U } ) \big ] = \sum _ { V \subseteq U } ( - 1 ) ^ { | U | - | V | } \mathbb { E } \big [ M _ { V | \lambda _ { - U } } ( \lambda _ { V } ) \big ] .\tag{76}
$$

Next, expand the blocked slice definition (Eq. (61)) and apply linearity of expectation to obtain

$$
\begin{array} { r l } { } & { \mathbb { E } \big [ M _ { V | \mathsf { \pmb { \lambda } } _ { - U } } ( \pmb { \lambda } _ { V } ) \big ] } \\ { } & { = \frac { 1 } { | \mathscr { D } _ { U \backslash V } | } \displaystyle \sum _ { \pmb { \lambda } _ { U \backslash V } \in \mathscr { D } _ { U \backslash V } } \mu ( \pmb { \lambda } _ { V } , \pmb { \lambda } _ { U \backslash V } , \pmb { \lambda } _ { - U } ) . } \end{array}\tag{77}
$$

(78)

Now average Eq. (76) uniformly over $\lambda _ { - U }$ (Eq. (69)). Because the design is a finite product space, averaging over $\lambda _ { - U }$ and over $\lambda _ { U \setminus V }$ is equivalent to averaging over all complementary coordinates $\lambda _ { - V }$ . Thus

$$
\mathbb { E } [ E _ { U } ( \lambda _ { U } ) ]
$$

$$
= \frac { 1 } { | \mathcal { D } _ { - U } | } \sum _ { \lambda _ { - U } } \mathbb { E } \big [ H _ { U | \lambda _ { - U } } ( \lambda _ { U } ) \big ]\tag{79}
$$

$$
= \sum _ { V \subseteq U } ( - 1 ) ^ { | U | - | V | } { \frac { 1 } { | { \mathcal { D } } _ { - V } | } } \sum _ { \lambda _ { - V } } \mu ( \lambda _ { V } , \lambda _ { - V } ) .\tag{80}
$$

$$
\mathop { = } ( T _ { V } \mu ) ( \lambda _ { V } )\tag{81}
$$

The final expression is exactly the Möbius inversion / Hoeffding component $\mu _ { U } ( \lambda _ { U } )$ (Rota, 1964), equivalent to the recursion in Theorem A.5. This proves Eq. (74). The centering property Eq. (75) then follows from the identifiability constraints in Theorem A.5. □

## A.4.7 Hypothesis Testing

We distinguish our hypothesis evaluation from nullhypothesis significance testing (NHST). Rather than accept–reject decisions based on p-values, which presuppose sampling variability from finite data, we evaluate directional scientific claims using exact predictive effect distributions computed analytically from the model’s next-token probabilities.

In our framework, the primitive objects of inference are centered effect random variables, namely paired main effects $E _ { c } ( \lambda _ { c } )$ (Eq. (42)) and paired interaction effects $E _ { U } ( \lambda _ { U } )$ (Eq. (69)). Their PMFs are computed deterministically via finite mixtures and paired comonotone contrast operations. Accordingly, there is no sampling error. Uncertainty in these effect distributions reflects intrinsic stochasticity of the model’s predictive behavior under the stimulus (i.e., the aleatoric uncertainty).

We summarize each centered effect distribution E using mean shift (E[E]), predictive dispersion $( \mathrm { S D } ( E ) )$ , discrete Probability of Direction $( \mathrm { d P D } ( E ) )$ , and a standardized signal-to-noise ratio $( \mathrm { S N R } ( E ) )$ based on the SD of the paired effect distribution.

## A.4.8 Discrete Probability of Direction

To evaluate directional hypotheses, we adopt a discrete formulation of the Probability of Direction (PD). In Bayesian inference, PD is typically defined for continuous posteriors, where $\mathbb { P } ( E = 0 ) = 0$ (Makowski et al., 2019). Here, predictive effect distributions are discrete and may place non-zero mass on exact equality. We therefore treat $\{ E = 0 \}$ as a distinct behavioral outcome.

Let E denote a centered effect random variable, either $E _ { c } ( \lambda _ { c } )$ or $E _ { U } ( \lambda _ { U } )$ . The discrete Probability of Direction (dPD) is

$$
\mathrm { d } \mathrm { P D } ( E ) = \operatorname* { m a x } \bigl ( \mathbb { P } ( E > 0 ) , \mathbb { P } ( E < 0 ) \bigr ) ,\tag{82}
$$

dPD quantifies the extent to which the effect exhibits consistent directionality, irrespective of sign.

## A.4.9 Effect Size (Mean Shift)

We define effect size as the expectation of the centered effect distribution

$$
\mathbb { E } [ E ] = \sum _ { z } z P _ { E } ( z ) ,\tag{83}
$$

where the sum ranges over the finite support of $E$ E[E] is the magnitude of the effect E.

## A.4.10 Predictive Dispersion (SD)

Intrinsic variability of the centered effect is captured by its predictive standard deviation

$$
\mathrm { S D } ( E ) = { \sqrt { \sum _ { z } \left( z - \mathbb { E } [ E ] \right) ^ { 2 } P _ { E } ( z ) } } .\tag{84}
$$

Because our effect distributions are constructed via paired contrasts, SD(E) reflects residual predictive heterogeneity in the effect itself, rather than dispersion induced by taking differences of independent draws.

## A.4.11 Signal-to-Noise Ratio

We summarize standardized effect magnitude using a signal-to-noise ratio computed directly on the paired effect distribution

$$
\mathrm { S N R } ( E ) = { \frac { | \mathbb { E } [ E ] | } { \mathrm { S D } ( E ) } } .\tag{85}
$$

When $\mathrm { S D } ( E ) = 0 $ , the effect is deterministic under the paired contrast. In this case $\operatorname { S N R } ( E )$ is treated as $+ \infty \mathrm { i f } \mathbb { E } [ E ] \neq 0$ and $0 { \mathrm { i f } } \mathbb { E } [ E ] = 0$

The signal-to-noise ratio SNR is closely related to standardized effect sizes such as Cohen’s d (Cohen, 2013), which quantifies mean deviations relative to within-group variability. However, in our setting, variability reflects intrinsic stochasticity of the language model rather than sampling error. We therefore use conventional heuristics (e.g., $\approx 0 . 2 , 0 . 5 , 0 . 8$ for small, medium, large) only as descriptive reference points, not as decision thresholds.

## A.4.12 Interpretation

Inferential conclusions are drawn from the joint configuration of dPD(E), E[E], SD(E), and $\operatorname { S N R } ( E )$ . High dPD(E) indicates strong directional regularity, while large SNR(E) indicates that the expected shift dominates the intrinsic predictive dispersion of the paired effect distribution. Discrepancies between these quantities distinguish regimes such as stable but weak effects, large but ambivalent effects, or low-signal effects with high dispersion (Table 4).

## A.4.13 Trend Effects

Some experimental factors admit an inherent ordering and a meaningful numeric scale (e.g., model size in billions of parameters or a discretized continuous manipulation intensity). To quantify the marginal per-unit sensitivity of the composite construct to such a factor without imposing a parametric functional form, we compute paired finite difference distributions across adjacent factor levels and aggregate them into an average trend distribution.

Let $c \in { \mathcal { C } }$ denote an ordered factor with $L = \Lambda _ { c }$ levels that admit numeric values

$$
\lambda _ { 1 } ^ { ( c ) } < \lambda _ { 2 } ^ { ( c ) } < \cdots < \lambda _ { L } ^ { ( c ) } \qquad ( \lambda _ { \ell } ^ { ( c ) } \in \mathbb { R } ) .
$$

Adjacent spacings may be uneven. Define increments

$$
\delta _ { \ell } ^ { ( c ) } = \lambda _ { \ell + 1 } ^ { ( c ) } - \lambda _ { \ell } ^ { ( c ) } > 0 , \ell = 1 , \ldots , L - 1 .
$$

<table><tr><td>Case</td><td> $P ( E )$ </td><td> $\mathbb { E } [ E ]$ </td><td>SD</td><td>SNR</td><td>dPD</td></tr><tr><td>A</td><td> $\{ - 2 5 { : } 0 . 9 , - 2 0 { : } 0 . 1 \}$ </td><td>-24.5</td><td>1.5</td><td>16.3</td><td>1.00</td></tr><tr><td>B</td><td> $\{ 1 { : } 0 . 9 9 , 1 0 0 { : } 0 . 0 1 \}$ </td><td>11.0</td><td>99.0</td><td>0.11</td><td>1.00</td></tr><tr><td>C</td><td> $\{ 1 0 0 { : } 0 . 5 1 , - 1 { : } 0 . 4 9 \}$ </td><td>50.5</td><td>49.5</td><td>1.02</td><td>0.51</td></tr><tr><td>D</td><td> $\{ 5 { : 0 . 3 4 } , - 5 { : 0 . 3 3 } , 0 { : 0 . 3 3 } \}$ </td><td>0.05</td><td>4.1</td><td>0.01</td><td>0.34</td></tr></table>

Table 4: Illustrative predictive effect distributions exhibiting distinct behavioral regimes. Cases A and B have identical directional consistency (dPD = 1) but sharply different standardized magnitudes (SNR). Case C shows a large expected effect despite weak directional consistency. Case D exhibits neither directional regularity nor an appreciable standardized signal.

To respect the factorial design, we compare adjacent levels within each block of the remaining factors $\lambda _ { - c } \in \mathcal { D } _ { - c } .$ . For each block and each adjacent pair of ordered levels, define the paired local difference random variable

$$
D _ { \ell | \lambda _ { - c } } ^ { ( c ) } : = S _ { ( \lambda _ { \ell + 1 } ^ { ( c ) } , \lambda _ { - c } ) } \ominus S _ { ( \lambda _ { \ell } ^ { ( c ) } , \lambda _ { - c } ) } ,\tag{86}
$$

$$
\ell = 1 , \ldots , L - 1 ,\tag{87}
$$

where denotes the paired comonotone difference from Def. A.8. Each PMF $P _ { D _ { \ell | \lambda _ { - } \epsilon } ^ { ( c ) } }$ is computed exactly via the discrete mass-matching formula in Eq. (40).

We then marginalize over blocks by a uniform mixture

$$
P _ { D _ { \ell } ^ { ( c ) } } = \frac { 1 } { | \mathscr { D } _ { - c } | } \sum _ { \lambda _ { - c } \in \mathscr { D } _ { - c } } P _ { D _ { \ell | \lambda _ { - c } } ^ { ( c ) } } .\tag{88}
$$

Intuitively, $D _ { \ell } ^ { ( c ) }$ is the predictive distribution of the change in the composite score induced by moving from $\lambda _ { \ell } ^ { ( c ) }$ to $\lambda _ { \ell + 1 } ^ { ( c ) } ,$ averaging uniformly over all other experimental factors while preserving a paired (within-block) comparison.

Define the per-unit local slope random variable as the rescaled difference

$$
\Delta _ { \ell } ^ { ( c ) } = \frac { D _ { \ell } ^ { ( c ) } } { \delta _ { \ell } ^ { ( c ) } } .\tag{89}
$$

Equivalently, $P _ { \Delta _ { \rho } ^ { ( c ) } }$ is the pushforward of $P _ { D _ { \rho } ^ { ( c ) } }$ under the ma $\mathfrak { p } z \mathbin { \mapsto } z / \delta _ { \ell } ^ { ( c ) }$ ; its support is rescaled by $1 / \delta _ { \ell } ^ { ( c ) }$ while probability masses are preserved.

To aggregate unevenly spaced local slopes into a single per-unit trend effect, we define the per-unit trend random variable $\gamma _ { c }$ as the length-weighted

mixture of scaled local slopes:

$$
P _ { \gamma _ { c } } = \sum _ { \ell = 1 } ^ { L - 1 } w _ { \ell } ^ { ( c ) } P _ { \Delta _ { \ell } ^ { ( c ) } } ,\tag{90}
$$

$$
w _ { \ell } ^ { ( c ) } = \frac { \delta _ { \ell } ^ { ( c ) } } { \sum _ { m = 1 } ^ { L - 1 } \delta _ { m } ^ { ( c ) } } = \frac { \delta _ { \ell } ^ { ( c ) } } { \lambda _ { L } ^ { ( c ) } - \lambda _ { 1 } ^ { ( c ) } }\tag{91}
$$

Dispersion or multimodality in $P _ { \gamma _ { c } }$ indicates heterogeneous local behavior (e.g., non-monotonic or regime-dependent trends) across adjacent factorlevel transitions.

Proposition A.12 (Expectation-level secant-slope identity). The expected per-unit trend equals the secant slope of the ordered-factor marginal mean across the observed range

$$
\mathbb { E } [ \gamma _ { c } ] = \frac { \mathbb { E } [ M _ { c } ( \lambda _ { L } ^ { ( c ) } ) ] - \mathbb { E } [ M _ { c } ( \lambda _ { 1 } ^ { ( c ) } ) ] } { \lambda _ { L } ^ { ( c ) } - \lambda _ { 1 } ^ { ( c ) } } .\tag{92}
$$

Proof. Fix $\ell \in \{ 1 , \ldots , L - 1 \}$ and a block $\lambda _ { - c } .$ By Def. A.8, the paired difference preserves the mean

$$
\begin{array} { r } { \mathbb { E } \Big [ D _ { \ell | \lambda _ { - c } } ^ { ( c ) } \Big ] = \mathbb { E } \Big [ S _ { ( \lambda _ { \ell + 1 } ^ { ( c ) } , \lambda _ { - c } ) } \Big ] - \mathbb { E } \Big [ S _ { ( \lambda _ { \ell } ^ { ( c ) } , \lambda _ { - c } ) } \Big ] . } \end{array}
$$

Averaging uniformly over $\lambda _ { - c }$ and using linearity of expectation gives

$$
\mathbb { E } [ D _ { \ell } ^ { ( c ) } ] = \mathbb { E } [ M _ { c } ( \lambda _ { \ell + 1 } ^ { ( c ) } ) ] - \mathbb { E } [ M _ { c } ( \lambda _ { \ell } ^ { ( c ) } ) ] ,
$$

because $M _ { c } ( \lambda _ { \ell } ^ { ( c ) } )$ is the uniform mixture of $S _ { ( \lambda _ { \ell } ^ { ( c ) } , \lambda _ { - c } ) }$ over λ<sub>−c</sub> (Eq. (20)), and expectations commute with finite mixtures (Theorem A.2).

By definition of $\Delta _ { \ell } ^ { ( c ) }$

$$
\mathbb { E } [ \Delta _ { \ell } ^ { ( c ) } ] = \frac { \mathbb { E } [ D _ { \ell } ^ { ( c ) } ] } { \delta _ { \ell } ^ { ( c ) } } .
$$

Taking expectations in the mixture definition (90) yields

$$
\mathbb { E } [ \gamma _ { c } ] = \sum _ { \ell = 1 } ^ { L - 1 } w _ { \ell } ^ { ( c ) } \mathbb { E } [ \Delta _ { \ell } ^ { ( c ) } ]\tag{93}
$$

$$
\frac { 1 } { \sum _ { m = 1 } ^ { L - 1 } \delta _ { m } ^ { ( c ) } } \sum _ { \ell = 1 } ^ { L - 1 }\tag{94}
$$

$$
\left( \mathbb { E } [ M _ { c } ( \lambda _ { \ell + 1 } ^ { ( c ) } ) ] - \mathbb { E } [ M _ { c } ( \lambda _ { \ell } ^ { ( c ) } ) ] \right) ,\tag{95}
$$

which telescopes $\begin{array} { r l r } { \mathrm { ~ t o ~ } } & { { } \ } & { ( \mathbb { E } [ M _ { c } ( \lambda _ { L } ^ { ( c ) } ) ] \mathrm { ~ \ ~ \ } - } \end{array}$ $\mathbb { E } [ M _ { c } ( \lambda _ { 1 } ^ { ( c ) } ) ] ) / ( \lambda _ { L } ^ { ( c ) } - \lambda _ { 1 } ^ { ( c ) } )$ , proving Eq. (92).

Directional interpretation. The full predictive distribution $P _ { \gamma _ { c } }$ supports directional summaries (e.g., $\mathbb { P } ( \gamma _ { c } > 0 ) , \mathbb { P } ( \gamma _ { c } < 0 ) )$ and can be summarized using the same quantities defined for other predictive contrasts (e.g., dPD, SD, and SNR based on $\mathrm { S D } ( \gamma _ { c } ) ,$ ).

## A.5 Prompt template

Figure 7 shows how we form the prompt template and cross Target Country with each model.

## A.6 Additional Analysis

Effect of Model Size. While our primary factorial design treats language models as discrete categorical subjects, model capacity is inherently continuous. Prior work has shown that scaling language models can systematically alter social and behavioral tendencies, including the amplification of biased responses (Wadi and Fredette, 2025). We therefore investigate whether ethnocentric tendencies, as measured by the CETSCALE, exhibit a systematic trend as a function of model size. Specifically, we ask whether increasing the number of parameters induces a consistent directional change in the latent ethnocentrism score. To address this question, we conduct a trend-effects analysis (derived in App. A.4.13).

We form four within-family size series, each evaluated under the same TARGET manipulation and CETSCALE instrument as in the main experiment. AYA (8B, 32B), GEMMA 3 (1B, 4B, 12B, 27B), MINISTRAL (3B, 8B, 14B), and QWEN3 (0.6B, 4B, 8B, 14B, 32B, Next-80B). <sup>5</sup> The trend effect of model size on ethnocentrism (Table 8) is large $( \mathbb { E } = 1 . 6 5 , \mathrm { S N R } = 5 . 8 7 )$ and directionally consistent $\mathrm { ( d P D > 0 . 9 9 ) }$ for AYA model family. This indicates steep linear scaling in ethnocentric tendencies. For every additional billion parameters, the model’s expected ethnocentrism score increases by 1.65 points. Over the range evaluated (8B to 32B), this effectively shifts a model from the lowest observed ethnocentrism to the highest, purely by scaling. For other models, size does not have a meaningful effect (low SNR and dPD) on ethnocentrism.

## A.7 Additional tables/figures

• Table 5 shows a comparison of Multivariate Consensus versus Entropy for the country-oforigin experiment.

• Figure 8 shows a comparison of the human baseline with the LLMs tested on the CETSCALE.

• Table 6 shows the full ANOVA decomposition (main and interaction effects) of the composite CETSCALE score.

• Figure 9 shows the main effect of the Target Country on the CETSCALE.

• Table 7 shows the pairwise marginal contrasts of the Target Countries.

• Figure 10 shows the exact PMF of the LLMs on the composite CETSCALE.

• Table 8 shows the effect of LLM size (number of parameters) on the CETSCALE.

• Table 9 shows a comparison between the estimated effect using aggregation (baseline) compared with factorial decomposition (our proposed method).

## A.8 Cost of Monte Carlo sampling

To quantify the cost of the standard Monte Carlo sampling paradigm, we treat the exact PMFs as ground truth and characterize the two distinct costs a sampling-based estimator of the composite CETSCALE score would incur: (i) stochastic sampling noise from finite N, and (ii) systematic decoding bias from non-default temperature or topp. All quantities are reported in composite-score units (the 17–119 CETSCALE scale), directly comparable to the effect sizes in Table 2.

![](images/9e85bce04157d578023ba557b4e74f3c371a887ae35d3a79396f9cd5d31310e0.jpg)  
Figure 7: Prompt setup: a fixed system prompt enforces the response schema, while Target Country (sample shown for Canada) deterministically instantiates item templates via a lexicon. Example shown is $T _ { 1 }$ from the CETSCALE; each CETSCALE item has its own template.

<table><tr><td>Model</td><td>Country</td><td>Cns</td><td>Entropy</td></tr><tr><td rowspan="5">Aya 32B</td><td>Canada</td><td>0.941</td><td>4.945</td></tr><tr><td>China</td><td>0.939</td><td>5.150</td></tr><tr><td>France</td><td>0.937</td><td>5.010</td></tr><tr><td>USA</td><td>0.924</td><td>7.315</td></tr><tr><td>Canada</td><td>0.959</td><td>3.565</td></tr><tr><td rowspan="4"></td><td>China</td><td>0.941</td><td>4.886</td></tr><tr><td>France</td><td>0.940</td><td>4.944</td></tr><tr><td>USA</td><td>0.948</td><td>4.533</td></tr><tr><td>Canada</td><td>0.924</td><td>5.620</td></tr><tr><td rowspan="4"></td><td>China</td><td>0.894</td><td>8.579</td></tr><tr><td>France</td><td>0.882</td><td>5.800</td></tr><tr><td>USA</td><td>0.947</td><td>3.929</td></tr><tr><td>Canada</td><td>0.669</td><td>35.519</td></tr><tr><td rowspan="4">Ministral 14B</td><td>China</td><td>0.646</td><td>37.521</td></tr><tr><td>France</td><td></td><td>36.444</td></tr><tr><td></td><td>0.668</td><td></td></tr><tr><td>USA</td><td>0.660</td><td>35.759</td></tr><tr><td rowspan="4">Qwen3 Next 80B</td><td>Canada</td><td>0.929</td><td>6.861</td></tr><tr><td>China</td><td>0.886</td><td>9.008</td></tr><tr><td>France</td><td>0.886</td><td>8.381</td></tr><tr><td>USA</td><td>0.935</td><td>5.950</td></tr></table>

Table 5: Consensus and Entropy of Models and Target Countries

Because CETSCALE items are conditionally independent (Sec. A.2), a single sampled composite response has variance $\begin{array} { r } { \sigma ^ { 2 } = \sum _ { k = 1 } ^ { K ^ { - } } \operatorname { V a r } ( Y _ { k } ) } \end{array}$ , and the sample-mean estimator of the composite score has standard error $\mathrm { S E } = \sigma / \sqrt { N }$ . Table 10 reports this SE across per-condition budgets N, alongside each model’s multivariate Consensus (Cns). We verified these analytical values against a Monte Carlo simulation (500 replications per cell), which matched to within-simulation error.

## A.8.1 Sampling Noise

Sampling error is non-trivial at realistic budgets. At N=100, the SE ranges from 0.10 (Gemma) to 0.52 (Ministral) composite-score points. The latter is a meaningful fraction of several effects in Table 2. The error decays only as $1 / \sqrt { N }$ , and hence each tenfold increase in N reduces SE by only $\approx 3 . 1 6 \times$ and nonzero error persists even at $N { = } 1 0 ^ { 6 }$ . Moreover, the magnitude of this noise is anti-correlated with Consensus $( r = - 1 . 0 0 , p < 0 . 0 0 1$ , across the 20 MODEL TARGET conditions). Low-consensus item PMFs demand far more samples for the same precision, so Ministral (Cns=0.66) requires  5 the samples of Gemma (Cns=0.95) at every N (Figure 11).

![](images/1246933ab27c7d3769e4c7779f7c65295dfd7735188f5d3a1948eaa3325a298c.jpg)  
Figure 8: Comparison of CETSCALE scores for the USA target condition between human populations reported in Shimp and Sharma (1987) and LLMs evaluated in this study. Points denote expected values, and error bars indicate standard deviation. All scores are based on the 17-item CETSCALE using a 7-point Likert scale. Notably, several LLMs exhibit ethnocentrism levels comparable to or exceeding those observed in highly ethnocentric human samples, while displaying substantially lower intrinsic dispersion.

## A.8.2 Decoding Bias

Whereas sampling noise vanishes as $N \to \infty ,$ nondefault decoding introduces a systematic bias that no amount of sampling can correct, because temperature and top-p reshape the target distribution itself. We quantify this deterministically. For each condition we recompute the exact expected composite score under a grid of temperatures $T \in$ 0.1, 0.5, 1.0, 1.3 and $\mathrm { t o p } \mathrm { - } p \in \{ 0 . 8 , 0 . 9 , 1 . 0 \}$ and measure the bias relative to the model’s native $( T { = } 1 . 0 , \mathrm { t o p } { - } p { = } 1 . 0 )$ distribution (Table 11).

Table 6: Main effect and interaction table for the factorial model.
<table><tr><td>PARAMETER</td><td>E</td><td>SD</td><td>SNR</td><td>dPD</td></tr><tr><td>µ∅</td><td>66.25</td><td>14.21</td><td></td><td></td></tr><tr><td>AYA-32B</td><td>21.19</td><td>13.01</td><td>1.63</td><td>0.99</td></tr><tr><td>GEMMA 3-27B</td><td>-11.99</td><td>13.15</td><td>0.91</td><td>0.77</td></tr><tr><td>LLAMA 3.3-70B</td><td>0.31</td><td>12.79</td><td>0.02</td><td>0.55</td></tr><tr><td>MINISTRAL-14B</td><td>5.11</td><td>9.59</td><td>0.53</td><td>0.62</td></tr><tr><td>QWEN3 NEXT-80B</td><td>-14.63</td><td>12.17</td><td>1.20</td><td>0.93</td></tr><tr><td>USA</td><td>2.84</td><td>5.43</td><td>0.52</td><td>0.55</td></tr><tr><td>CANADA</td><td>4.62</td><td>5.72</td><td>0.81</td><td>0.90</td></tr><tr><td>CHINA</td><td>-6.46</td><td>6.19</td><td>1.04</td><td>0.95</td></tr><tr><td>FRANCE</td><td>-1.00</td><td>4.72</td><td>0.21</td><td>0.54</td></tr><tr><td> $\mathrm { A Y A } { - } 3 2 \mathrm { B } \times \mathrm { U S A }$ </td><td>-1.18</td><td>3.03</td><td>0.39</td><td>0.55</td></tr><tr><td> $\mathrm { A Y A } { - } 3 2 \mathrm { B } \times \mathrm { C A N A D A }$ </td><td>-3.77</td><td>2.88</td><td>1.31</td><td>0.94</td></tr><tr><td> $\mathrm { A Y A } { - } 3 2 \mathrm { B } \times \mathrm { C H I N A }$ </td><td>5.35</td><td>4.53</td><td>1.18</td><td>0.77</td></tr><tr><td> $\mathbf { A Y A - } 3 2 \mathbf { B } \times \mathbf { F R A N C E }$ </td><td>-0.39</td><td>2.89</td><td>0.14</td><td>0.42</td></tr><tr><td>GEMMA  $3 { \cdot } 2 7 \mathrm { B } \times \mathrm { U } \mathrm { S } \mathrm { A }$ </td><td>3.21</td><td>5.34</td><td>0.60</td><td>0.63</td></tr><tr><td>GEMMA  $3 – 2 7 \mathbf { B } \times \mathbf { C } _ { \mathrm { A N A D A } }$ </td><td>2.14</td><td>5.63</td><td>0.38</td><td>0.47</td></tr><tr><td>GEMMA  $3 – 2 7 \mathrm { B } \times \mathrm { C H I N A }$ </td><td>-5.40</td><td>8.63</td><td>0.63</td><td>0.61</td></tr><tr><td>GEMMA  $3 { \cdot } 2 7 \mathbf { B } \times \mathrm { F R A N C E }$ </td><td>0.05</td><td>5.34</td><td>0.01</td><td>0.51</td></tr><tr><td> $\mathrm { L L A M A } 3 . 3 \mathrm { - } 7 0 \mathrm { B } \times \mathrm { U S A }$ </td><td>2.59</td><td>5.86</td><td>0.44</td><td>0.58</td></tr><tr><td> $\operatorname { L L A M A } 3 . 3 \mathbf { - } 7 0 \mathbf { B } \times \mathbf { C } \mathbf { A N A D A }$ </td><td>5.20</td><td>5.49</td><td>0.95</td><td>0.77</td></tr><tr><td> $\operatorname { L L A M A } 3 . 3 – 7 0 \mathbf { B } \times \mathbf { C } \mathbf { H I N A }$ </td><td>-4.29</td><td>9.12</td><td>0.47</td><td>0.58</td></tr><tr><td> $\mathrm { L L A M A } ~ 3 . 3 \ – 7 0 \mathbf { B } \times \mathrm { F R A N C E }$ </td><td>-3.50</td><td>5.55</td><td>0.63</td><td>0.67</td></tr><tr><td> $\mathbf { M I N I S T R A L - } 1 4 \mathbf { B } \times \mathbf { U S A }$ </td><td>-4.00</td><td>3.10</td><td>1.29</td><td>0.88</td></tr><tr><td> $\mathrm { M I N I S T R A L - 1 4 B \times C A N A D A }$ </td><td>-2.99</td><td>3.16</td><td>0.95</td><td>0.73</td></tr><tr><td> $\mathrm { M I N I S T R A L } - 1 4 \mathrm { B } \times \mathrm { C H I N A }$ </td><td>4.54</td><td>4.11</td><td>1.11</td><td>0.72</td></tr><tr><td> $\mathbf { M I N I S T R A L - 1 4 B \times F R A N C E }$ </td><td>2.45</td><td>2.98</td><td>0.82</td><td>0.87</td></tr><tr><td> $\mathrm { Q w e n 3 ~ N E X T  – 8 0 B \times U S A }$ </td><td>-0.61</td><td>2.83</td><td>0.22</td><td>0.58</td></tr><tr><td>QWEN3  $\mathbf { N E X T - 8 0 B } \times \mathbf { C A N A D A }$ </td><td>-0.57</td><td>2.87</td><td>0.20</td><td>0.46</td></tr><tr><td> $\mathrm { Q w e N 3 ~ N E X T - 8 0 B \times C H I N A }$ </td><td>-0.20</td><td>5.11</td><td>0.04</td><td>0.46</td></tr><tr><td>QWEN3  $\mathrm { N E X T - 8 0 B \times F R A N C E }$ </td><td>1.39</td><td>2.81</td><td>0.49</td><td>0.53</td></tr></table>

![](images/8f8772de0782ac11254368a149c0f5ba2fdf1930edc815cb2595b320477d6de4.jpg)  
Figure 9: Main Effect of Target Country

The distortion is small for high-consensus models (Gemma: 0.18 points) but substantial for lowconsensus Ministral $( M _ { | B i a s | } = 1 . 3 8 .$ , reaching up to +2.79 points at low temperature and reversing sign at $T { = } 1 . 3 ;$ Figures 12–13), comparable to the effects in Table 2. As with sampling noise, the bias is strongly anti-correlated with Consensus $( r ~ = ~ - 0 . 8 5 , ~ p ~ < ~ 0 . 0 0 1 , ~ n { = } 2 0 $ conditions). Intuitively, the results confirm that a sharp, high-consensus PMF has almost no dispersed mass for temperature or nucleus truncation to reshape, whereas a flat, low-consensus PMF has its mean substantially shifted.

Table 7: Pairwise marginal contrasts between Target Countries.
<table><tr><td>CONTRAST</td><td>E</td><td>SD</td><td>SNR</td><td>dPD</td></tr><tr><td>USA - CANADA</td><td>-1.78</td><td>1.83</td><td>0.97</td><td>0.74</td></tr><tr><td>USA - CHINA</td><td>9.30</td><td>6.93</td><td>1.34</td><td>0.95</td></tr><tr><td>USA - FRANCE</td><td>3.84</td><td>4.40</td><td>0.87</td><td>0.80</td></tr><tr><td>CANADA - CHINA</td><td>11.08</td><td>7.60</td><td>1.46</td><td>1.00</td></tr><tr><td> $\mathrm { C A N A D A \mathrm { ~ - ~ } F R A N C E }$ </td><td>5.62</td><td>5.05</td><td>1.11</td><td>0.84</td></tr><tr><td> $\mathrm { C H I N A \cdot F R A N C E }$ </td><td>-5.47</td><td>3.79</td><td>1.44</td><td>0.80</td></tr></table>

## A.9 Reliability of sampling-based estimators under ideal decoding

The exact-PMF framework eliminates sampling error by construction. To quantify what a researcher stands to lose by instead adopting the standard

![](images/f45d8a02516712930f0932edf540ea020c18f1ac664287c719968c631fcc1d09.jpg)  
(a) Aya Expanse 32B

![](images/a597b196b1c2df0f6d6d02b6c5bb4b2246414914e83b7cb4dd93bc6a67a4c93b.jpg)  
(b) Gemma 3 27B

![](images/e3b3e58a3cc9483a4b816adc4a18f9cb65bc33ff4a0256cfc1cae52017a98ce9.jpg)  
(c) Llama 3.3 70B

![](images/2b098d27473349e9b08cee7b8586f82df26ddd9e03b221b854014be017fc67a0.jpg)  
(d) Ministral 14B

![](images/a9704ad1f26c540a2960d526858e399f90d346d4fd8307c5d2c9bbf81988f0c7.jpg)  
(e) Qwen3 Next 80B  
Figure 10: Distribution of CETSCALE for Model and Target Country.

![](images/6914d2ca476e1aaaa966c10af9cd05bd388071f6f58bf3e12ae605c20e273379.jpg)

Table 8: Trend effects for model size (billions of parameters).
<table><tr><td>MODEL FAMILY</td><td>E</td><td>SD</td><td>SNR</td><td>dPD</td></tr><tr><td>AYA</td><td>1.65</td><td>0.28</td><td>5.87</td><td>1.00</td></tr><tr><td>GEMMA</td><td>1.12</td><td>7.21</td><td>0.16</td><td>0.70</td></tr><tr><td>MINISTRAL</td><td>0.87</td><td>1.12</td><td>0.78</td><td>0.56</td></tr><tr><td>QWEN3</td><td>-0.00</td><td>1.79</td><td>0.00</td><td>0.89</td></tr></table>

Figure 11: Standard error of the sampling-based CETSCALE estimator versus per-condition sample size N, by model. Low-consensus models (e.g., Ministral) incur substantially larger error at every budget.  
![](images/ad7b764f8a0b8e52222c17fef8068a38a733af6d698e1262cc355d7cc40aad27.jpg)  
Figure 12: Mean decoding bias versus temperature, colored by Consensus. Low-consensus models exhibit large, sign-changing distortions.

![](images/b77ac9ea1082589f936e03fbdb9a92f3813cd8375bee2d3356c3607bf9800230.jpg)  
Figure 13: Mean decoding bias versus top-p, colored by Consensus.

Monte Carlo sampling paradigm, we ask a decisionrelevant question: what fraction of the causal effects we report could a sampling-based study get wrong (i.e., recover with the opposite sign) purely as an artifact offinite sampling?

Setup. We treat our exact effect distributions as ground truth. For each factorial parameter (i.e., the MODEL and TARGET main effects and every MODEL  TARGET interaction) we compute its exact effect size $E _ { \mathrm { e x a c t } }$ using the sum-to-zero contrasts consistent with Theorem 3.1. We then model a researcher who estimates the same contrast from N sampled CETSCALE responses per condition. Because items are conditionally independent (Sec. A.2), a single composite draw has variance $\Sigma _ { k = 1 } ^ { K }$ Var(Y<sub>k</sub>), so the contrast estimator is asymptotically

Table 9: Aggregate versus factorial decomposition of Target-Country effects on CETSCALE ethnocentrism, computed with identical OT-coupled estimators. The aggregate benchmark (collapsing over MODEL) reports a single Target-Country main effect that is, by construction, uniform across models. The factorial MODEL  TARGET interaction isolates the country-of-origin bias that aggregation cannot represent. Bold interactions indicate a sign reversal relative to the aggregate main effect (i.e., conditions for which the aggregate view is directionally incorrect).
<table><tr><td rowspan="2">TARGET</td><td rowspan="2">MODEL (ORIGIN)</td><td colspan="4">ISOLATED EFFECT</td></tr><tr><td>E</td><td>SD</td><td>SNR</td><td>DPD</td></tr><tr><td colspan="7">Panel A. Aggregate benchmark (Target main effect; collapses over Model)</td></tr><tr><td>USA</td><td>— (ALL MODELS)</td><td>2.84</td><td>5.43</td><td>0.52</td><td>0.55</td></tr><tr><td>CANADA</td><td>— (ALL MODELS)</td><td>4.62</td><td>5.72</td><td>0.81</td><td>0.90</td></tr><tr><td>CHINA</td><td>— (ALL MODELS)</td><td>-6.46</td><td>6.19</td><td>1.04</td><td>0.95</td></tr><tr><td>FRANCE</td><td>— (ALL MODELS)</td><td>-1.00</td><td>4.72</td><td>0.21</td><td>0.54</td></tr><tr><td colspan="6">Panel B. Factorial interaction (Model × Target; isolated country-of-origin bias)</td></tr><tr><td rowspan="5">USA</td><td>LLAMA 3.3 70B (US)†</td><td>2.59</td><td>5.86</td><td>0.44</td><td>0.58</td></tr><tr><td>GEMMA 3 27B (US)†</td><td>3.21</td><td>5.34</td><td>0.60</td><td>0.63</td></tr><tr><td>AYA EXPANSE 32B (CA)</td><td>-1.18</td><td>3.03</td><td>0.39</td><td>0.55</td></tr><tr><td>MINISTRAL 14B (FR)</td><td>-4.00</td><td>3.10</td><td>1.29</td><td>0.88</td></tr><tr><td>QWEN3 NEXT 80B (CN)</td><td>-0.61</td><td>2.83</td><td>0.22</td><td>0.58</td></tr><tr><td rowspan="6">CANADA</td><td>LLAMA 3.3 70B (US)</td><td>5.20</td><td>5.49</td><td>0.95</td><td>0.77</td></tr><tr><td>GEMMA 3 27B (US)</td><td>2.14</td><td>5.63</td><td>0.38</td><td>0.47</td></tr><tr><td>AYA EXPANSE 32B (CA)†</td><td>-3.77</td><td>2.88</td><td>1.31</td><td>0.94</td></tr><tr><td>MINISTRAL 14B (FR)</td><td>-2.99</td><td>3.16</td><td>0.95</td><td>0.73</td></tr><tr><td>QWEN3 NEXT 80B (CN)</td><td>-0.57</td><td>2.87</td><td>0.20</td><td>0.46</td></tr><tr><td>LLAMA 3.3 70B (US)</td><td>-4.29</td><td>9.12</td><td>0.47</td><td>0.58</td></tr><tr><td rowspan="5">CHINA</td><td>GEMMA 3 27B (US)</td><td>-5.40</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>8.63 4.53</td><td>0.63 1.18</td><td>0.61 0.77</td></tr><tr><td>AYA EXPANSE 32B (CA) MINISTRAL 14B (FR)</td><td>5.35 4.54</td><td>4.11</td><td>1.11</td><td>0.72</td></tr><tr><td>QWEN3 NEXT 80B (CN)†</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>-0.20</td><td>5.11</td><td>0.04</td><td>0.46</td></tr><tr><td rowspan="5">FRANCE</td><td>LLAMA 3.3 70B (US)</td><td>-3.50</td><td>5.55</td><td>0.63</td><td>0.67</td></tr><tr><td>GEMMA 3 27B (US)</td><td>0.05</td><td>5.34</td><td>0.01</td><td>0.51</td></tr><tr><td>AYA EXPANSE 32B (CA)</td><td>0.39</td><td>2.89</td><td>0.14</td><td>0.42</td></tr><tr><td>MINISTRAL 14B (FR)†</td><td>2.45</td><td>2.98</td><td>0.82</td><td>0.87</td></tr><tr><td>QWEN3 NEXT 80B (CN)</td><td>1.39</td><td>2.81</td><td>0.49</td><td>0.53</td></tr></table>

Note. All effects are isolated distributions computed via the same optimal-transport (comonotone) coupling used throughout the paper; E is the mean shift on the composite CETSCALE score, $\mathrm { S N } \hat { \mathbf { R } } = | \mathbb { E } | / \mathrm { S D }$ , and dPD is the discrete Probability of Direction. Panel A is the entire country signal an aggregate benchmark can express; it is identical across models by construction. Panel B isolates the MODEL × TARGET interaction, which is orthogonal to both marginals and therefore identically zero under aggregation. Boldface marks interactions whose sign reverses relative to the corresponding Panel A main effect. <sup>†</sup>Ingroup cell (Target country matches the model’s country of origin).

$$
\hat { E } \sim \mathcal { N } \Bigg ( E _ { \mathrm { e x a c t } } , \sum _ { c } \alpha _ { c } ^ { 2 } \sigma _ { c } ^ { 2 } / N \Bigg )
$$

where $\alpha _ { c }$ are the contrast coefficients and $\sigma _ { c } ^ { 2 }$ is the composite variance of condition c. The flip probability $( P _ { \mathrm { f l i p } } )$ is the probability that the sam-

pling estimate carries the wrong sign,

$$
P _ { \mathrm { f i p } } = \Phi \left( - \frac { \mathrm { s i g n } ( E _ { \mathrm { e x a c t } } ) \mathbb { E } [ \hat { E } ] } { \mathrm { S D } ( \hat { E } ) } \right) ,\tag{96}
$$

where Φ is the standard normal CDF. We evaluate the realistic per-condition budgets $N \in$ 10, 100, 1000 under the default decoding setting $( T { = } 1 . 0 , \mathrm { t o p } { - } p { = } 1 . 0 )$ , isolating the cost of sampling noise alone.

Results. Table 12 bins parameters by their exact absolute effect size (in composite-score units on the 17–119 CETSCALE) and reports the mean flip probability. Figure 14 shows the full relationship between effect size and $P _ { \mathrm { f l i p } }$

Table 10: Standard error (composite-score units) of a sampling-based estimator of the CETSCALE score, averaged per model, as a function of the per-condition sample size N. Models are sorted by multivariate Consensus.
<table><tr><td>MODEL</td><td>Cns</td><td> $N { = } 1 0$ </td><td> $N { = } 1 0 ^ { 2 }$ </td><td> $N { = } 1 0 ^ { 3 }$ </td><td> $N { = } 1 0 ^ { 6 }$ </td></tr><tr><td>GEMMA 3-27B</td><td>0.947</td><td>0.304</td><td>0.096</td><td>0.030</td><td>0.001</td></tr><tr><td>AYA-32B</td><td>0.935</td><td>0.358</td><td>0.113</td><td>0.036</td><td>0.001</td></tr><tr><td>LLAMA 3.3-70B</td><td>0.912</td><td>0.493</td><td>0.156</td><td>0.049</td><td>0.002</td></tr><tr><td>QWEN3 NEXT-80B</td><td>0.909</td><td>0.517</td><td>0.164</td><td>0.052</td><td>0.002</td></tr><tr><td>MINISTRAL-14B</td><td>0.661</td><td>1.656</td><td>0.524</td><td>0.166</td><td>0.005</td></tr></table>

Table 11: Decoding-induced bias (composite-score units) relative to the native distribution, aggregated over the T  top-p grid. $M _ { | B i a s | }$ is the mean absolute bias; $S D _ { B i a s }$ its standard deviation across the grid. Models sorted by Consensus.
<table><tr><td>MODEL</td><td> $\mathrm { C n s }$ </td><td> $M _ { | B i a s | }$ </td><td> $S D _ { B i a s }$ </td></tr><tr><td>GEMMA 3-27B</td><td>0.95</td><td>0.18</td><td>0.25</td></tr><tr><td>AYA-32B</td><td>0.94</td><td>0.24</td><td>0.38</td></tr><tr><td>LLAMA 3.3-70B</td><td>0.91</td><td>0.59</td><td>0.75</td></tr><tr><td>QWEN3 NEXT-80B</td><td>0.91</td><td>0.33</td><td>0.53</td></tr><tr><td>MINISTRAL-14B</td><td>0.66</td><td>1.38</td><td>1.39</td></tr></table>

Table 12: Mean sign-flip probability of a sampling-based estimator under default decoding $( T { = } 1 . 0 , \mathrm { t o p } { - } p { = } 1 . 0 )$ stratified by exact effect size (composite-score units) and per-condition sample size N. Params is the share of all factorial parameters falling in each bin.
<table><tr><td> $| E _ { \mathrm { E X A C T } } |$ </td><td>PARAMS (%)</td><td> $N { = } 1 0$ </td><td> $N { = } 1 0 0$ </td><td> $N { = } 1 0 0 0$ </td></tr><tr><td>[0, 1)</td><td>24.1</td><td>0.180</td><td>0.063</td><td>0.013</td></tr><tr><td>[1, 2)</td><td>6.9</td><td>0.004</td><td>0.000</td><td>0.000</td></tr><tr><td>[2, 5)</td><td>41.4</td><td>0.002</td><td>0.000</td><td>0.000</td></tr><tr><td>[5, 10)</td><td>17.2</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>≥10</td><td>10.3</td><td>0.000</td><td>0.000</td><td>0.000</td></tr></table>

![](images/96f8c6bfff3af78c7d62baf97373c73bf04dcb3dc443cdf93ee742383e8544a4.jpg)  
Figure 14: Sign-flip probability of a sampling-based estimator as a function of the exact absolute effect size, for per-condition budgets $N \in \{ 1 0 , 1 0 0 , 1 0 0 0 \}$ under default decoding. Reliability degrades sharply for small effects and low sampling budgets, while large effects are recovered reliably at any N.

Findings. We observe three important findings. First, large and moderate effects are robust. Every parameter with $| E _ { \mathrm { e x a c t } } | \ge 2$ (a cumulative 68.9% of all effects) has a flip probability below 0.01 even at the smallest budget $N { = } 1 0$ . A researcher would recover the sign of these effects reliably. Second, the risk is concentrated entirely in the small-effect regime. Parameters with $| E _ { \mathrm { e x a c t } } | < 1$ , which constitute 24.1% of the factorial design, flip 18% of the time at $N { = } 1 0$ and still 6.3% of the time at the commonly used budget $N { = } 1 0 0$ . In our study these small effects include several substantively interesting interactions (e.g., the near-null Qwen3 China term), precisely the subtle country-of-origin signals a controlled study aims to detect. Third, the risk decays slowly: reducing the flip rate from 18% to 1.3% requires a 100 increase in sample size (from $N { = } 1 0 \ \mathrm { t o } \ N { = } 1 0 0 0 )$ , reflecting the $1 / \sqrt { N }$ convergence of the estimator. Because the exact-PMF framework attains the $N  \infty$ limit in a single forward pass, it removes this reliability–compute tradeoff, and guarantees that the sign and magnitude of every reported effect are free of sampling artifacts.

## A.10 Extending the Framework to Prompt Framing

A key strength of casting behavioral evaluation as a fully crossed factorial design is extensibility: any new source of variation can be introduced as an additional factor and analyzed through the identical distributional-ANOVA pipeline (Theorem 3.1, Corollaries A.9–A.11). We demonstrate this by promoting the administration prompt to a first-class experimental factor, $\lambda _ { \mathrm { P r o m p t } }$ . This yields a new capability. Rather than treating prompt phrasing as a fixed background choice, the framework isolates its causal effect on the measured construct, quantifies how it interacts with each model, and recovers the latent trait with framing held under statistical control.

We construct six system-prompt framings that preserve the measurement instrument exactly, namely, all retain the 1–7 scale, the agreement polarity, and the digit-only response schema, and hold every CETSCALE item verbatim. Only the surrounding framing wording varies (Table 13). We evaluate a fully crossed MODEL TARGET PROMPT design over three models spanning the consensus and ethnocentrism ranges observed in the main study (Gemma 3 27B, Aya Expanse 32B, Ministral 14B), four target countries, and the six framings, for $3 \times 4 \times 6 \times 1 7 = 1 { , } 2 2 4$ exact itemlevel PMFs.

Isolated effects under the three-factor design. Table 14 reports the full decomposition: The grand-mixture baseline $\mu _ { \emptyset }$ , the isolated main effects for all three factors, and the three families of two-way interactions (MODEL TARGET, MODEL  PROMPT, TARGET  PROMPT). Each row is an exact effect distribution summarized by mean shift (E), predictive dispersion (SD), signalto-noise ratio (SNR), and discrete probability of direction (dPD).

Three capabilities of the framework are visible directly in Table 14. First, the country-of-origin signal is recovered with framing controlled. The TARGET main effects reproduce the pattern of the main study (Canada +4.7, China 6.4, France $- 0 . 4 , \mathrm { U S A } + 2 . 1 )$ , and every TARGET  PROMPT interaction is small (all $| \mathbb { E } | < 1 . 5$ $\mathrm { S N R } \prec \ 0 . 4$ dPD near chance).

Second, prompt framing is quantified as its own causal effect. The PROMPT main effect spans roughly 13 composite-score points (+6.9 for v5 to 6.7 for v2), giving a direct, exact measurement of how administration framing shifts the construct level.

Third, model-specific framing sensitivity becomes measurable. The MODEL PROMPT block shows Gemma 3 27B interacting weakly with framing (all $| \mathbb { E } | < 3 . 9 , \mathbb { S } \mathbb { N } \mathbb { R } < 0 . 5 )$ , whereas Aya 32B and Ministral 14B interact strongly (Aya v2 SNR 2.2; Ministral v0 SNR 1.6). Consistent with these interactions, the prompt-controlled MODEL main effects recover the trait ordering of the main study, with Aya 32B highest (+9.1), Ministral 14B intermediate (+4.0), and Gemma 3 27B lowest ( 13.0). The reduced magnitudes relative to Table 6 reflect the three-model subset and its distinct grand mean $( \mu _ { \emptyset } = 6 2 . 3 )$ .

Table 13: Prompt framing variants. All preserve the 1–7 scale, agreement polarity, and digit-only response schema; every CETSCALE item is held verbatim.
<table><tr><td>Variant</td><td>Framing</td></tr><tr><td>v0_original</td><td>Consumer market-research persona; explicit 7-point scale; digit only.</td></tr><tr><td>v1_participant</td><td>Survey-participant framing; agreement scale  $1 - 7 ;$  single digit only.</td></tr><tr><td> $\mathbf { v } 2$  _concise</td><td>Minimal instruction; rate 1–7; output only the number.</td></tr><tr><td> $\mathbf { v } 3 _ { - }$  _instructional</td><td>Task-style instruction; scale 1 (disagree) to 7 (agree); digit 1–7 only.</td></tr><tr><td>v4_roleplay</td><td>Everyday-shopper roleplay; 7 = agree, 1 = disagree; digit only.</td></tr><tr><td> $\mathbf { v } 5 _ { - }$  _original_no_whitespace</td><td>v0 wording with whitespace/formatting normalized.</td></tr></table>

Table 14: Full three-factor effect decomposition (Part 1 of 2: Main effects and Model Target). Rows are isolated effect distributions summarized by mean shift (E), predictive dispersion (SD), signal-to-noise ratio (SNR), and discrete probability of direction (dPD).
<table><tr><td>PARAMETER</td><td>E</td><td>SD</td><td>SNR</td><td>dPD</td></tr><tr><td>µ0</td><td>62.34</td><td>13.49</td><td></td><td></td></tr><tr><td>AYA-32B</td><td>9.08</td><td>12.67</td><td>0.72</td><td>0.76</td></tr><tr><td>GEMMA 3-27B</td><td>-13.03</td><td>11.74</td><td>1.11</td><td>0.90</td></tr><tr><td>MINISTRAL-14B</td><td>3.96</td><td>9.61</td><td>0.41</td><td>0.70</td></tr><tr><td>CANADA</td><td>4.74</td><td>5.24</td><td>0.90</td><td>0.93</td></tr><tr><td>CHINA</td><td>-6.38</td><td>6.08</td><td>1.05</td><td>0.97</td></tr><tr><td>FRANCE</td><td>-0.44</td><td>4.78</td><td>0.09</td><td>0.48</td></tr><tr><td>USA</td><td>2.08</td><td>5.43</td><td>0.38</td><td>0.44</td></tr><tr><td>V0_ORIGINAL</td><td>6.74</td><td>10.16</td><td>0.66</td><td>0.64</td></tr><tr><td>V1_PARTICIPANT</td><td>-0.17</td><td>7.22</td><td>0.02</td><td>0.58</td></tr><tr><td>V2_CONCISE</td><td>-6.68</td><td>9.73</td><td>0.69</td><td>0.82</td></tr><tr><td>V3_INSTRUCTIONAL</td><td>-4.08</td><td>7.76</td><td>0.53</td><td>0.62</td></tr><tr><td>V4_ROLEPLAY</td><td>-2.67</td><td>6.94</td><td>0.38</td><td>0.68</td></tr><tr><td>V5_NO_WHITESPACE</td><td>6.86</td><td>9.32</td><td>0.74</td><td>0.74</td></tr><tr><td>AYA-32B × CANADA</td><td>0.31</td><td>2.14</td><td>0.15</td><td>0.35</td></tr><tr><td>AYA-32B × CHINA</td><td>0.36</td><td>4.01</td><td>0.09</td><td>0.51</td></tr><tr><td>AYA-32B × FRANCE</td><td>0.08</td><td>3.33</td><td>0.03</td><td>0.53</td></tr><tr><td>AYA-32B × USA</td><td>-0.75</td><td>3.47</td><td>0.22</td><td>0.48</td></tr><tr><td>GEMMA 3-27B × CANADA</td><td>2.14</td><td>3.28</td><td>0.65</td><td>0.62</td></tr><tr><td>GEMMA 3-27B × CHINA</td><td>-4.40</td><td>9.86</td><td>0.45</td><td>0.59</td></tr><tr><td>GEMMA 3-27B × FRANCE</td><td>-1.83</td><td>5.38</td><td>0.34</td><td>0.55</td></tr><tr><td>GEMMA 3-27B × USA</td><td>4.09</td><td>3.47</td><td>1.18</td><td>0.88</td></tr><tr><td>MINISTRAL-14B × CANADA</td><td>-2.45</td><td>3.08</td><td>0.80</td><td>0.77</td></tr><tr><td>MINISTRAL-14B × CHINA</td><td>4.04</td><td>4.52</td><td>0.89</td><td>0.82</td></tr><tr><td>MINISTRAL-14B × FRANCE</td><td>1.74</td><td>2.48</td><td>0.70</td><td>0.74</td></tr><tr><td>MINISTRAL-14B × USA</td><td>-3.34</td><td>4.10</td><td>0.81</td><td>0.77</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 15: Full three-factor effect decomposition (Part 2 of 2: Model  Prompt and Target  Prompt interactions).
<table><tr><td>PARAMETER</td><td>E</td><td>SD</td><td>SNR</td><td>dPD</td></tr><tr><td>AYA-32B × V0</td><td>9.20</td><td>14.27</td><td>0.64</td><td>0.66</td></tr><tr><td>AYA-32B × V1</td><td>-3.14</td><td>7.48</td><td>0.42</td><td>0.66</td></tr><tr><td>AYA-32B × V2</td><td>-9.64</td><td>4.36</td><td>2.21</td><td>1.00</td></tr><tr><td> $\mathbf { A Y A } – 3 2 \mathbf { B } \times \mathbf { V } 3$ </td><td>-5.12</td><td>6.00</td><td>0.85</td><td>0.84</td></tr><tr><td> $\mathbf { A Y A - } 3 2 \mathbf { B } \times \mathbf { V } 4$ </td><td>0.13</td><td>8.15</td><td>0.02</td><td>0.51</td></tr><tr><td>AYA-32B × V5</td><td>8.57</td><td>12.80</td><td>0.67</td><td>0.66</td></tr><tr><td>GEMMA 3-27B × v0</td><td>-0.99</td><td>5.35</td><td>0.18</td><td>0.47</td></tr><tr><td>GEMMA 3-27B × V1</td><td>-1.48</td><td>3.10</td><td>0.48</td><td>0.64</td></tr><tr><td>GEMMA 3-27B × V2</td><td>3.89</td><td>2.14</td><td>1.82</td><td>0.97</td></tr><tr><td>GEMMA 3-27B × V3</td><td>0.90</td><td>2.80</td><td>0.32</td><td>0.67</td></tr><tr><td>GEMMA 3-27B × V4</td><td>-0.11</td><td>3.41</td><td>0.03</td><td>0.44</td></tr><tr><td>GEMMA 3-27B × V5</td><td>-2.20</td><td>4.84</td><td>0.46</td><td>0.56</td></tr><tr><td>MINISTRAL-14B × V0</td><td>-8.22</td><td>5.09</td><td>1.61</td><td>1.00</td></tr><tr><td>MINISTRAL-14B × V1</td><td>4.62</td><td>3.39</td><td>1.36</td><td>0.99</td></tr><tr><td>MINISTRAL-14B × V2</td><td>5.75</td><td>3.77</td><td>1.53</td><td>0.95</td></tr><tr><td>MINISTRAL-14B × V3</td><td>4.23</td><td>3.80</td><td>1.11</td><td>0.96</td></tr><tr><td>MINISTRAL-14B × V4</td><td>-0.01</td><td>3.81</td><td>0.00</td><td>0.64</td></tr><tr><td>MINISTRAL-14B × V5</td><td>-6.37</td><td>4.41</td><td>1.44</td><td>0.96</td></tr><tr><td>CANADA × V0</td><td>-1.04</td><td>2.88</td><td>0.36</td><td>0.80</td></tr><tr><td>CANADA × V1</td><td>0.26</td><td>2.58</td><td>0.10</td><td>0.59</td></tr><tr><td>CANADA × V2</td><td>1.28</td><td>3.36</td><td>0.38</td><td>0.57</td></tr><tr><td>CANADA × V3</td><td>0.01</td><td>2.66</td><td>0.00</td><td>0.42</td></tr><tr><td>CANADA × V4</td><td>-0.22</td><td>2.27</td><td>0.10</td><td>0.36</td></tr><tr><td>CANADA × V5</td><td>-0.29</td><td>3.17</td><td>0.09</td><td>0.54</td></tr><tr><td>CHINA × V0</td><td>0.46</td><td>3.56</td><td>0.13</td><td>0.65</td></tr><tr><td>CHINA × V1</td><td>0.75</td><td>3.30</td><td>0.23</td><td>0.37</td></tr><tr><td>CHINA × V2</td><td>-1.45</td><td>4.88</td><td>0.30</td><td>0.46</td></tr><tr><td>CHINA × V3</td><td>-0.35</td><td>3.52</td><td>0.10</td><td>0.58</td></tr><tr><td>CHINA × V4</td><td>0.76</td><td>3.64</td><td>0.21</td><td>0.45</td></tr><tr><td>CHINA × v5</td><td>-0.17</td><td>4.16</td><td>0.04</td><td>0.56</td></tr><tr><td>FRANCE × V0</td><td>0.42</td><td>3.43</td><td>0.12</td><td>0.37</td></tr><tr><td>FRANCE × V1</td><td>-0.93</td><td>3.37</td><td>0.27</td><td>0.62</td></tr><tr><td>FRANCE × V2</td><td>-0.21</td><td>3.38</td><td>0.06</td><td>0.46</td></tr><tr><td>FRANCE × V3</td><td>1.19</td><td>4.08</td><td>0.29</td><td>0.59</td></tr><tr><td>FRANCE × V4</td><td>-0.55</td><td>3.44</td><td>0.16</td><td>0.41</td></tr><tr><td>FRANCE × V5</td><td>0.07</td><td>3.73</td><td>0.02</td><td>0.42</td></tr><tr><td>USA × v0</td><td>0.16</td><td>2.39</td><td>0.06</td><td>0.45</td></tr><tr><td>USA × V1</td><td>-0.09</td><td>2.17</td><td>0.04</td><td>0.44</td></tr><tr><td>USA × V2</td><td>0.38</td><td>3.14</td><td>0.12</td><td>0.46</td></tr><tr><td>USA × V3</td><td>-0.84</td><td>2.43</td><td>0.35</td><td>0.57</td></tr><tr><td>USA × v4</td><td>0.01</td><td>2.20</td><td>0.00</td><td>0.46</td></tr><tr><td>USA × v5</td><td>0.39</td><td>2.91</td><td>0.13</td><td>0.48</td></tr></table>