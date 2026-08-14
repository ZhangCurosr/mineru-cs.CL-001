# Decoupled Contrastive Decoding via Expert-Aligned Drafting

Zhixuan Liu<sup>1,2</sup>\* Zhichen Dong<sup>1,2</sup> Yuanfu Wang<sup>2</sup> Chao Yang<sup>2</sup>

<sup>1</sup>Shanghai Jiao Tong University

<sup>2</sup>Shanghai Artificial Intelligence Laboratory

## Abstract

Contrastive Decoding (CD) improves generation quality, but its amateur-model pass makes decoding expensive. Accelerating CD with speculative decoding raises a proposalalignment question: should the contrastive signal shape the drafter, or should it remain only in verification? We study this question in the lightweight feature-level drafter regime. Two controlled diagnostics, matched Cross-α training and an Approximate Dual-Drafter decomposition, give the same diagnosis: contrastive-aware drafting does not consistently improve over expert-aligned drafting because the contrastive correction is usually weaker than drafter error, and reconstruction can amplify that error. We introduce Decoupled Contrastive Decoding (DCD), which drafts with an expert-aligned lightweight proposer and applies the amateur only in unchanged CD verification. Standard speculative verification preserves the vanilla-CD output distribution. Across the main 8B settings, EAGLE3-based DCD achieves average greedy speedups of 1.65 to 1.95× over vanilla CD and reduces MMLU proposal-path latency by about 5 to 12× relative to amateur-coupled proposal paths.

## 1 Introduction

Large Language Models (LLMs) still hallucinate and make reasoning errors (Huang et al., 2025; Plaat et al., 2024). Contrastive Decoding (CD) (Li et al., 2023; O’Brien and Lewis, 2023) improves generation by correcting a strong expert with a weaker amateur. It also underlies lightweight tuning schemes such as Emulator Fine-Tuning (Mitchell et al., 2024) and Proxy Tuning (Liu et al., 2024), but each token requires both models.

![](images/8523073362166b9796da8425329a34ace6eddbb0d0413faefb868b49d71ed84a.jpg)  
Figure 1: DCD keeps the CD target in verification while avoiding an amateur-coupled proposal route. The expert $\mathcal { M } _ { p }$ and amateur $\mathcal { M } _ { q }$ still define $\pi _ { C D } ;$ the lightweight proposer E changes only the serial draft path.

Speculative decoding can reduce this cost, but CD adds a proposal-alignment choice absent from single-model decoding: the serial proposal path can use the amateur signal, imitate the full contrastive distribution, or leave contrastive scoring to verification. We study three proposal routes: amateurcoupled proposals, contrastive-aware lightweight proposals, and expert-aligned proposals with contrastive verification. Figure 1 illustrates the systemlevel paths in the main comparison: vanilla CD, amateur-coupled SCD, and DCD with expertaligned proposals plus contrastive verification.

We first diagnose proposal-side contrastive alignment under matched lightweight-drafter conditions. In the Cross-α diagnostic, a dual-input EAGLE variant receives concatenated expert and amateur hidden states; within that variant, varying train-α changes only the target distribution, while the architecture, data, and training recipe remain fixed. An Approximate Dual-Drafter separately approximates expert and amateur distributions and recombines them at inference. Both diagnostics show the same failure mode: the contrastive signal is usually weaker than the drafter error it must overcome, and reconstruction can amplify that error. Across 24 configurations, 81.1% of positions have a contrastive signal below 1.0, while 48.7% have expert-side proposal $D _ { K L } \geq 2 . 0 ;$ positive ∆Top-1 values concentrate in high-signal, low-error cells. Neither contrastive-aware variant yields consistent accepted-length gains over expert-aligned drafting (Section 3.1).

DCD validates the expert-aligned route at system level. It reuses a lightweight expert-aligned drafter for proposals and retains the amateur only in verification, so EAGLE-style drafting can accelerate CD without retraining for each amateur. SCD and CoS keep the amateur in the serial proposal loop.

Instantiated with EAGLE3 (Li et al., 2024a,b, 2025), $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ achieves average greedy speedups of 1.65 to 1.95× over vanilla CD and reduces MMLU per-step proposal latency by about 5 to 12× relative to amateur-coupled proposal paths. $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ remains close to 2× in a greedyonly 70B extension, and a DCD variant with a matched N-gram proposer also gives average speedups above vanilla CD. Because DCD changes only the proposal path, its lossless guarantee follows from standard speculative verification.

Our contributions are threefold: (1) we formulate proposal alignment as the central design choice in speculative contrastive decoding, separating amateur-coupled, contrastive-aware, and expert-aligned proposal routes; (2) we provide controlled diagnostics showing that contrastive-aware lightweight drafting does not reliably improve accepted length over expert-aligned drafting, because the contrastive signal is often smaller than proposal error and reconstruction can amplify that error; and (3) we instantiate this diagnosis as Decoupled Contrastive Decoding (DCD), a lossless CD accelerator that keeps contrastive scoring in verification and achieves deployment-level speedups with both EA-GLE3 and matched N-gram proposers.

Our code is available at https://github. com/chadlzx/dcd.

## 2 Methodology

This section formalizes proposal alignment and the DCD verification recipe.

## 2.1 Preliminaries

Contrastive Decoding. Given history $h ,$ expert ${ \mathcal { M } } _ { p } ,$ and amateur $\mathcal { M } _ { q } ,$ Contrastive Decoding (Li et al., 2023; O’Brien and Lewis, 2023) favors tokens high under the expert and low under the amateur:

$$
\begin{array} { r } { \log \pi _ { C D } ( x \mid h ) = ( 1 + \alpha ) \log \pi _ { p } ( x \mid h ) \qquad } \\ { - \alpha \log \pi _ { q } ( x \mid h ) - \log Z _ { C D } ( h ) , } \end{array}
$$

where $\alpha \geq 0$ controls contrastive strength. Equivalently,

$$
\pi _ { C D } ( x \mid h ) = \frac { 1 } { Z _ { C D } ( h ) } \pi _ { p } ( x \mid h ) ^ { 1 + \alpha } \pi _ { q } ( x \mid h ) ^ { - \alpha } ,
$$

$$
Z _ { C D } ( h )\tag{1}
$$

Rewriting Eq. 1 as $\pi _ { C D } ( x \mid h ) \propto \pi _ { p } ( x$ | $h ) \left( \frac { \pi _ { p } ( x | h ) } { \pi _ { q } ( x | h ) } \right) ^ { \mathrm { { o } } }$ separates the dominant expert term from the contrastive factor that a contrastive-aware lightweight drafter must also model.

The original CD plausibility constraint (Li et al., 2023) is also compatible with DCD: masking Eq. 1 to $\begin{array} { r } { \mathcal { P } _ { \beta } ( h ) = \{ x : \pi _ { p } ( x \mid h ) \ge \beta \operatorname* { m a x } _ { x ^ { \prime } } \pi _ { p } ( x ^ { \prime } \mid h ) \} } \end{array}$ and renormalizing still yields a valid speculativeverification target. Proposals outside the mask simply receive zero target probability, so losslessness is unchanged after renormalization. Our experiments use the untruncated target to isolate proposal alignment without adding a threshold hyperparameter.

Speculative Decoding. Speculative Decoding (SD) (Leviathan et al., 2023; Chen et al., 2023) drafts $\gamma$ candidate tokens and verifies them in parallel with the target model. Its speed depends on draft-target alignment; we use $D _ { K L } ( \pi _ { \mathrm { d r a f t } } \Vert \pi _ { \mathrm { t a r g e t } } )$ as a diagnostic proxy and report acceptance directly as mean accepted length L.

## 2.2 Diagnosing Proposal-Side Alternatives in the Lightweight-Drafter Regime

We compare three routes under lightweight drafting. They differ only in where the amateur signal enters the speculative decoding pipeline: directly in proposal generation, indirectly through a contrastive-aware lightweight proposer, or only during verification.

Route 1: Direct amateur drafting. This route uses $\pi _ { q }$ for proposals even though the unnormalized CD score assigns coefficient −α to log $\pi _ { q } ( x \mid$ $h )$ . It therefore has both a distributional mismatch and a systems cost: serial proposal cost remains tied to the amateur, roughly $\gamma \cdot t _ { q }$ before methodspecific bonus or delayed-drafting terms.

Route 2: Contrastive-aware lightweight drafting. A lightweight drafter must approximate $\pi _ { p }$ and the extra contrastive factor $\left( { \frac { \pi _ { p } } { \pi _ { q } } } \right) ^ { \alpha }$ . Training on the contrastive target increases modeling burden without removing baseline proposal error, and inference-time reconstruction adds another error source. Section 3.1 shows that this added signal is usually weaker than the proposal error it must overcome.

Route 3: Expert-aligned proposals with contrastive verification. Decoupled Contrastive Decoding (DCD) proposes $\tilde { y } _ { 1 : \gamma }$ without using $\mathcal { M } _ { q }$ in the serial proposal path. The expert and amateur are coupled only during verification, where proposals are evaluated against the unchanged $\pi _ { C D }$ Algorithm 1 summarizes one DCD round. $\mathsf { A p - }$ pendix B.3 proves that replacing the proposer does not change the output distribution as long as speculative verification targets the normalized CD distribution $\pi _ { C D } ;$ the proposer only changes acceptance rate and efficiency. We instantiate the proposer E with EAGLE3 (Li et al., 2025) for the main deployment experiments because it gives the strongest average operating point among our tested proposers. This EAGLE3 instantiation is expertaligned; the same DCD verification rule can also wrap other amateur-independent proposers, such as the N-gram proposer.

DCD separates distributional correctness from proposal efficiency. Because verification still targets $\pi _ { C D }$ , changing the proposer does not change the output distribution of vanilla CD under standard speculative sampling. The change falls on the serial proposal path: DCD replaces the amateur cost of roughly $\gamma \cdot t _ { q }$ with the cost of the chosen proposer, namely $\gamma \cdot t _ { e }$ for EAGLE3 with $t _ { e } \ll t _ { q } ,$ or near-zero measured proposal latency for the Ngram variant.

## 2.3 Alpha-Robustness Intuition

Effective Draft Alignment means that the proposer places more mass than the amateur on tokens where the expert has a larger log-likelihood advantage. Under this condition, the amateur proposal’s drafttarget KL grows faster with α than an expertaligned drafter’s KL. This slope comparison suggests that expert-aligned DCD can lose accepted length more slowly even when SCD or CoS start with a higher acceptance baseline. Appendix B proves the KL-slope inequality and connects it to the observed robustness trend.

Algorithm 1: Decoupled Contrastive De  
coding (One Round)   
Data: $\mathbf { y } = [ \cdots , y _ { 0 } ] , \mathbf { h } _ { d r a f t } ^ { ( 0 ) } , \mathcal { M } _ { p } , \mathcal { M } _ { q } , E , \gamma , \alpha$   
Result: updated prefix after one DCD round   
1 y˜<sub>0</sub> $ y _ { 0 }$   
$/ /$ Draft generation   
2 for $i  1 \ : \mathrm { t } \mathbf { 0 } \ : \gamma$ do   
3 $\pi _ { e } ^ { ( i ) } , \mathbf { h } _ { d r a f t } ^ { ( i ) }  E ( \mathbf { h } _ { d r a f t } ^ { ( i - 1 ) } , \tilde { y } _ { i - 1 } )$   
4 $\tilde { y } _ { i } \sim \pi _ { e } ^ { ( i ) }$   
5 $\tilde { \mathbf { y } }  [ \tilde { y } _ { 1 } , \dotsc , \tilde { y } _ { \gamma } ]$   
// Parallel model evaluation   
6 $[ \pi _ { p } ^ { ( 1 ) } , \ldots , \pi _ { p } ^ { ( \gamma + 1 ) } ]  M _ { p } ( \mathbf { y } \oplus \tilde { \mathbf { y } } )$   
7 $[ \pi _ { q } ^ { ( 1 ) } , \ldots , \pi _ { q } ^ { ( \gamma + 1 ) } ] \gets M _ { q } ( \mathbf { y } \oplus \tilde { \mathbf { y } } )$   
// Speculative accept or fallback   
8 for i ← 1 to γ do   
9 compute $\pi _ { C D } ^ { ( i ) }$ from $\mathrm { E q . } ( 1 )$ using $\pi _ { p } ^ { ( i ) }$ and $\pi _ { q } ^ { ( i ) }$   
and draw $r _ { i } \sim \mathcal { U } ( 0 , \hat { 1 } )$   
10 k ← min $\begin{array} { r } { \left( \left\{ i \mid r _ { i } > \frac { \pi _ { C D } ^ { ( i ) } ( \tilde { y } _ { i } ) } { \pi _ { e } ^ { ( i ) } ( \tilde { y } _ { i } ) } \right\} \cup \{ \gamma + 1 \} \right) } \end{array}$   
11 if $\natural \leq \gamma$ then   
12 accept $\tilde { y } _ { 1 : k - 1 }$   
13 $P _ { r } ( \bar { x } ) $   
Normalize(max $( 0 , \pi _ { C D } ^ { ( k ) } ( x ) - \pi _ { e } ^ { ( k ) } ( x ) ) )$   
14 sample $y _ { k } \sim P _ { r }$ and return $\mathbf { y } \oplus { \tilde { y } } _ { 1 : k - 1 } \oplus y _ { k }$   
15 else   
16 compute $\pi _ { C D } ^ { ( \gamma + 1 ) }$ from Eq. (1) using $\pi _ { p } ^ { ( \gamma + 1 ) }$ and   
$\pi _ { q } ^ { ( \bar { \gamma } + 1 ) }$   
17 sample $y _ { \gamma + 1 } \sim \pi _ { C D } ^ { ( \gamma + 1 ) }$ and return   
$\mathbf { y } \oplus \tilde { y } _ { 1 : \gamma } \oplus y _ { \gamma + 1 }$

## 3 Experiments

The experiments follow the design question from Section 2.2. We first ask whether lightweight drafters benefit from absorbing the contrastive signal, then evaluate the expert-aligned DCD route at deployment scale and explain its speed source.

Shared setting. Unless noted otherwise, we use 200 samples per dataset (or the full dataset if smaller), greedy decoding $( T { = } 0 )$ , draft length $\gamma { = } 5 ,$ , and chain-style, greedy drafting for EAGLE3 proposers. Main deployment tables report threerun averages; diagnostics and ablation studies use single-run traces. Runtime flags are listed in Appendix H.

## 3.1 Why Not Contrastive-Aware Drafting?

This subsection examines Route 2 from Section 2.2: moving the contrastive signal into lightweight proposal generation before verification. We first characterize where such alignment could help, then test whether lightweight drafters can exploit that region under matched controls, and finally examine whether the negative result persists with a substantially stronger proposer.

![](images/3a9b4a7d992b605e6c46ae1f09a57bbf90b2b805362af63a44005e2b9a6d119a.jpg)

(b) Effect by signal and proposal error  
(c) KL(π<sub>e</sub> ∥ π<sub>p</sub>) distribution  
![](images/fc79a75b8dd892f3e727f14d64082d22d553629f1235e3f9a165886c3626633a.jpg)

![](images/4d17d5dc9faed59ee0b7b9dc0ec2fcec21d7038016f4e40debbc1c18dd09bb77.jpg)  
KL $\left( \pi _ { e } \parallel \pi _ { p } \right)$ bucket  
Figure 2: Contrastive-aware lightweight drafting has only a small favorable region to exploit. Most positions have weak signal (81.1% below 1.0), gains concentrate in high-signal/low-error cells, and 48.7% have KL $\left( \pi _ { e } \| \pi _ { p } \right) \ge$ 2.0. Black dots mark cells with fewer than 100 positions. Signal is $| \log ( \pi _ { p } ( x ^ { \star } \mid h ) / \pi _ { q } ( x ^ { \star } \mid h ) ) |$ at $x ^ { \star } =$ arg max π (x | h); ∆Top-1 compares contrastive-aligned and expert-aligned Top-1 agreement with $x ^ { \star }$

Signal–Proposal-Error Diagnostic. Before testing contrastive-aware drafting, we first ask whether the contrastive signal occupies a sufficiently favorable region relative to baseline proposal error. At each position, let $x ^ { \star } \ = \ \arg \operatorname* { m a x } _ { x } \pi _ { C D } ( x \ |$ h). We measure the contrastive signal magnitude as $\begin{array} { r } { \left| \log { \frac { \pi _ { p } ( x ^ { \star } | h ) } { \pi _ { q } ( x ^ { \star } | h ) } } \right| } \end{array}$ , baseline expert-side proposal error as $D _ { K L } ( \dot { \pi } _ { e } \| \pi _ { p } )$ for the expert-aligned drafter $\pi _ { e } ,$ and ∆Top-1 as the difference between contrastive-aligned and expert-aligned Top-1 agreement with $x ^ { \star }$ . Figure 2 aggregates these quantities, count-weighted, over 24 configurations: LLAMA-3, LLAMA-EFT, and QWEN3, each evaluated on GSM8K and MMLU at α ∈ {0.1, 0.3, 0.5, 1.0}.

Across all pooled positions in this countweighted diagnostic, 81.1% have signal below 1.0 and 48.7% have $D _ { K L } ( \pi _ { e } \| \pi _ { p } ) ~ \geq ~ 2 . 0 \colon$ positive ∆Top-1 values concentrate in high-signal, lowerror cells. Thus, the tested regime leaves only a relatively small favorable region for contrastiveaware proposal alignment, while substantial baseline proposal error is widespread. We use ∆Top-1 only as a position-level diagnostic; whether these local gains translate into accepted-length improvements is tested directly in the matched experiments below.

Matched Lightweight Drafting Experiments. We next test whether a lightweight drafter can exploit this signal in a controlled setting. Both variants use the matched LLAMA-3/QWEN3 pairs on GSM8K/MMLU, with the dual-input EAGLE architecture, ShareGPT training data, and training recipe fixed. The direct variant learns the contrastive target itself; the decomposed variant separately learns expert- and amateur-aligned distributions and recombines them at inference time.

Direct Contrastive-Aligned Drafter. The most direct realization trains a dual-input EAGLE drafter to model the contrastive residual in Eq. 1:

$$
\begin{array} { r l } & { \frac { \pi _ { C D } ( x \mid h ) } { \pi _ { p } ( x \mid h ) } \propto \Big ( \frac { \pi _ { p } ( x \mid h ) } { \pi _ { q } ( x \mid h ) } \Big ) ^ { \alpha } = e ^ { \alpha s ( x \mid h ) } , } \\ & { \qquad s ( x \mid h ) = \log \frac { \pi _ { p } ( x \mid h ) } { \pi _ { q } ( x \mid h ) } . } \end{array}
$$

We vary both train-α and infer-α over {0.0, 0.1, 0.3, 0.5} while holding the remaining setup fixed; train-α = 0 recovers the expert-aligned training target. Table 1 shows no consistent gain from non-zero train-α: only Llama-3/GSM8K has a small gain averaged over the four infer-α values (+0.015 in L, or 0.77%), whereas train-α = 0 is best in the other three settings. Thus, in this

## (a) Effect by signal and proposal error

![](images/71e6d5e4a764d251745036f3da2a5a7622123e19da89a3da27b4ae45fa5d69dd.jpg)

(b) Proposer-side KL marginal  
![](images/6e56a93535364e54ceddd8598bbe83f430964cc652bdaecc1b9e6ff97ac6055b.jpg)  
KL $, ( \pi _ { r } \parallel \pi _ { p } )$ bucket  
Figure 3: Even a stronger same-family proposer leaves gains confined to high-signal, low-KL regions, with negative overall effect. ∆Top-1 is the CD-aware reconstruction’s Top-1 agreement gain over the unmodified 3B proposer.

matched control, directly learning the contrastive residual does not reliably translate into acceptedlength gains. The full matrix is in Appendix A.2.

Decomposed Dual-Drafter. Directly modeling $\pi _ { C D }$ may be unnecessarily difficult, so we also test a more favorable decomposition that separates expert- and amateur-side approximation before recombining the two analytically at inference time,

$$
\hat { \pi } _ { C D } ( x \mid h ) \propto \pi _ { e } ( x \mid h ) ^ { 1 + \alpha } \pi _ { f } ( x \mid h ) ^ { - \alpha } ,
$$

where $\pi _ { e }$ and $\pi _ { f }$ are the expert- and amateuraligned draft distributions. This separates expertand amateur-side approximation, but it amplifies expert-side log-approximation error by $1 + \alpha ; \mathrm { A p } \cdot$ pendix A.1 provides the reconstruction derivation.

Even under this more favorable formulation, Approximate Dual-Drafter underperforms expertaligned drafting at every reported α in mean accepted length L, draft-to-target $D _ { K L } ( \cdot \| \pi _ { C D } )$ , and Top-1 Match, with gaps widening as α increases (Appendix Table 6). At the position level, πˆ<sub>CD</sub> is closer to π<sub>CD</sub> than $\pi _ { e }$ at only 31.9–39.7% of positions.

<table><tr><td>Model/Data</td><td>Best train-α</td><td> $\mathbf { A v g } \ \Delta L$   $\mathbf { v } \mathbf { s } \alpha { = } 0$ </td><td> $\mathbf { A v g } \ \% \ \mathbf { g a i n }$   $\mathbf { v } \mathbf { s } \alpha { = } 0$ </td><td> $\alpha { = } 0$  wins</td></tr><tr><td>Llama-3/GSM8K</td><td>0.1</td><td>+0.015</td><td>+0.77%</td><td>0/4</td></tr><tr><td>Llama-3/MMLU</td><td>0.0</td><td>+0.000</td><td>+0.00%</td><td>3/4</td></tr><tr><td>Qwen3/GSM8K</td><td>0.0</td><td>+0.000</td><td>+0.00%</td><td>3/4</td></tr><tr><td>Qwen3/MMLU</td><td>0.0</td><td>+0.000</td><td>+0.00%</td><td>4/4</td></tr></table>

Table 1: Cross-α control: contrastive-target training does not consistently improve L. Best train-α maximizes average $\overline { { L } } ;$ only Llama-3/GSM8K shows a small non-zero gain. Full matrix in Appendix A.2.

Stress Test with a Stronger Full-LM Proposer. The preceding negative results could reflect limitations of the lightweight proposer rather than the contrastive objective itself. We therefore repeat the position-level diagnostic with LLAMA-3.2-3B-INSTRUCT as a stronger Llama-family proposer, while fixing the LLAMA-3.1-8B-INSTRUCT/LLAMA-3.2-1B-INSTRUCT expert–amateur pair. On shared response histories, we apply contrastive correction using true amateur logits. This is an optimistic offline check on GSM8K and MMLU at $\alpha \in \{ 0 . 1 , 0 . 5 \}$ rather than an online rollout evaluation. Figure 3 shows that the stronger proposer enlarges the favorable high-signal, low-error pocket, but the settinglevel mean ∆Top-1 remains negative in all four settings (see also Appendix A.3). The stronger proposer therefore enlarges the favorable region in this check without reversing the aggregate disadvantage.

Takeaway. Neither direct nor decomposed contrastive-aware lightweight drafting consistently outperforms expert-aligned drafting in the tested regime. The stronger full-LM stress test enlarges the favorable region but does not reverse the position-level disadvantage. We therefore keep proposals expert-aligned and amateur-independent in the system experiments, applying contrastive scoring only during verification.

## 3.2 Deployment-Level Setup

Datasets and model pairs. Following Fu et al. (2025), we evaluate HumanEval (Chen et al., 2021), GSM8K (Cobbe et al., 2021), MMLU (Hendrycks et al., 2021), and CNN/DM (See et al., 2017) under three 8B-class configurations: LLAMA-3, QWEN3, and LLAMA-EFT. Their expert and amateur pairs are LLAMA-3.1-8B-INSTRUCT/LLAMA-3.2-1B-INSTRUCT, QWEN3-8B/QWEN3-1.7B, and LLAMA-3.1-8B-INSTRUCT/LLAMA-3.1-8B. Baselines are vanilla CD, SCD, and CoS (Fu et al., 2025). $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ uses community-released EA-GLE3 drafters and requires no new drafter training. $\mathrm { D C D } _ { \mathrm { N G R A M } }$ is reported as an alternative DCD instantiation with a simple prefix-matching N-gram proposer. Appendix G lists the checkpoint identifiers.

Fairness protocol. All deployment methods use the same SGLang stack, single NVIDIA H200 GPU, prompts, maximum generation lengths, and expert and amateur pair within each operating point. The main speed tables use 200 samples per dataset with maximum generation length 256 and report three-run averages; the task-level accuracy check uses 1000 samples with maximum generation length 1024. Appendix Table 13 gives the draft-length tuning check.

Operating points and metrics. The main greedy and $T = 1$ comparisons use $\alpha \in \{ 0 . 1 , 0 . 5 \}$ ; the latency decomposition uses $\alpha = 0 . 1 \mathrm { ; }$ and the ablations sweep α and γ. We report tokens/sec and speedup over vanilla CD. When diagnosing speed sources, we also report accepted length and latency components. Appendix H gives runtime details.

## 3.3 System Validation of the Diagnostic: DCD

Main comparison. Table 2 shows $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ has the highest average speedup for every family and both α values. At the harder QWEN3, $\alpha = 0 . 5$ setting, SCD and CoS drop below 1.0× on MMLU/CNN-DM, while $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ reaches $1 . 6 0 { \times } / 1 . 3 9 { \times } . \mathrm { D C D } _ { \mathrm { N G R A M } }$ is included as a proposeragnostic sanity check: it is weaker than EA-GLE3 but remains above vanilla CD on average. Appendix E.1 gives the T=1 breakdown, Appendix F.1 reports multi-request serving, and Appendix F.2 gives the KV-cache result.

This pattern validates the proposal-path diagnostic at system level. All methods use the same contrastive target and the same expert and amateur models, but DCD alone moves the expensive amateur out of the serial proposal path. The next two subsections separate the two remaining concerns: Figure 4 checks that the verified CD target is preserved in task accuracy, and the latency decomposition in Section 3.5 shows why the cheaper proposal path can dominate even when SCD or CoS accept more draft tokens.

<table><tr><td>Method</td><td>HE</td><td>GSM</td><td>MMLU</td><td>CNN/DM</td><td>Avg.</td></tr><tr><td colspan="6">LLAMA-3</td></tr><tr><td colspan="6"> $\alpha = 0 . 1$ </td></tr><tr><td>SCD</td><td>1.52×</td><td>1.53×</td><td>1.26×</td><td>1.40×</td><td>1.43×</td></tr><tr><td>CoS</td><td>1.71×</td><td>1.77×</td><td>1.29×</td><td>1.37×</td><td>1.54×</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>1.48×</td><td>1.43×</td><td>1.29×</td><td>2.07×</td><td>1.56×</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>2.07×</td><td>1.84×</td><td>1.60×</td><td>1.80×</td><td>1.83×</td></tr><tr><td colspan="6"> $\alpha = 0 . 5$ </td></tr><tr><td>SCD</td><td>1.59×</td><td>1.43×</td><td>1.21×</td><td>1.12×</td><td>1.34×</td></tr><tr><td>CoS</td><td>1.67×</td><td>1.56×</td><td>1.20×</td><td>1.13×</td><td>1.40×</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>1.39×</td><td>1.34×</td><td>1.55×</td><td>2.10×</td><td>1.59×</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>1.89×</td><td>1.66×</td><td>1.44×</td><td>1.59×</td><td>1.65×</td></tr><tr><td colspan="6">QWEN3</td></tr><tr><td colspan="6"> $\alpha = 0 . 1$ </td></tr><tr><td>SCD</td><td>1.38×</td><td>1.35×</td><td>0.99×</td><td>1.05×</td><td>1.19×</td></tr><tr><td>CoS</td><td>1.45×</td><td>1.33×</td><td>1.17×</td><td>1.14×</td><td>1.27×</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>1.45×</td><td>1.35×</td><td>1.17×</td><td>1.14×</td><td>1.28×</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>2.30×</td><td>2.01×</td><td>1.66×</td><td>1.85×</td><td>1.95×</td></tr><tr><td colspan="6"> $\alpha = 0 . 5$ </td></tr><tr><td>SCD</td><td>1.23×</td><td>1.24×</td><td>0.88×</td><td>0.82×</td><td>1.05×</td></tr><tr><td>CoS</td><td>1.25×</td><td>1.31×</td><td>0.97×</td><td>0.79×</td><td>1.08×</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>1.32×</td><td>1.28×</td><td>1.17×</td><td>1.00×</td><td>1.19×</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>1.98×</td><td>1.85×</td><td>1.60×</td><td>1.39×</td><td>1.71×</td></tr><tr><td colspan="6">LLAMA-EFT</td></tr><tr><td colspan="6"> $\alpha = 0 . 1$ </td></tr><tr><td>SCD</td><td>1.20×</td><td>1.16×</td><td>1.06×</td><td>1.03×</td><td>1.12×</td></tr><tr><td>CoS</td><td>1.29×</td><td>1.22×</td><td>1.01×</td><td>1.06×</td><td>1.15×</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>1.43×</td><td>1.31×</td><td>1.32×</td><td>2.06×</td><td>1.52×</td></tr><tr><td> $\mathbf { D C D _ { E A G L E 3 } }$ </td><td>2.03×</td><td>1.74×</td><td>1.55×</td><td>1.74×</td><td>1.77×</td></tr><tr><td colspan="6"> $\alpha = 0 . 5$ </td></tr><tr><td>SCD</td><td>1.19×</td><td>1.14×</td><td>0.96×</td><td>0.92×</td><td>1.05×</td></tr><tr><td>CoS</td><td>1.31×</td><td>1.15×</td><td>0.98×</td><td>0.97×</td><td>1.10×</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>1.55×</td><td>1.35×</td><td>1.31×</td><td>2.11×</td><td>1.58×</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>2.06×</td><td>1.62×</td><td>1.61×</td><td>1.61×</td><td>1.72×</td></tr></table>

Table 2: $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ is strongest on average in every model family; $\mathrm { D C D } _ { \mathrm { N G R A M } }$ also stays above vanilla CD on average. Avg. averages datasets; highlights mark $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$

$\alpha = 0$ boundary. At $\alpha \ = \ 0 .$ CD reduces to the expert distribution. Table 3 positions DCD against autoregressive decoding and expert-only speculative decoding with the same EAGLE3 drafter. $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ remains above vanilla CD and autoregressive decoding at $1 . 7 5 { \times } / 1 . 4 5 { \times }$ on GSM8K/MMLU, while the expert-only row marks the expected upper bound when contrastive verification is removed.

70B-class validation. In the greedy-only 70B extension, Table 4 uses the same $\alpha \in \{ 0 . 1 , 0 . 5 \}$ operating points as the main comparison. DCD<sub>EAGLE3</sub> remains fastest on average, reaching 2.02× at $\alpha = 0 . 1$ and 1.96× at $\alpha = 0 . 5$ . Appendix F.4 gives the per-dataset results.

## 3.4 Task-Level Consistency Check

DCD changes only proposals, so the efficiency gains in Table 2 should not be interpreted as a new decoding objective. Verification still targets π<sub>CD</sub>; Appendix B.3 proves losslessness. Figure 4 shows the GSM8K greedy sweep for $\mathrm { L L A M A } { - } 3$ with $\gamma { = } 5$ and a matched $\mathrm { D C D } _ { \mathrm { N G R A M } }$ comparison; Appendix E.2 gives the full GSM8K/MMLU sweep under both $T { = } 0$ and $T { = } 1$

<table><tr><td>Setting</td><td>AR</td><td>Expert SD</td><td> $\mathbf { D C D } _ { \mathrm { N G R A M } }$ </td><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td></tr><tr><td>GSM8K</td><td>1.27×</td><td>2.55×</td><td>1.43×</td><td>1.75×</td></tr><tr><td>MMLU</td><td>1.31×</td><td>1.88×</td><td>1.34×</td><td>1.45×</td></tr></table>

Table 3: A $\alpha { = } 0 , \mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ remains faster than vanilla CD and AR on Llama-3. Expert SD uses the same EAGLE3 drafter.
<table><tr><td></td><td colspan="2">Average speedup over vanilla CD</td></tr><tr><td>Method</td><td> $\alpha = 0 . 1$ </td><td> $\alpha = 0 . 5$ </td></tr><tr><td>SCD</td><td>1.79×</td><td>1.69×</td></tr><tr><td>CoS</td><td>1.89×</td><td>1.75×</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>1.46×</td><td>1.44×</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>2.02×</td><td>1.96×</td></tr></table>

Table 4: $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ stays near $2 \times$ average speedup in the greedy 70B extension. Values average four datasets; full results are in Appendix F.4.

Consistency. Figure 4 shows that $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ tracks vanilla CD across α on GSM8K under greedy decoding, with error bars largely overlapping across the operating range. It indicates that DCD preserves the contrastive target while changing only the proposal mechanism.

## 3.5 Source of the DCD Speedup

Table 5 decomposes MMLU speedup using S ≈ $( L + 1 ) / T _ { \mathrm { i t e r } }$ and reports accepted length plus drafter, expert, and amateur latencies. The “Theo.” column evaluates this coarse serial model with the measured components; measured speed remains the deployment metric.

Serial proposal cost explains the speedup. Across the three MMLU settings, SCD/CoS spend 3.5 to 7.4ms per proposal step, while DCD<sub>EAGLE3</sub> uses about 0.6ms. On LLAMA-3, SCD reaches $L = 2 . 4 5$ and $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ reaches $L = 1 . 4 4$ , yet $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ is faster because cheaper proposals offset the shorter accepted length. Appendix D gives the multiplicative breakdown into acceptance and serial-cost factors.

## 3.6 Ablation Studies

We test the α-robustness intuition behind DCD.

Figure 5 normalizes L to 100% at α=0. This view separates robustness from absolute acceptance rate: SCD or CoS can start from a higher L at $\alpha { = } 0 .$ while the diagnostic tracks how quickly each proposal distribution drifts away from the contrastive target as the amateur penalty strengthens.

![](images/e9cc93c33b7d0affff67fa62976374c64b04f675343700b22c94a1190ffa7e1f.jpg)  
Figure 4: $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ tracks vanilla CD accuracy across α on Llama-3 GSM8K $( \gamma { = } 5 )$ . Error bars are 95% confidence intervals; the dashed line is AR.

$\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ degrades more gradually than amateur-coupled baselines in all four panels. On $\mathrm { L L A M A } { - } 3$ MMLU, the gap is moderate: at $\alpha { = } 0 . 5$ $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ retains 79.5% of its $\alpha { = } 0$ accepted length, while SCD and CoS retain about 74%. The separation becomes larger on QWEN3 MMLU, where SCD retains 62.9% and $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ retains 77.4%. This empirical pattern is consistent with the alignment intuition: as the contrastive target moves away from the amateur distribution, an expert-aligned lightweight drafter can lose acceptance more slowly than a proposal path tied to the amateur.

The GSM8K panels show the same trend but with different severity. LLAMA-3 GSM8K is relatively stable for all methods, yet $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ still remains closest to its $\alpha { = } 0$ acceptance level at α=0.5 (95.1% versus about 88% for SCD/CoS). QWEN3 GSM8K is harder: the amateur-coupled curves fall to 65.8%, while $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ stays at 81.9%. The cross-task consistency matters because the main speedups in Table 2 come from both proposal cost and acceptance; robustness in L prevents the serialcost advantage from being erased at larger α. Appendix Figure 11 and Table 14 give the $\gamma$ and throughput-grid details.

## 4 Related Work

Speculative Decoding. Speculative decoding accelerates generation with draft-thenverify (Leviathan et al., 2023; Chen et al., 2023). Recent work improves proposals through parallel heads, speculation trees and trajectories, self-speculation, distillation, and graph-structured acceptance (Cai et al., 2024; Miao et al., 2024; Fu et al., 2024; Elhoushi et al., 2024; Zhang et al., 2024; Zhou et al., 2024; Gong et al., 2024). The EAGLE series (Li et al., 2024a,b, 2025) drafts at the feature level. A contrastive target raises a distinct question: should the drafter imitate π<sub>CD</sub>, or should the amateur appear only in verification?

![](images/ff3a91ea55acacb4631ae3acdf45b1a0dd5f76a353051c287b5e660d979e9efb.jpg)  
Figure 5: $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ loses accepted length more mildly as contrastive strength increases. Relative mean accepted length is normalized to $\alpha { = } 0$ in each panel.

<table><tr><td>Method</td><td> $\pmb { L }$ </td><td> $t _ { e }$ </td><td> $t _ { p }$ </td><td> $t _ { q }$ </td><td>Speed</td><td>Theo.</td></tr><tr><td colspan="7">Llama-3 (8B-Ins / 1B-Ins)</td></tr><tr><td>CD</td><td>0.00</td><td></td><td>6.95</td><td>3.83</td><td>1.00x</td><td>1.00x</td></tr><tr><td>SCD</td><td>2.45</td><td>-</td><td>6.84</td><td>3.47</td><td>1.34x</td><td>1.48x</td></tr><tr><td>CoS</td><td>2.45</td><td></td><td>7.08</td><td>3.59</td><td>1.34x</td><td>1.48x</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>0.48</td><td>0.00</td><td>7.04</td><td>4.06</td><td>1.36x</td><td>1.43x</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>1.44</td><td>0.63</td><td>7.98</td><td>4.38</td><td>1.50x</td><td>1.70x</td></tr><tr><td colspan="7">Qwen3 (8B / 1.7B)</td></tr><tr><td>CD</td><td>0.00</td><td></td><td>8.47</td><td>6.52</td><td>1.00x</td><td>1.00x</td></tr><tr><td>SCD</td><td>2.49</td><td>一</td><td>9.06</td><td>6.53</td><td>1.14x</td><td>1.20x</td></tr><tr><td>CoS</td><td>2.49</td><td>一</td><td>9.15</td><td>6.63</td><td>1.17x</td><td>1.23x</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>0.41</td><td>0.00</td><td>9.07</td><td>7.23</td><td>1.22x</td><td>1.30x</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>1.48</td><td>0.61</td><td>10.15</td><td>7.93</td><td>1.61x</td><td>1.76x</td></tr><tr><td colspan="7">Llama-EFT (8B-Ins / 8B)</td></tr><tr><td>CD</td><td>0.00</td><td></td><td>7.19</td><td>7.25</td><td>1.00x</td><td>1.00x</td></tr><tr><td>SCD</td><td>2.64</td><td>、</td><td>6.77</td><td>6.47</td><td>1.20x</td><td>1.28x</td></tr><tr><td>CoS</td><td>2.64</td><td></td><td>7.67</td><td>7.38</td><td>1.11x</td><td>1.17x</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>0.49</td><td>0.00</td><td>6.98</td><td>7.12</td><td>1.45x</td><td>1.52x</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>1.46</td><td>0.60</td><td>7.42</td><td>7.50</td><td>1.79x</td><td>1.98x</td></tr></table>

Table 5: $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ beats SCD/CoS by lowering MMLU serial proposal latency despite smaller L. Results use ${ T { = } 0 , \alpha { = } 0 . 1 , \gamma { = } 5 ; { t _ { e } } , { t _ { p } } }$ , and $t _ { q }$ are per-step ms.

Contrastive Decoding. Contrastive Decoding (CD) (Li et al., 2023) amplifies the expert–amateur gap. Later work applies this idea to layer contrast, contextual understanding, noisy retrieval, visionlanguage hallucination mitigation, and tuning-byproxy (Chuang et al., 2024; Zhao et al., 2024; Kim et al., 2024; Leng et al., 2024; Wang et al., 2024;

Mitchell et al., 2024; Liu et al., 2024). CD still requires two model passes per token.

Accelerating Contrastive Decoding. Existing CD acceleration methods mostly retain the amateur in proposal generation. SCD (Yuan et al., 2024) and CoS (Fu et al., 2025) use it as a drafter and retain its serial cost. Section 3.1 finds no consistent benefit from adding the contrastive signal to a constrained lightweight proposer, motivating DCD’s expert-aligned path.

## 5 Conclusion

Contrastive Decoding (CD) improves generation but adds an expensive amateur pass. We ask whether speculative CD should use the contrastive signal during drafting or only verification. Matched Cross-α and Approximate Dual-Drafter diagnostics show that contrastive-aware lightweight drafting does not consistently outperform expert-aligned drafting: the correction is usually smaller than drafter error, which reconstruction can amplify. Decoupled Contrastive Decoding (DCD) therefore uses an expert-aligned lightweight proposer and reserves the amateur for unchanged CD verification, preserving vanilla CD’s output distribution. Across the main 8B settings, EAGLE3-based DCD yields 1.65–1.95× average greedy speedups and cuts MMLU proposal-path latency by 5–12× relative to amateur-coupled proposal paths.

## Limitations

Our evaluation follows the public benchmark mix and fixed decoding budgets used in the main study. It does not exhaust every workload shape: very long contexts, multilingual or domain-specific prompts, and adaptive choices of α, draft length, or the CD plausibility threshold may shift the best operating point.

## References

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. 2024. MEDUSA: Simple LLM inference acceleration framework with multiple decoding heads. In Proceedings ofthe 41st International Conference on Machine Learning, pages 5209–5235.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, and 39 others. 2021. Evaluating large language models trained on code. Preprint, arXiv:2107.03374.

Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James R. Glass, and Pengcheng He. 2024. DoLa: Decoding by contrasting layers improves factuality in large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed A Aly, Beidi Chen, and Carole-Jean Wu. 2024. LayerSkip: Enabling early exit inference and self-speculative decoding. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 12622–12642. Association for Computational Linguistics.

Jiale Fu, Yuchu Jiang, Junkai Chen, Jiaming Fan, Xin Geng, and Xu Yang. 2025. Fast large language

model collaborative decoding via speculation. In Forty-second International Conference on Machine Learning.

Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. 2024. Break the sequential dependency of LLM inference using lookahead decoding. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

Zhuocheng Gong, Jiahao Liu, Ziyue Wang, Pengfei Wu, Jingang Wang, Xunliang Cai, Dongyan Zhao, and Rui Yan. 2024. Graph-structured speculative decoding. In Findings ofthe Associationfor Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 11404– 11415. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. Proceedings ofthe International Conference on Learning Representations (ICLR).

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Trans. Inf. Syst., 43(2):42:1– 42:55.

Youna Kim, Hyuhng Joon Kim, Cheonbok Park, Choonghyun Park, Hyunsoo Cho, Junyeob Kim, Kang Min Yoo, Sang-goo Lee, and Taeuk Kim. 2024. Adaptive contrastive decoding in retrieval-augmented generation for handling noisy contexts. In Findings of the Association for Computational Linguistics: EMNLP 2024, Miami, Florida, USA, November 12- 16, 2024, pages 2421–2431. Association for Computational Linguistics.

Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. 2024. Mitigating object hallucinations in large visionlanguage models through visual contrastive decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13872–13882.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pages 19274–19286. PMLR.

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori B Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2023. Contrastive decoding: Open-ended text generation as optimization. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12286–12312.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024a. EAGLE-2: Faster inference of language models with dynamic draft trees. In Empirical Methods in Natural Language Processing.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024b. EAGLE: Speculative sampling requires rethinking feature uncertainty. In International Conference on Machine Learning.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2025. EAGLE-3: Scaling up inference acceleration of large language models via training-time test. In Annual Conference on Neural Information Processing Systems.

Alisa Liu, Xiaochuang Han, Yizhong Wang, Yulia Tsvetkov, Yejin Choi, and Noah A. Smith. 2024. Tuning language models by proxy. CoRR, abs/2401.08565.

Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. 2024. SpecInfer: Accelerating large language model serving with tree-based speculative inference and verification. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, pages 932–949.

Eric Mitchell, Rafael Rafailov, Archit Sharma, Chelsea Finn, and Christopher D. Manning. 2024. An emulator for fine-tuning large language models using small language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Sean O’Brien and Mike Lewis. 2023. Contrastive decoding improves reasoning in large language models. CoRR, abs/2309.09117.

Aske Plaat, Annie Wong, Suzan Verberne, Joost Broekens, Niki van Stein, and Thomas Bäck. 2024. Reasoning with large language models, a survey. CoRR, abs/2407.11511.

Abigail See, Peter J. Liu, and Christopher D. Manning. 2017. Get to the point: Summarization with pointergenerator networks. In Proceedings ofthe 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1073– 1083, Vancouver, Canada. Association for Computational Linguistics.

Xintong Wang, Jingheng Pan, Liang Ding, and Chris Biemann. 2024. Mitigating hallucinations in large vision-language models with instruction contrastive decoding. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 15840–15853. Association for Computational Linguistics.

Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, and Zhifang Sui. 2024. Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 7655–7671, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.

Hongyi Yuan, Keming Lu, Fei Huang, Zheng Yuan, and Chang Zhou. 2024. Speculative contrastive decoding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 56–64.

Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra. 2024. Draft& verify: Lossless large language model acceleration via self-speculative decoding. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 11263–11282. Association for Computational Linguistics.

Zheng Zhao, Emilio Monti, Jens Lehmann, and Haytham Assem. 2024. Enhancing contextual understanding in large language models through contrastive decoding. In Proceedings ofthe 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024, pages 4225–4237. Association for Computational Linguistics.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark W. Barrett, and Ying Sheng. 2024. SGLang: Efficient execution of structured language model programs. Advances in Neural Information Processing Systems, 37:62557–62583.

Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-François Kagy, and Rishabh Agarwal. 2024. DistillSpec: Improving speculative decoding via knowledge distillation. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

## A Proposal-Side Diagnostics

## A.1 Analysis of Contrastive-Aligned Drafting Strategies

This subsection reports supplementary measurements for the contrastive-aligned drafting results in Section 3.1. It restates the matched-control setup, then reports per-position reconstruction checks and bucketed diagnostics over model groups, datasets, and contrastive strengths. The matched control remains the two-model-pair, two-dataset study in Section 3.1; the later views are supplementary diagnostics.

Measurement Methodology. We use two related measurement setups. For the matched-control KL and reconstruction checks referenced in the main text, we run per-position forward passes on 200 GSM8K examples for the LLAMA-3 and QWEN3 model pairs under greedy decoding $( T = 0 )$ with draft length $\gamma = 5$ . We record full-vocabulary logits from the expert model $\mathcal { M } _ { p }$ , the amateur model $\mathcal { M } _ { q } .$ , and the expert-aligned drafter E instantiated with EAGLE3. We compute $D _ { K L } ( \pi _ { p } \| \pi _ { C D } )$ by constructing $\pi _ { C D }$ from Eq. 1 at each position, and we compute $D _ { K L } ( \pi _ { e } \| \pi _ { p } )$ directly from drafter and expert logits.

For the supplementary bucketed regime analysis, we aggregate configurations across three model groups (LLAMA-3, LLAMA-EFT, and QWEN3), two datasets (GSM8K and MMLU), and four contrastive strengths $( \alpha \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 1 . 0 \} )$ . We use the same shared evaluation convention as Section 3: 200 evaluation samples per dataset, greedy decoding $( T ~ = ~ 0 )$ , and draft length $\gamma ~ = ~ 5$ Here h denotes the current decoding history, and $x ^ { \star } = \arg \operatorname* { m a x } _ { x } \pi _ { C D } ( x \mid h )$ denotes the CD top-1 token at that position. The per-position contrastivesignal statistic is the expert–amateur absolute logratio at that token, $\begin{array} { r } { \left| \log \frac { \overline { { \pi } } _ { p } ( x ^ { \star } | h ) } { \pi _ { q } ( x ^ { \star } | h ) } \right| } \end{array}$ . Table 7 is countweighted: we sum the per-bucket position counts over all configurations and divide by the total count. The reported 81.1% is therefore the sum of the first three buckets, [0, 0.25), [0.25, 0.5), and [0.5, 1).

Error Amplification in the Approximate Dual-Drafter. A simple error decomposition explains why Path 2, the Approximate Dual-Drafter, performs poorly. Let $\begin{array} { r l r } { \hat { \pi } _ { C D } ( x ) } & { { } \propto } & { \pi _ { e } ( x ) ^ { 1 . } } \end{array}$ +α $\pi _ { f } ( x ) ^ { - \alpha }$ be the combined distribution, where $\pi _ { f }$ is the amateur-aligned EAGLE head approximating $\pi _ { q } .$ Define the log approximation errors $\epsilon _ { p } ( x ) = \log ( \pi _ { e } ( x ) / \pi _ { p } ( x ) )$ and $\epsilon _ { q } ( x ) =$ log $\mathinner { \ ; { \left( \pi _ { f } ( x ) / \pi _ { q } ( x ) \right) } }$

Lemma A.1 (Reconstruction Error Decomposition). The reconstructed distribution πˆ<sub>CD</sub> is an exponential tilt of π<sub>CD</sub>:

$$
\begin{array} { c } { { \hat { \pi } _ { C D } ( x ) = \displaystyle \frac { \pi _ { C D } ( x ) \cdot e ^ { \Delta ( x ) } } { Z _ { \Delta } } , } } \\ { { Z _ { \Delta } = \mathbb { E } _ { \pi _ { C D } } [ e ^ { \Delta ( X ) } ] , } } \end{array}
$$

where the amplified combined error is $\Delta ( x ) =$ $( 1 + \alpha ) \epsilon _ { p } ( x ) - \alpha \epsilon _ { q } ( x )$

Proof. Direct expansion:

$$
\begin{array} { c } { { \pi _ { e } ( x ) ^ { 1 + \alpha } \cdot \pi _ { f } ( x ) ^ { - \alpha } = \left[ \pi _ { p } ( x ) e ^ { \epsilon _ { p } ( x ) } \right] ^ { 1 + \alpha } } } \\ { { \cdot \left[ \pi _ { q } ( x ) e ^ { \epsilon _ { q } ( x ) } \right] ^ { - \alpha } } } \\ { { = \pi _ { p } ( x ) ^ { 1 + \alpha } \pi _ { q } ( x ) ^ { - \alpha } } } \\ { { \cdot e ^ { ( 1 + \alpha ) \epsilon _ { p } ( x ) - \alpha \epsilon _ { q } ( x ) } } } \\ { { = Z _ { C D } \pi _ { C D } ( x ) \cdot e ^ { \Delta ( x ) } . } } \end{array}
$$

Normalizing yields the lemma.

The lemma makes the amplification mechanism explicit: the expert EAGLE log-error $\epsilon _ { p }$ is scaled by $( 1 + \alpha )$ and the amateur EAGLE error $\epsilon _ { q }$ by $\alpha .$ Even when the two heads have similar approximation quality $( D _ { K L } ( \pi _ { e } \| \pi _ { p } ) ~ \approx ~ 2 . 6 $ vs. $D _ { K L } ( \pi _ { f } \Vert \pi _ { q } )$ ≈ 2.4 on Llama-3), the combined error ∆ grows before normalization. Consequently, $D _ { K L } \big ( \hat { \pi } _ { C D } \| \pi _ { C D } \big )$ becomes sensitive to the error distribution.

Per-Position Verification. We also measure the fraction of autoregressive positions at which the Approximate Dual-Drafter reconstruction $\hat { \pi } _ { C D }$ is closer to $\pi _ { C D }$ than the expert-aligned draft $\pi _ { e } .$ , that is, where $D _ { K L } ( \hat { \pi } _ { C D } \| \pi _ { C D } ) ~ < ~ D _ { K L } ( \pi _ { e } \| \pi _ { C D } )$ On Llama-3 (GSM8K), this holds for 39.7% of positions at $\alpha ~ = ~ 0 . 1$ and 33.1% at $\alpha \ = \ 0 . 5 ;$ Qwen3 shows 39.2% and 31.9%, respectively. This position-level check is consistent with the underperformance of Approximate Dual-Drafter in the tested regime.

Table 6 reports the corresponding aggregate matched-control results over accepted length, draftto-target KL, and Top-1 Match.

Supplementary Aggregated Regime Analysis Beyond the Matched Control. Beyond the matched control above, we also aggregate configurations spanning three model groups (LLAMA-3, LLAMA-EFT, and QWEN3), two datasets (GSM8K and MMLU), and four contrastive strengths $( \alpha ~ \in ~ \{ 0 . 1 , 0 . 3 , 0 . 5 , 1 . 0 \} )$ . For each configuration, we record the Top-1 gap ∆Top-1 = Top-1(contrastive-aligned) − Top-1(expertaligned), where Top-1 denotes the fraction of positions whose argmax matches arg max<sub>x</sub> π<sub>CD</sub>(x | $h )$ , across bucketed views of signal, $D _ { K L } ( \pi _ { e } \| \pi _ { p } )$ and $D _ { K L } ( \pi _ { f } \Vert \pi _ { q } )$ . The main text now summarizes the signal distribution, the signal × $D _ { K L } ( \pi _ { e } \| \pi _ { p } )$ effect view, and the marginal $\mathbf { K L } ( \pi _ { e } \| \pi _ { p } )$ distribution; this appendix adds the complementary views from that broader diagnostic.

<table><tr><td rowspan="2">α</td><td rowspan="2">Method</td><td colspan="2">Mean Accepted Length L</td><td rowspan="2">Avg.  $D _ { K L } ( \cdot \| \pi _ { C D } )$ </td><td rowspan="2">Avg. Top-1 Match (%)</td></tr><tr><td>GSM8K</td><td>MMLU</td></tr><tr><td>Llama-3 (8B-Ins / 1B-Ins)</td></tr><tr><td>Expert-aligned</td><td>1.966</td><td>1.476</td><td>2.98</td><td>67.7</td></tr><tr><td>0.1</td><td>Approx. Dual</td><td>1.945 1.458</td><td>3.13</td><td>66.7 66.1</td></tr><tr><td rowspan="2">0.3</td><td>Expert-aligned</td><td>1.932</td><td>1.424</td><td>3.26</td></tr><tr><td>Approx. Dual</td><td>1.853</td><td>1.357</td><td>3.84 62.9</td></tr><tr><td>0.5</td><td>Expert-aligned</td><td>1.863 1.355</td><td>3.54</td><td>64.4</td></tr><tr><td>Approx. Dual</td><td>1.711</td><td>1.241</td><td>4.74</td><td>58.6</td></tr><tr><td colspan="7">Qwen3 (8B / 1.7B)</td></tr><tr><td rowspan="2">0.1</td><td>Expert-aligned</td><td>1.736</td><td>1.122</td><td>6.45</td><td>60.6</td></tr><tr><td>Approx. Dual</td><td>1.717</td><td>1.104</td><td>6.69</td><td>59.7</td></tr><tr><td rowspan="2">0.3</td><td>Expert-aligned</td><td>1.656</td><td>1.027</td><td>6.55</td><td>58.1</td></tr><tr><td>Approx. Dual</td><td>1.593</td><td>0.973</td><td>7.39</td><td>55.2</td></tr><tr><td rowspan="2">0.5</td><td>Expert-aligned</td><td>1.544</td><td>0.904</td><td>6.69</td><td>54.7</td></tr><tr><td>Approx. Dual</td><td>1.420</td><td>0.820</td><td>8.31</td><td>49.3</td></tr></table>

Table 6: Approximate dual-drafting vs. expert-aligned $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ . GSM8K/MMLU columns report datasetspecific mean accepted length L. The last two columns average over GSM8K and MMLU: $D _ { K L } ( \cdot \| \pi _ { C D } )$ is the draft-to-target $\mathrm { K L , }$ and Top-1 Match is the argmax match rate with π . The CD reconstruction route fails on all three diagnostics: it has lower L, higher draft-totarget $\mathrm { K L , }$ and lower Top-1 Match, with larger gaps as α increases.

Table 7 gives the detailed count-weighted signalmass breakdown behind the main-text distribution panel. In total, 81.1% of positions fall below signal strength 1.0, while 18.9% lie at or above 1.0.

Main-text Figure 2 summarizes the countweighted signal distribution, the marginal KL ${ \mathrm { \Omega } } ( \pi _ { e } \| { \pi } _ { p } )$ distribution, and the signal × $D _ { K L } ( \pi _ { e } \| \pi _ { p } )$ mechanism view. The remaining appendix figures add complementary aggregated views.

Figure 6 provides the complementary configuration-level view over $\operatorname { K L } ( \pi _ { f } \parallel \pi _ { q } )$ buckets. Along this axis, the values vary more by configuration and do not form a stable monotonic pattern.

Figure 7 provides a compact signal-centric aggregation over mutually exclusive signal buckets and shows that the low-signal failure pattern persists in most settings.

<table><tr><td>Signal bucket</td><td>Share</td><td>Cumulative</td></tr><tr><td>[0, 0.25)</td><td>64.4%</td><td>64.4%</td></tr><tr><td>[0.25,0.5)</td><td>8.4%</td><td>72.8%</td></tr><tr><td>[0.5, 1)</td><td>8.3%</td><td>81.1%</td></tr><tr><td>[1, 2)</td><td>7.0%</td><td>88.1%</td></tr><tr><td>[2, 4)</td><td>5.2%</td><td>93.3%</td></tr><tr><td>[4, 8)</td><td>3.6%</td><td>96.9%</td></tr><tr><td> $\geq 8$ </td><td>3.1%</td><td>100.0%</td></tr><tr><td> $< 1 . 0$ </td><td>81.1%</td><td>一</td></tr><tr><td> $\geq 1 . 0$ </td><td>18.9%</td><td>一</td></tr></table>

Table 7: Count-weighted share of autoregressive positions by contrastive-signal bucket, aggregated over 24 configurations spanning three model groups, two datasets, and four contrastive strengths.

Figure 8 breaks this pattern down by model group and dataset. Larger α usually pushes the low-signal buckets further negative, while some higher-signal buckets improve when the contrastive signal is informative enough.

Across Figures 2, 6, 7, and 8, low-signal buckets remain mostly negative, the dependence on expertside proposal error is strong, and the dependence on amateur-side reconstruction error is noisier.

Why Contrastive-Aligned Training Also Underperforms in Our Setting (Path 1). For Path 1, there is no inference-time reconstruction step, but the same scale mismatch remains: the lightweight EAGLE head must learn both the dominant expert distribution $\pi _ { p }$ and the much smaller contrastive signal $\Delta \propto \pi _ { q } ^ { - \alpha }$ . This increases the modeling burden without reducing baseline proposal error. Within our CD-Aligned-EAGLE family, Table 8 therefore still favors expert-aligned training (trainα=0) as the strongest default.

## A.2 Cross-alpha Experiment Details

Table 1 in the main text summarizes the withinfamily controlled single-run ablation. Table 8 reports the full cross-α matrix for the Contrastive-Aligned Dual-Input Drafter in Section 3.1 (Path 1), where rows vary train-α and columns vary infer-α. All numeric variants share the same dual-input architecture and ShareGPT training setup; only the target distribution differs. We also include the Official checkpoint from our main experiments, evaluated under the same inference protocol, as an external reference.

<table><tr><td>Train-α Infer-α=0.0</td><td>Infer-α=0.1</td><td>Infer-α=0.3</td><td>Infer-α=0.5</td></tr><tr><td colspan="4">Llama-3 (8B-Ins / 1B-Ins) —GSM8K</td></tr><tr><td>OFFICIAL CHECKPOINT 1.555</td><td>1.537</td><td>1.546</td><td>1.476</td></tr><tr><td>0.0</td><td>2.002 1.968</td><td>1.956</td><td>1.880</td></tr><tr><td>0.1</td><td>2.008 1.988</td><td>1.977</td><td>1.892</td></tr><tr><td>0.3</td><td>1.995 1.964</td><td>1.963</td><td>1.881</td></tr><tr><td>0.5 1.936</td><td>1.915</td><td>1.896</td><td>1.833</td></tr><tr><td colspan="4">Llama-3 (8B-Ins / 1B-Ins) - MMLU</td></tr><tr><td>OFFICIAL CHECKPOINT 1.523</td><td>1.428</td><td>1.344</td><td>1.214</td></tr><tr><td>0.0</td><td>1.655 1.596</td><td>1.508</td><td>1.404</td></tr><tr><td>0.1</td><td>1.663 1.576</td><td>1.503</td><td>1.370</td></tr><tr><td>0.3</td><td>1.623 1.554</td><td>1.469</td><td>1.320</td></tr><tr><td>0.5 1.618</td><td>1.552</td><td>1.466</td><td>1.328</td></tr><tr><td colspan="4">Qwen3 (8B / 1.7B) )—GSM8K</td></tr><tr><td>OFFICIAL CHECKPOINT 2.083</td><td>2.061</td><td>1.983</td><td>1.842</td></tr><tr><td>0.0 1.732</td><td>1.691</td><td>1.602</td><td>1.477</td></tr><tr><td>0.1</td><td>1.714 1.664</td><td>1.583</td><td>1.477</td></tr><tr><td>0.3</td><td>1.715 1.678</td><td>1.575</td><td>1.474</td></tr><tr><td>0.5 1.691</td><td>1.665</td><td>1.584</td><td>1.487</td></tr><tr><td colspan="4">Qwen3 (8B / 1.7B) —MMLU</td></tr><tr><td>OFFICIAL CHECKPOINT 1.656</td><td>1.650</td><td>1.552</td><td>1.386</td></tr><tr><td>0.0 1.438</td><td>1.426</td><td>1.317</td><td>1.189</td></tr><tr><td>0.1 1.411</td><td>1.404</td><td>1.294</td><td>1.159</td></tr><tr><td>0.3 1.418</td><td>1.411</td><td>1.317</td><td>1.176</td></tr><tr><td>0.5</td><td>1.392 1.380</td><td>1.284</td><td>1.156</td></tr></table>

Table 8: Full cross-α matrix for the Contrastive-Aligned Dual-Input Drafter. Numeric train-α rows report our CD-Aligned-EAGLE family: rows vary the training target, columns vary infer- $- \alpha ,$ and cells report Mean Accepted Length (L). The OFFICIAL CHECKPOINT row reports the publicly released model used in our main experiments, evaluated under the same inference protocol, as an external reference. Train-α=0.0 is our expert-aligned baseline. Highlighted rows mark this baseline, and bold marks the best value in each infer-α column over all displayed rows.

## A.3 Optimistic Offline Diagnostic with a Stronger Full-LM Proposer

This subsection extends the diagnosis in Section 3.1 to a stronger same-family proposer. It tests whether the mechanism in Figure 2 disappears once the proposal side is replaced by a substantially stronger full LM.

Setup. We keep the expert and amateur fixed as LLAMA-3.1-8B-INSTRUCT and LLAMA-3.2-1B-INSTRUCT, and replace the lightweight proposal proxy with LLAMA-3.2-3B-INSTRUCT. Let $\pi _ { r }$ denote the 3B proposer distribution. For each shared response history $h ,$ we compare the true contrastive target

$$
\pi _ { C D } ( x \mid h ) \propto \pi _ { p } ( x \mid h ) ^ { 1 + \alpha } \pi _ { q } ( x \mid h ) ^ { - \alpha }
$$

with the stronger-proposer approximation

$$
\hat { \pi } _ { C D } ^ { r } ( x \mid h ) \propto \pi _ { r } ( x \mid h ) ^ { 1 + \alpha } \pi _ { q } ( x \mid h ) ^ { - \alpha } .
$$

We evaluate GSM8K and MMLU on 200 aligned examples each with $\alpha \in \{ 0 . 1 , 0 . 5 \}$ . Reusing the true amateur logits and computing all statistics on shared response histories gives the proposer an optimistic offline diagnostic; online rollout performance is outside this check.

Figure 9 reports the same three-part diagnostic as Figure 2. Under this optimistic proposer replacement, favorable gains remain concentrated in higher-signal, lower-error regions, while low-signal or high-proposer-error buckets remain unfavorable. The stronger proposer therefore enlarges the favorable pocket but does not overturn the conclusion that proposal-side approximation error limits contrastive-aligned drafting.

## B Theory for Expert-Aligned Drafting in Contrastive Acceleration

This appendix provides the robustness analysis summarized in Section 2.3. It characterizes the expert-aligned lightweight-drafter regime used by $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ and connects the analysis to the observed relative degradation trends; the empirical comparisons remain the primary evidence.

## B.1 Distribution Alignment and Robustness Analysis

We analyze how draft distributions align with the contrastive decoding target distribution $\pi _ { C D }$ and why expert-aligned drafting is more robust to α under the stated alignment condition. This analysis supports the empirical finding that mean accepted length decays more gradually in the expert-aligned $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ instantiation as contrastive strength increases.

KL Divergence Decomposition. Let $\pi _ { p } , \pi _ { q } ,$ and $\pi _ { e }$ denote the expert, amateur, and lightweightdrafter distributions, respectively. The target distribution is defined as $\begin{array} { r l } { \pi _ { C D } ( x ) } & { { } = } \end{array}$ $\scriptstyle { \frac { 1 } { Z _ { C D } } } \pi _ { p } ( x ) ^ { 1 + \alpha } \pi _ { q } ( x ) ^ { - \alpha }$

Lemma B.1 (KL Divergence to Target).

$$
\begin{array} { r l } & { D _ { K L } ( \pi _ { p } \| \pi _ { C D } ) = \log Z _ { C D } - \alpha D _ { K L } ( \pi _ { p } \| \pi _ { q } ) , } \\ & { D _ { K L } ( \pi _ { q } \| \pi _ { C D } ) = \log Z _ { C D } } \\ & { \qquad + \left( 1 + \alpha \right) D _ { K L } ( \pi _ { q } \| \pi _ { p } ) , } \\ & { D _ { K L } ( \pi _ { e } \| \pi _ { C D } ) = \log Z _ { C D } } \\ & { \qquad + \left( 1 + \alpha \right) D _ { K L } ( \pi _ { e } \| \pi _ { p } ) } \\ & { \qquad - \alpha D _ { K L } ( \pi _ { e } \| \pi _ { q } ) . } \end{array}
$$

Proof. Substituting the definition of $\pi _ { C D }$ into $D _ { K L } ( \cdot \| \pi _ { C D } )$ and expanding the logarithms gives the result. □

Static Analysis. We first examine the absolute distance between each draft distribution and the target $\pi _ { C D }$ at a fixed α.

From Lemma B.1, the KL divergence gap between the amateur model and the drafter relative to the target is:

$$
\begin{array} { r l } & { D _ { K L } ( \pi _ { q } | | \pi _ { C D } ) - D _ { K L } ( \pi _ { e } | | \pi _ { C D } ) } \\ & { \quad = ( 1 + \alpha ) [ D _ { K L } ( \pi _ { q } | | \pi _ { p } ) - D _ { K L } ( \pi _ { e } | | \pi _ { p } ) ] } \\ & { \quad \quad + \alpha D _ { K L } ( \pi _ { e } | | \pi _ { q } ) . } \end{array}\tag{2}
$$

This identity follows directly from Lemma B.1, where the log $Z _ { C D }$ terms cancel.

Remark (Static Capacity Gap). In practice, if the absolute distance from the drafter to the target is larger than the corresponding distance from the amateur $( D _ { K L } ( \pi _ { e } \| \pi _ { C D } ) ~ > ~ D _ { K L } ( \pi _ { q } \| \pi _ { C D } ) )$ , it implies:

$$
\begin{array} { l } { { \displaystyle { \cal D } _ { K L } ( \pi _ { e } \| \pi _ { p } ) > { \cal D } _ { K L } ( \pi _ { q } \| \pi _ { p } ) } } \\ { { \displaystyle ~ + \frac { \alpha } { 1 + \alpha } { \cal D } _ { K L } ( \pi _ { e } \| \pi _ { q } ) . } } \end{array}
$$

This inequality quantifies the static capacity gap. Although the lightweight drafter is trained to mimic the expert, its approximation error $D _ { K L } ( \pi _ { e } \| \pi _ { p } )$ can still exceed the amateur’s baseline error by a margin large enough to offset the alignment term. This interpretation matches our acceptance-rate comparisons, in which the lightweight drafter can start from a lower acceptance baseline than the amateur model.

Dynamic Analysis: Robustness to Contrastive Penalty. Although $\pi _ { e }$ may be farther from the target in absolute terms, its response to increasing α can still be more favorable under the alignment condition below.

To keep the analysis tied to a meaningful draft model, we use the following alignment condition.

Definition B.2 (Effective Draft Alignment). A draft distribution $\pi _ { e }$ is effectively aligned $i f \colon$

$$
\mathbb { E } _ { \boldsymbol { x } \sim \pi _ { e } } \left[ \log \frac { \pi _ { p } ( \boldsymbol { x } ) } { \pi _ { q } ( \boldsymbol { x } ) } \right] > \mathbb { E } _ { \boldsymbol { x } \sim \pi _ { q } } \left[ \log \frac { \pi _ { p } ( \boldsymbol { x } ) } { \pi _ { q } ( \boldsymbol { x } ) } \right] .
$$

This condition is equivalent to

$$
\sum _ { x } ( \pi _ { e } ( x ) - \pi _ { q } ( x ) ) \log { \frac { \pi _ { p } ( x ) } { \pi _ { q } ( x ) } } > 0 .\tag{3}
$$

Remark. The condition is practical. Because $\pi _ { e }$ is trained to approximate $\pi _ { p } ,$ it tends to move probability mass toward tokens on which the expert has a larger log-likelihood advantage over the amateur. Condition (3) only requires the drafter to separate expert-preferred tokens from the rest at least as well as the amateur does.

Theorem B.3 (Slope Comparison under Effective Draft Alignment). Assuming $\pi _ { e }$ satisfies the Effective Draft Alignment condition, the distancefrom the amateur distribution $\pi _ { q }$ to the target distribution π<sub>CD</sub> growsfaster than the distancefrom the lightweight drafter $\pi _ { e }$ as α increases:

$$
\frac { \partial } { \partial \alpha } D _ { K L } ( \pi _ { q } | | \pi _ { C D } ) > \frac { \partial } { \partial \alpha } D _ { K L } ( \pi _ { e } | | \pi _ { C D } ) .
$$

Proof of Theorem B.3. Empirically, SCD often has a higher acceptance rate at $\alpha = 0 ;$ , whereas expert-aligned drafting typically deteriorates more slowly as α increases. We explain this trend by tracking how the distance to the target changes with α.

Proof. The target distribution is:

$$
\pi _ { C D } ( x ) = \frac { 1 } { Z _ { C D } } \pi _ { p } ( x ) ^ { 1 + \alpha } \pi _ { q } ( x ) ^ { - \alpha } .
$$

Substituting its log-form into the KL divergence formula $D _ { K L } ( \cdot \| \pi _ { C D } )$ and differentiating with respect to α:

For the amateur model $\mathcal { M } _ { q }$ :

$$
\begin{array} { l } { { \mathrm { S l o p e } _ { q } = \displaystyle \frac { \partial } { \partial \alpha } D _ { K L } ( \pi _ { q } | | \pi _ { C D } ) } } \\ { { ~ = \displaystyle \frac { \partial } { \partial \alpha } \mathbb { E } _ { x \sim \pi _ { q } } [ \mathrm { l o g } \pi _ { q } ( x ) } } \\ { { ~ - ~ ( ( 1 + \alpha ) \mathrm { l o g } \pi _ { p } ( x ) } } \\ { { ~ - ~ \alpha \mathrm { l o g } \pi _ { q } ( x ) - \mathrm { l o g } Z _ { C D } ) ] } } \\ { { ~ = \mathbb { E } _ { x \sim \pi _ { q } } [ \mathrm { l o g } \pi _ { q } ( x ) - \mathrm { l o g } \pi _ { p } ( x ) ] } } \\ { { ~ + ~ \frac { \partial \mathrm { l o g } Z _ { C D } } { \partial \alpha } . } } \end{array}
$$

For the drafter $\pi _ { e } { : }$

$$
\begin{array} { l } { { \displaystyle \mathrm { S l o p e } _ { e } = \frac { \partial } { \partial \alpha } D _ { K L } ( \pi _ { e } | | \pi _ { C D } ) } } \\ { ~ = \frac { \partial } { \partial \alpha } \mathbb { E } _ { x \sim \pi _ { e } } [ \log \pi _ { e } ( x ) } \\ { ~ - ~ ( ( 1 + \alpha ) \log \pi _ { p } ( x ) } \\ { ~ - ~ \alpha \log \pi _ { q } ( x ) - \log Z _ { C D } ) ] }  \\ { { \displaystyle = \mathbb { E } _ { x \sim \pi _ { e } } [ \log \pi _ { q } ( x ) - \log \pi _ { p } ( x ) ] } } \\ { ~ + ~ \frac { \partial \log Z _ { C D } } { \partial \alpha } . } \end{array}
$$

Considering the slope difference $\Delta _ { \mathrm { s l o p e } } .$ , the normalization term $\frac { \partial \log { \bar { Z } } _ { C D } } { \partial \alpha }$ cancels out:

$$
\begin{array} { r l } { \Delta _ { \mathrm { s h y e } } = S \log \mathrm { e } _ { q } - S \log \mathrm { e } _ { e } } & { } \\ & { = \mathbb { E } _ { z \sim x _ { \mathrm { e f f } } } \left[ \log z _ { q } ( x ) - \log \pi _ { p } ( x ) \right] } \\ & { \phantom { = } - \mathbb { E } _ { \mathrm { x e r g } } [ \log z _ { q } ( x ) - \log \pi _ { p } ( x ) ] } \\ & { = \sum _ { x } \pi _ { q } ( x ) \log \frac { \pi _ { p } ( x ) } { \pi _ { p } ( x ) } } \\ & { \phantom { = } - \displaystyle \sum _ { x } \pi _ { \epsilon } ( x ) \log \frac { \pi _ { q } ( x ) } { \pi _ { p } ( x ) } } \\ & { = \displaystyle \sum _ { x } \left( \pi _ { q } ( x ) - \pi _ { \mathrm { e f f } } ( x ) \right) \log \frac { \pi _ { q } ( x ) } { \pi _ { p } ( x ) } } \\ & { = \displaystyle \sum _ { x } \left( \pi _ { q } ( x ) - \pi _ { \mathrm { e f f } } ( x ) \right) \log \frac { \pi _ { q } ( x ) } { \pi _ { p } ( x ) } } \\ & { = \displaystyle \sum _ { x } \left( \pi _ { \epsilon } ( x ) - \pi _ { q } ( x ) \right) \log \frac { \pi _ { p } ( x ) } { \pi _ { q } ( x ) } . } \end{array}\tag{4}
$$

By Definition B.2, the expression in (4) is strictly positive. Thus $\Delta _ { \mathrm { s l o p e } } > 0$

Conclusion and Discussion. As α increases, the amateur model $\mathcal { M } _ { q }$ used in SCD diverges from the contrastive target faster than the lightweight drafter under Effective Draft Alignment. This explains the acceptance-side robustness of expert-aligned drafting in DCD. The ablations in Section 3.6 show the empirical counterpart: the relative mean accepted length of DCD declines more slowly than that of SCD as α increases.

Synthesis. This analysis clarifies the empirical pattern: although the lightweight drafter may begin farther from the target because of its capacity limits (Eq. 2), it has a more favorable dynamic slope under Theorem B.3. This split between static gap and dynamic robustness explains the acceptance-side degradation trends observed in the experiments.

## B.2 Efficiency Advantage Condition

Let $\tau _ { m }$ and $T _ { m }$ denote the mean output tokens and serial time per iteration for method m. Under the throughput surrogate $S _ { m } = \tau _ { m } / T _ { m }$ , DCD is faster than an amateur-coupled baseline b if and only if

$$
\frac { \tau _ { \mathrm { D C D } } } { \tau _ { b } } > \frac { T _ { \mathrm { D C D } } } { T _ { b } } .\tag{5}
$$

Thus, a lower accepted-token yield can be offset by a sufficiently large reduction in serial cost. Appendix D expands $T _ { m }$ for DCD, SCD, and CoS and reports the measured decomposition used in Section 3.5.

## B.3 Proof of Lossless Property

DCD inherits the lossless guarantee of speculative decoding by construction. It changes only the choice of proposal distribution while leaving the verification procedure unchanged.

Theorem B.4 (Lossless Property of DCD). DCD produces outputs distributed identically to autoregressive decodingfrom $\pi _ { C D }$

Proof. Speculative decoding is lossless for any target distribution $\pi _ { \mathrm { t a r g e t } }$ that is a valid probability distribution (Leviathan et al., 2023; Chen et al., 2023). DCD instantiates $\pi _ { \mathrm { t a r g e t } } = \pi _ { C D }$ , where:

$$
\pi _ { C D } ( x ) = \frac { \pi _ { p } ( x ) ^ { 1 + \alpha } \pi _ { q } ( x ) ^ { - \alpha } } { \sum _ { x ^ { \prime } } \pi _ { p } ( x ^ { \prime } ) ^ { 1 + \alpha } \pi _ { q } ( x ^ { \prime } ) ^ { - \alpha } } .
$$

<table><tr><td>Setting</td><td>LHS</td><td>RHS</td><td>Gap</td></tr><tr><td>QWEN3 / GSM8K</td><td>0.392</td><td>-0.348</td><td>0.740</td></tr><tr><td>QWEN3 /MMLU</td><td>0.181</td><td>-0.571</td><td>0.753</td></tr><tr><td>LLAMA-3 / GSM8K</td><td>-0.305</td><td>-0.332</td><td>0.028</td></tr><tr><td>LLAMA-3 /MMLU</td><td>-0.274</td><td>-0.485</td><td>0.211</td></tr></table>

Table 9: Empirical alignment-gap summary in four greedy settings. Here LHS = $\mathbb { E } _ { \pi _ { e } } [ \log ( \pi _ { p } / \pi _ { q } ) ]$ , RHS $= \mathbb { E } _ { \pi _ { q } } [ \log ( \pi _ { p } / \pi _ { q } ) ]$ , and $\mathrm { \ g a p = L H S - R H S }$ ; positive gaps support Effective Draft Alignment (Definition B.2).

Since softmax outputs satisfy $\pi _ { p } ( x ) , \pi _ { q } ( x ) > 0$ for all $x \in \nu$ , the numerator is strictly positive and the normalization is well defined. Thus $\pi _ { C D }$ is a valid distribution, and the lossless property follows immediately. □

Remark B.5. The choice ofproposer affects only the acceptance rate (efficiency), never output correctness, as long as verification uses the CD target distribution. This separation is fundamental to speculative decoding and carries over to DCD unchanged.

Remark B.6 (Plausibility-constrained CD). The same argument applies when vanilla CD uses the expert-plausibility constraint ofLi et al. (2023). In that case, the target distribution is the masked-andrenormalized contrastive distribution over $\mathcal { P } _ { \beta } ( h )$ and speculative verification still samples from a valid normalized target distribution. DCD therefore remains a lossless acceleration of the corresponding truncated contrastive target; only the acceptance rate changes when the proposer emits tokens outside the mask.

## C Alignment Gap Estimation Methodology

Table 9 summarizes the empirical evidence for Definition B.2. We measure the estimates in four greedy settings: QWEN3/GSM8K, QWEN3/MMLU, LLAMA-3/GSM8K, and LLAMA-3/MMLU. Following the shared evaluation protocol in Section 3, we use 200 evaluation samples per dataset, greedy decoding $( T ~ = ~ 0 )$ , and a base draft length of $\gamma = 5$ . This appendix describes how we estimate $\mathbb { E } _ { \pi _ { e } } [ \log ( \pi _ { p } / \pi _ { q } ) ]$ and $\mathbb { E } _ { \pi _ { q } } [ \log ( \pi _ { p } / \pi _ { q } ) ]$ from draft trees collected during DCD inference.

Sampling. For each evaluation sample in the reported settings, we collect the EAGLE3 draft tree generated by greedy $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ inference with $\alpha = 0 . 1$ . At each node, we record full-vocabulary logits from the expert model $\mathcal { M } _ { p }$ , the amateur model $\mathcal { M } _ { q } .$ , and the drafter E.

Estimator. Let $r ( x ) = \log \pi _ { p } ( x ) - \log \pi _ { q } ( x )$ We then compute:

$$
\mathbb { E } _ { \pi _ { e } } \left[ \log \frac { \pi _ { p } } { \pi _ { q } } \right] = \sum _ { x \in \mathcal { V } } \pi _ { e } ( x ) \cdot r ( x ) ,
$$

$$
\mathbb { E } _ { \pi _ { q } } \left[ \log \frac { \pi _ { p } } { \pi _ { q } } \right] = \sum _ { x \in \mathcal { V } } \pi _ { q } ( x ) \cdot r ( x ) ,
$$

These expectations are evaluated over the full vocabulary at each position and then averaged across all positions and samples.

## D Throughput Analysis

This section makes explicit the coarse throughput surrogate used in Section 3.5. The formulas below use measured latency components to form analytic approximations to runtime throughput.

## D.1 DCD

DCD decouples proposal generation from the amateur model through an amateur-independent proposer $E .$

At each iteration, the proposer generates γ draft tokens, and the expert and amateur then run forward passes on the same draft sequence for verification.

We approximate the serial time per iteration by

$$
\begin{array} { r } { T _ { \mathrm { D C D } } = \gamma \cdot t _ { e } + t _ { p } + t _ { q } , } \end{array}
$$

where $t _ { e } , t _ { p } ,$ , and $t _ { q }$ denote the latency components reported in Section 3.5 for the proposer, expert path, and amateur path, respectively.

Under the same coarse surrogate $S ~ \approx ~ ( L +$ $1 ) / T _ { \mathrm { i t e r } }$ , the throughput approximation is

$$
\mathrm { S p e e d } _ { \mathrm { D C D } } = \frac { L + 1 } { \gamma \cdot t _ { e } + t _ { p } + t _ { q } } ,
$$

where L denotes the average number of accepted tokens per iteration, excluding the additional token.

When the proposer is lightweight or retrievalbased, $t _ { e } \ll t _ { q } ,$ , so the proposal overhead of DCD is much lower than that of SCD.

## D.2 SCD with Delayed Drafting

SCD (Yuan et al., 2024) with delayed drafting uses conditional drafting to reduce unnecessary draft generation.

In each iteration, the amateur model first generates $\gamma$ draft tokens. An additional draft token is generated only when all γ draft tokens are accepted by the contrastive distribution, which occurs with probability $\rho ^ { \gamma }$

The expected number of draft operations per iteration is therefore

$$
\begin{array} { r } { N _ { \mathrm { d r a f t } } = \gamma + \rho ^ { \gamma } , } \end{array}
$$

where $\rho$ denotes the single-step acceptance rate. The corresponding throughput approximation is

$$
\mathrm { S p e e d } _ { \mathrm { S C D } } = \frac { L + 1 } { \left( \gamma + \rho ^ { \gamma } \right) \cdot t _ { q } + t _ { p } } ,
$$

where L denotes the average number of accepted tokens per iteration, excluding the additional token.

In practice, we infer an effective acceptance rate $\rho$ from the observed $L$ value using the local relation $L \approx \sum _ { i = 1 } ^ { \gamma } \rho ^ { i }$

## D.3 CoS

CoS (Fu et al., 2025) uses an alternating proposal mechanism to reduce draft steps.

If, in the previous iteration, all amateur-model draft tokens and the expert proposal token are accepted, the current iteration can skip one amateurmodel draft step.

The probability of saving a draft step is

$$
P _ { \mathrm { s a v e } } = \rho _ { p } ^ { \gamma _ { p } } \cdot \rho _ { q } ^ { \gamma _ { q } } ,
$$

where $\rho _ { p }$ denotes the per-token acceptance rate for the expert proposal $( \gamma _ { p }$ is typically set to 1), and $\rho _ { q }$ denotes the per-token acceptance rate for the amateur model, so that $\rho _ { q } ^ { \gamma _ { q } }$ is the probability that all $\gamma _ { q }$ draft tokens are accepted.

Conversely, when all tokens in the current iteration are accepted, an additional amateur-model draft step is required with probability $\rho _ { q } ^ { \gamma _ { q } }$ to verify the expert proposal.

The expected number of draft steps per iteration is therefore

$$
\begin{array} { r } { N _ { \mathrm { d r a f t } } = \gamma - \rho _ { p } ^ { \gamma _ { p } } \cdot \rho _ { q } ^ { \gamma _ { q } } + \rho _ { q } ^ { \gamma _ { q } } . } \end{array}
$$

The corresponding throughput approximation is

$$
\mathrm { S p e e d } _ { \mathrm { C o S } } = \frac { L + 1 } { ( \gamma - \rho _ { p } ^ { \gamma _ { p } } \cdot \rho _ { q } ^ { \gamma _ { q } } + \rho _ { q } ^ { \gamma _ { q } } ) \cdot t _ { q } + t _ { p } } ,
$$

where L denotes the average number of accepted tokens per iteration, excluding the additional token.

In practice, we infer an effective amateur acceptance rate $\rho _ { q }$ from the observed L value using the same local relation. Because $\rho _ { p }$ is not measured directly, we set it to 0.9 as a calibration constant.

<table><tr><td>Model</td><td>Base</td><td>Acc.</td><td>Cost</td><td>Pred.</td><td>Meas.</td></tr><tr><td>LLAMA-3</td><td>SCD</td><td>0.71</td><td>1.62</td><td>1.14x</td><td>1.12x</td></tr><tr><td rowspan="2">QWEN3</td><td>CoS</td><td>0.71</td><td>1.62</td><td>1.15x</td><td>1.12x</td></tr><tr><td>SCD CoS</td><td>0.71 0.71</td><td>2.06 2.01</td><td>1.46x</td><td>1.41x</td></tr><tr><td>LLAMA-EFT</td><td>SCD</td><td></td><td></td><td>1.43x</td><td>1.38x</td></tr><tr><td></td><td></td><td>0.68</td><td>2.30</td><td>1.55x</td><td>1.49x</td></tr><tr><td></td><td>CoS</td><td>0.68</td><td>2.50</td><td>1.69x</td><td>1.61x</td></tr></table>

Table 10: Multiplicative decomposition of the speed advantage of $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ on MMLU $( T { = } 0 , \ \alpha { = } 0 . 1$ $\gamma { = } 5 )$ . For an amateur-coupled baseline b, the predicted ratio is approximated by an acceptance factor $( L _ { \mathrm { D C D } _ { \mathrm { E A G L E 3 } } } + 1 ) / ( L _ { b } + 1 )$ times a serial-cost factor $T _ { b } / T _ { \mathrm { D C D _ { E A G L E 3 } } }$ . Acc. below 1 means $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ accepts fewer tokens per speculative round; Cost above 1 means $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ has a cheaper serial proposal path. Across all model families, the cost factor is the dominant term.

## D.4 Multiplicative Throughput Decomposition

The speed formulas above yield the same coarse multiplicative decomposition for the advantage of DCD over any amateur-coupled baseline b evaluated at the same operating point:

$$
\frac { \mathrm { S p e e d } _ { \mathrm { D C D } } } { \mathrm { S p e e d } _ { b } } \approx \frac { L _ { \mathrm { D C D } } + 1 } { L _ { b } + 1 } \cdot \frac { T _ { b } } { T _ { \mathrm { D C D } } } .
$$

The first term captures acceptance and the second captures serial cost. This decomposition is useful because it separates token acceptance from generation latency. In our measured regime, the acceptance term is typically below 1, whereas the serial-cost term is well above 1 and dominates the product.

## E Additional Results and Ablations

## E.1 Main-Result Breakdowns

This subsection reports the full throughput tables underlying Table 2.

Table 11 reports the greedy results for $\alpha \in$ {0.1, 0.5}, including raw tokens/sec and reported standard deviations.

Table 12 reports the corresponding sampling results at $T = 1$

## E.2 Full Accuracy Sweep Details

This subsection documents the full evaluation configuration underlying Section 3.4.

Setup. We evaluate the Llama-3.1-8B-Instruct and Llama-3.2-1B-Instruct pair on GSM8K and

MMLU with 1000 examples per setting, draft length $\gamma { = } 5$ , and $\alpha \in \{ 0 . 0 , 0 . 1 , \ldots , 1 . 0 \}$ . We report $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ , SCD, CoS, vanilla CD, and the expert-only autoregressive baseline under both temperatures, with the matched $\mathrm { D C D } _ { \mathrm { N G R A M } }$ reference overlaid where available. Error bars denote 95% confidence intervals.

Figure 10 presents the complete accuracy sweep for both greedy decoding $( T { = } 0 )$ and sampling $( T \mathrm { = } 1 )$ . The matched $\mathrm { D C D } _ { \mathrm { N G R A M } }$ reference curve is overlaid on both temperature settings, alongside the main EAGLE3-based DCD trajectory.

## E.3 Additional Hyperparameter Ablations

Figure 11 reports the complementary draft-length sweep referenced in the main text. Table 13 summarizes the same study in the fixed-γ versus best-γ format discussed there. Table 14 reports the full $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ throughput grid at $\alpha { = } 0 . 1$ . Each row corresponds to one evaluated combination of speculative steps, draft tokens, and top-k, and the last two columns report the resulting throughput on GSM8K and MMLU.

## F Extended System Evaluation

## F.1 Multi-Request Speedups on GSM8K

Table 15 reports multi-request serving speedups on GSM8K. The $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ , SCD, and matched $\mathrm { D C D } _ { \mathrm { N G R A M } }$ rows use the same serving configuration: five speculative steps with chain-style greedy drafting and six draft tokens per speculative round. Each entry reports throughput normalized by vanilla CD at the same request count.

## F.2 Runtime KV-Cache Occupancy

We examine runtime KV-cache occupancy in the representative LLAMA-3 8B setting from the main paper, where LLAMA-3.1-8B-INSTRUCT is the expert, LLAMA-3.2-1B-INSTRUCT is the amateur, and $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ uses the same EAGLE3 drafter. AR serves only the expert, vanilla CD and SCD keep both the expert and amateur active, and $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ changes only the proposal path through a lightweight feature-level drafter. In both the single-request and shared-batching settings, vanilla CD, SCD, and $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ show nearly identical steady-state KV-cache occupancy because the persistent verification-time KV state remains dominated by the expert and amateur decoders.

## F.3 Spec-Bench Evaluation

To test whether the same speed pattern extends beyond the four main datasets, we also report results on Spec-Bench (Xia et al., 2024), a standardized benchmark for speculative decoding with six subtasks: multi-turn conversation (MT-Bench), translation, summarization, question answering, mathematical reasoning, and retrieval-augmented generation. Each subtask contains 80 prompts, giving 480 prompts in total. Table 16 reports per-subtask throughput and average speedup under this setting.

## F.4 70B-Class Results

We report a greedy-only 70B extension with Llama-3.3-70B-Instruct as the expert and Llama-3.1-8B-Instruct as the amateur. The sweep uses the same four datasets and $\alpha \in$ $\{ 0 . 0 , 0 . 1 , 0 . 5 \}$ ; these results are single-run diagnostics for checking whether the main trend extends to a larger expert.

Table 17 reports the per-dataset throughput breakdown for the EAGLE3 and matched N-gram proposer settings. Table 18 reports the corresponding accepted-length robustness, normalized to each method at $\alpha = 0 . 0$

## G Model Details

Table 19 lists the off-the-shelf models and EAGLE3 proposer checkpoints used in the deployment experiments. The controlled proposal-side diagnostics that train dual-input or auxiliary EAGLE heads are described separately in Section 3.1 and Appendix A.2.

## H Implementation Details

Experimental Environment. Unless noted otherwise, benchmarks run on a single NVIDIA H200 141GB GPU under the SGLang (Zheng et al., 2024) inference framework.

Artifacts and Intended Use. We use publicly released benchmark datasets, model checkpoints, EAGLE3 proposer checkpoints, and the SGLang inference framework under their respective licenses and access terms. These artifacts are used only for research benchmarking of decoding efficiency and task-level consistency, and we do not redistribute third-party datasets or model weights.

Inference Configuration. To measure latency cleanly, the main single-request deployment and diagnostic experiments use single-card serial execution. Table 20 summarizes the key singlerequest runtime parameters. Runtime KV-cache occupancy under this setup is examined in Appendix F.2; multi-request serving uses the dedicated concurrent-request settings in Appendix F.1. For EAGLE3, speculative-eagle-topk= 1 implements chain-style greedy drafting. For $\mathrm { D C D } _ { \mathrm { N G R A M } } .$ , the proposer uses a draft-token budget of 5, an N-gram match-window range from 1 to 12, BFS breadth 1 for chain-style proposal selection, and branch length 18.

Dataset and Prompt Configuration. For the main deployment and task-accuracy evaluations, we use the official chat templates for all models and evaluate two settings:

• Main Experiments (Speed Analysis): We use 200 samples per dataset with a maximum generation length of 256 tokens.

• Performance Tests (Accuracy Verification): We use 1000 samples with a maximum generation length of 1024 tokens and task-specific zero-shot templates with structured output formats, for example, “Final Answer:”, to support automated evaluation.

![](images/6ab8f95d35b5d70fb441fa573dde2db6e2870cde313a5f93853167d7e99c9fa2.jpg)  
Figure 6: Configuration-level ∆Top-1 across $\operatorname { K L } ( \pi _ { f } \parallel \pi _ { q } )$ buckets. The association is heterogeneous across model groups and contrastive strengths: for $\alpha \leq 0 . 5$ , KL-axis patterns are weak or mixed, whereas at $\alpha = 1 . 0$ the gap often becomes less negative at higher KL. No configuration-invariant positive-gain regime emerges.

![](images/d8e4f11761c945a6889b1fe8499ecb774b15b6d7d4cc784eea7f7c6a179cc600.jpg)  
Figure 7: Compact regime heatmap for the expanded root-level contrastive-signal bucket analysis. Here contrastive signal denotes the per-position expert–amateur absolute log-ratio $\begin{array} { r } { \left| \log \frac { \pi _ { p } ( x | h ) } { \pi _ { q } ( x | h ) } \right| , } \end{array}$ . Rows are grouped by model family and dataset, with per-row tick labels indicating $\alpha ,$ , and columns are mutually exclusive contrastive-signa buckets. Low-signal buckets are mostly negative, while several higher-signal buckets are positive. Warm colors favor contrastive-aligned drafting; cool colors favor expert-aligned drafting. To preserve visualization contrast for moderate variations, the color scale is truncated at $\pm 0 . 3 .$

![](images/7e75226a177a20e39130a6508d97f6e6ed1b5cfe94151990bc454cc792bf1655.jpg)  
Figure 8: Per-group ∆Top-1 trends across contrastive-signal buckets. Here contrastive signal denotes the perposition expert–amateur absolute log-ratio $\begin{array} { r } { \left| \log \frac { \pi _ { p } ( x | h ) } { \pi _ { q } ( x | h ) } \right| } \end{array}$ . Each subplot fixes a model group and dataset, and each line corresponds to one contrastive strength α. Larger α usually worsens low-signal buckets but can amplify gains in higher-signal buckets, especially for the Llama-EFT settings. For clarity of the moderate-signal trends, the extreme outlier bucket (≥ 8) is excluded from the horizontal axis.

(a) Contrastive signal distribution  
![](images/7750e6293769c1a76bebe89c3f7eae6c3267b98dd984f67f70c1e09d298605ab.jpg)  
Signal bucket

(b) Effect by signal and proposal error  
![](images/c7cc4906e1fec65d8a7f9be76564b948706d4d9c6a739efe76f345f45b1e388b.jpg)  
KL(π<sub>r</sub> ∥ π<sub>p</sub>)

(c) KL(π<sub>r</sub> ∥ π<sub>p</sub>) distribution  
![](images/9788b5950f16cf21468a918accbe54451642e86994861ac5299c74036eacbc58.jpg)  
$\mathrm { K L } ( \pi _ { r } \parallel \pi _ { p } )$ bucket  
Figure 9: Full offline 3B proposer diagnostic. ∆Top-1 is the CD-aware reconstruction’s Top-1 agreement gain over the unmodified 3B proposer, measured against arg max<sub>x</sub> $\pi _ { C D } ( x \mid h )$ . Low-signal rows remain mostly negative; setting-level mean ∆Top-1 values are slightly negative in all four settings (GSM8K $( \alpha = 0 . 1 )$ -0.0028; GSM8K $( \alpha = 0 . 5 ) \ - \mathrm { - } 0 . 0 1 2 3 $ ; MMLU (α = 0.1) -0.0030; MMLU $( \alpha = 0 . 5 ) \lrcorner 0 . 0 1 4 4 )$

![](images/9f6188b7f62a6e19977c4dfda47dd008b66d8d86209e1505f69b7b646376ee8e.jpg)

![](images/3c07fcc0fa9991284c05d1a05a58704da41440d659940a0e1692297f9dea4e19.jpg)

![](images/56aa77d266e2a5b93844a43e95cea88d2f93be926ba03940c3a5b7b00a44e8a7.jpg)

![](images/4750fb3eac7de91a73eae4cc92a167b81683d5f9910166073bf43caf6118bda3.jpg)  
Contrastive Strength (α)  
DCD DCD SCD CoS Vanilla CD AR

Figure 10: Full task-accuracy sweep across contrastive strengths α under the actual SGLang setup. The plot shows $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ , SCD, CoS, and vanilla CD on the Llama-3.1-8B-Instruct + Llama-3.2-1B-Instruct pair with $\gamma { = } 5$ for GSM8K and MMLU under both greedy decoding $( T { = } 0 )$ and sampling $( T { = } 1 )$ . A matched $\mathrm { D C D } _ { \mathrm { N G R A M } }$ reference is overlaid on both temperature settings where the N-gram proposer run is available. Error bars indicate 95% confidence intervals, and the dashed gray line shows the expert-only autoregressive baseline. From left to right: GSM8K with greedy decoding $( T { = } 0 )$ , GSM8K with sampling $( T { = } 1 )$ , MMLU with greedy decoding $( T { = } 0 )$ , and MMLU with sampling $( T { = } 1 )$ .

![](images/8d408068c28bf8759149f89c30e22cec1371310b63591cfbec764fa0b5d74581.jpg)  
Draft Tokens (γ)

![](images/346e21dd9f57972212ade56c06b64653cb83067a0a683b9d285ec0fa802be9f7.jpg)  
DCD<sub>EAGLE3</sub>

![](images/3b79fdb482081044d930cb9a7aeae4849baa4f7fa55efde408764da9845f32ec.jpg)  
Draft Tokens (γ)  
SCD CoS

![](images/30fb4b9f5b38622c0bbc9035032c3c5ca06cc9e3ccfd19262e5e861f6fb81713.jpg)  
Draft Tokens (γ)  
Figure 11: Appendix-only γ ablation. The y-axis reports speedup relative to vanilla CD. $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ stays efficient across a broader draft-length range, while the amateur-coupled baselines usually peak earlier as larger draft lengths make their heavier proposal paths less worthwhile.

<table><tr><td>Method</td><td>HumanEval</td><td>GSM8K</td><td>MMLU</td><td>CNN/DM</td><td>Avg.</td></tr><tr><td colspan="6">Llama-3 (Llama-3.1-8B-Instruct / Llama-3.2-1B-Instruct)</td></tr><tr><td colspan="6"> $\alpha = 0 . 1$ </td></tr><tr><td>CD</td><td> $9 1 . 3 { \pm } 4 . 3 \left( 1 . 0 0 \times \right)$ </td><td> $8 6 . 6 { \pm } 5 . 6 \left( 1 . 0 0 { \times } \right)$ </td><td> $9 3 . 1 { \pm } 2 . 7 \left( 1 . 0 0 \times \right)$ </td><td> $8 4 . 0 { \pm } 0 . 6 \left( 1 . 0 0 { \times } \right)$ </td><td>88.7 (1.00×)</td></tr><tr><td>SCD</td><td> $1 3 9 . 2 { \pm } 4 . 7 \ : ( 1 . 5 2 \times )$ </td><td> $1 3 2 . 1 { \pm } 2 . 4 \left( 1 . 5 3 \times \right)$ </td><td> $1 1 7 . 4 { \pm } 1 . 3 \ : ( 1 . 2 6 \times )$ </td><td> $1 1 7 . 4 { \pm } 2 . 5 \ : ( 1 . 4 0 \times )$ </td><td>126.5 (1.43×)</td></tr><tr><td>CoS</td><td> $1 5 6 . 5 { \pm } 1 6 . 9 \left( 1 . 7 1 \times \right)$ </td><td> $1 5 3 . 7 { \pm } 2 . 6 \left( 1 . 7 7 \times \right)$ </td><td> $1 1 9 . 9 { \pm } 4 . 7 \ : ( 1 . 2 9 \times )$ </td><td> $1 1 5 . 0 { \pm } 6 . 4 \ : ( 1 . 3 7 \times )$ </td><td>136.3 (1.54×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td> $1 3 5 . 0 { \pm } 1 . 3 \ : ( 1 . 4 8 \times )$  </td><td> $1 2 4 . 0 { \pm } 1 . 4 \left( 1 . 4 3 \times \right)$ </td><td> $1 2 0 . 5 { \pm } 0 . 1 \ : ( 1 . 2 9 \times )$ </td><td> $1 7 3 . 6 { \pm } 3 . 0 \left( 2 . 0 7 \times \right)$ </td><td>138.3 (1.56×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td> $\mathbf { 1 8 9 . 2 } \pm 2 . 5 \left( 2 . \mathbf { 0 7 } \times \right)$ </td><td> $\mathbf { 1 5 9 . 7 { \pm } 1 . 6 \left( 1 . 8 4 \times \right)} $ </td><td> $\mathbf { 1 4 9 . 0 { \pm } 7 . 2 \ ( 1 . 6 0 \times ) }$ </td><td> $\mathbf { 1 5 1 . 4 { \pm } 2 . 1 \ ( 1 . 8 0 \times ) }$ </td><td>162.3 (1.83×)</td></tr><tr><td colspan="6">α = 0.5</td></tr><tr><td>CD</td><td> $9 5 . 7 { \pm } 0 . 6 \left( 1 . 0 0 \times \right)$ </td><td> $9 1 . 7 { \pm } 3 . 7 \ : ( 1 . 0 0 \times )$ </td><td> $8 3 . 0 { \pm } 0 . 7 \left( 1 . 0 0 { \times } \right)$ </td><td> $9 0 . 0 { \pm } 3 . 5 \ : ( 1 . 0 0 { \times } )$ </td><td> $9 0 . 1 \left( 1 . 0 0 \times \right)$ </td></tr><tr><td>SCD</td><td> $1 5 2 . 2 { \pm } 0 . 7 \ : ( 1 . 5 9 \times )$ </td><td> $1 3 0 . 8 { \pm } 1 . 0 \left( 1 . 4 3 \times \right)$ </td><td> $1 0 0 . 1 { \pm } 0 . 6 \left( 1 . 2 1 \times \right)$ </td><td> $1 0 0 . 5 { \pm } 2 . 3 \ : ( 1 . 1 2 \times )$ </td><td>120.9 (1.34×)</td></tr><tr><td>CoS</td><td> $1 5 9 . 6 { \pm } 0 . 2 \ : ( 1 . 6 7 { \times } )$ </td><td> $1 4 2 . 8 { \pm } 0 . 9 \left( 1 . 5 6 \times \right)$ </td><td> $9 9 . 7 { \pm } 0 . 6 \left( 1 . 2 0 \times \right)$ </td><td> $1 0 1 . 6 { \pm } 3 . 4 \left( 1 . 1 3 \times \right)$ </td><td>126.0 (1.40×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td> $1 3 3 . 3 { \pm } 0 . 7 \ : ( 1 . 3 9 \times )$ </td><td> $1 2 2 . 7 { \pm } 0 . 2 \ : ( 1 . 3 4 \times )$ </td><td> $1 2 8 . 3 { \pm } 2 . 3 \ ( 1 . 5 5 { \times } )$ </td><td> $1 8 8 . 8 { \pm } 1 . 4 \ : ( 2 . 1 0 { \times } )$ </td><td>143.3 (1.59×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td> $\mathbf { 1 8 1 . 4 { \pm } 1 . 1 \left( 1 . 8 9 \times \right) }$ </td><td> $\mathbf { 1 5 2 . 1 } { \pm } 2 . 6 \left( \mathbf { 1 . 6 6 } \times \right)$ </td><td> $\mathbf { 1 1 9 . 5 { \pm } 0 . 2 \ ( 1 . 4 4 \times ) }$ </td><td> $1 4 3 . 2 { \pm } 3 . 5 \ : ( 1 . 5 9 \times )$ </td><td>149.0 (1.65×)</td></tr><tr><td colspan="6"> $Q w e n 3 \left( Q w e n 3  – 8 B / Q w e n 3 – I . 7 B \right)$ </td></tr><tr><td colspan="6"> $\alpha = 0 . 1$ </td></tr><tr><td>CD</td><td> $5 9 . 9 { \pm } 5 . 7 \left( 1 . 0 0 \times \right)$ </td><td> $6 5 . 7 { \pm } 1 . 8 \left( 1 . 0 0 \times \right)$ </td><td> $6 5 . 7 { \pm } 0 . 4 \ : ( 1 . 0 0 { \times } )$ </td><td> $6 1 . 0 { \pm } 4 . 1 \ ( 1 . 0 0 { \times } )$ </td><td>63.1 (1.00×)</td></tr><tr><td>SCD</td><td> $8 2 . 8 { \pm } 3 . 9 \left( 1 . 3 8 \times \right)$ </td><td> $8 8 . 5 { \pm } 0 . 9 \left( 1 . 3 5 { \times } \right)$ </td><td> $6 4 . 8 { \pm } 0 . 2 ( 0 . 9 9 { \times } )$ </td><td> $6 4 . 0 { \pm } 3 . 6 \left( 1 . 0 5 { \times } \right)$ </td><td>75.0 (1.19×)</td></tr><tr><td>CoS</td><td> $8 7 . 1 { \pm } 2 . 1 \ ( 1 . 4 5 { \times } )$ </td><td> $8 7 . 4 { \pm } 6 . 5 ( 1 . 3 3 { \times } )$ </td><td> $7 6 . 6 { \pm } 0 . 4 \left( 1 . 1 7 \times \right)$ </td><td> $6 9 . 3 { \pm } 2 . 4 \left( 1 . 1 4 \times \right)$ </td><td>80.1 (1.27×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td> $8 7 . 1 { \pm } 0 . 8 \left( 1 . 4 5 { \times } \right)$ </td><td> $8 8 . 9 { \pm } 1 . 2 \left( 1 . 3 5 \times \right)$ </td><td> $7 7 . 1 { \pm } 1 . 4 \left( 1 . 1 7 \times \right)$ </td><td> $6 9 . 6 { \pm } 2 . 4 \left( 1 . 1 4 \times \right)$ </td><td>80.7 (1.28×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td> $\mathbf { 1 3 7 . 8 \pm 0 . 8 \ ( 2 . 3 0 \times ) }$ </td><td> $\mathbf { 1 3 1 . 9 { \pm } 4 . 1 \ : ( 2 . 0 1 \times ) }$  </td><td> $\mathbf { 1 0 8 . 8 { \pm } 4 . 8 \ ( 1 . 6 6 \times ) }$ </td><td> $\mathbf { 1 1 3 . 0 { \pm } 1 . 2 \left( 1 . 8 5 \times \right)} $ </td><td>122.9 (1.95×)</td></tr><tr><td colspan="6">α = 0.5</td></tr><tr><td></td><td> $6 4 . 1 { \pm } 4 . 1 \ : ( 1 . 0 0 \times )$ </td><td> $6 6 . 9 { \pm } 0 . 2 \ : ( 1 . 0 0 { \times } )$ </td><td> $6 4 . 4 { \pm } 2 . 0 \left( 1 . 0 0 \times \right)$ </td><td> $6 3 . 2 { \pm } 3 . 3 \ : ( 1 . 0 0 { \times } )$ </td><td> $6 4 . 6 \left( 1 . 0 0 \times \right)$ </td></tr><tr><td>SCD</td><td> $7 8 . 8 { \pm } 0 . 7 \ ( 1 . 2 3 { \times } )$ </td><td> $8 3 . 1 { \pm } 1 . 4 \left( 1 . 2 4 \times \right)$ </td><td> $5 6 . 8 { \pm } 6 . 0 ( 0 . 8 8 { \times } )$ </td><td> $5 1 . 9 { \pm } 0 . 9 \ : ( 0 . 8 2 \times )$ </td><td>67.6 (1.05×)</td></tr><tr><td>CoS</td><td> $8 0 . 1 { \pm } 2 . 6 \left( 1 . 2 5 { \times } \right)$ </td><td> $8 7 . 7 { \pm } 0 . 9 \left( 1 . 3 1 \times \right)$ </td><td> $6 2 . 1 { \pm } 2 . 2 \ : ( 0 . 9 7 { \times } )$ </td><td> $4 9 . 7 { \pm } 2 . 9 \ : ( 0 . 7 9 \times )$ </td><td>69.9 (1.08×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td> $8 4 . 5 { \pm } 0 . 2 \ : ( 1 . 3 2 { \times } )$ </td><td> $8 5 . 3 { \pm } 0 . 5 \left( 1 . 2 8 \times \right)$  </td><td> $7 5 . 2 { \pm } 1 . 8 \left( 1 . 1 7 \times \right)$ </td><td> $6 3 . 4 { \pm } 1 . 2 ( 1 . 0 0 \times )$ </td><td>77.1 (1.19×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td> $\mathbf { 1 2 7 . 1 { \pm } 0 . 7 \ : ( 1 . 9 8 \times ) }$ </td><td> $1 2 3 . 4 { \pm } 4 . 8 \left( 1 . 8 5 \times \right)$ </td><td> $\mathbf { 1 0 3 . 0 { \pm } 4 . 5 \ : ( 1 . 6 0 \times ) }$ </td><td> $\mathbf { 8 8 . 1 \pm 3 . 3 ( 1 . 3 9 \times ) }$ </td><td>110.4 (1.71×)</td></tr><tr><td colspan="6"> $L l a m a { - } E F T ( L l a m a { - } 3 . I { - } 8 B { - } I n s t r u c t / L l a m a { - } 3 . I { - } 8 B )$ </td></tr><tr><td colspan="6">α = 0.1</td></tr><tr><td>CD</td><td> $7 5 . 9 { \pm } 1 . 5 \left( 1 . 0 0 \times \right)$ </td><td> $7 5 . 0 { \pm } 0 . 8 \ : ( 1 . 0 0 { \times } )$ </td><td> $7 3 . 1 { \pm } 3 . 6 \left( 1 . 0 0 { \times } \right)$ </td><td> $6 9 . 1 { \pm } 0 . 5 \ : ( 1 . 0 0 { \times } )$ </td><td>73.3 (1.00×)</td></tr><tr><td>SCD</td><td> $9 1 . 4 { \pm } 2 . 1 \ : ( 1 . 2 0 \times )$ </td><td> $8 7 . 1 { \pm } 1 . 0 \left( 1 . 1 6 \times \right)$ </td><td> $7 7 . 5 { \pm } 3 . 1 \ ( 1 . 0 6 \times )$ </td><td> $7 1 . 4 { \pm } 1 . 1 \ : ( 1 . 0 3 \times )$ </td><td>81.9 (1.12×)</td></tr><tr><td>CoS</td><td> $9 8 . 0 { \pm } 3 . 5 \ ( 1 . 2 9 { \times } )$ </td><td> $9 1 . 5 { \pm } 1 . 3 \left( 1 . 2 2 \times \right)$ </td><td> $7 3 . 9 { \pm } 1 . 3 \left( 1 . 0 1 \times \right)$ </td><td> $7 3 . 2 { \pm } 5 . 5 \left( 1 . 0 6 \times \right)$ </td><td>84.1 (1.15×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td> $1 0 8 . 3 { \pm } 0 . 5 \ : ( 1 . 4 3 { \times } )$ </td><td> $9 7 . 9 { \pm } 1 . 3 \left( 1 . 3 1 \times \right)$ </td><td> $9 6 . 2 { \pm } 0 . 4 \left( 1 . 3 2 \times \right)$ </td><td> $1 4 2 . 1 { \pm } 1 . 8 \ : ( 2 . 0 6 \times )$ </td><td>111.1 (1.52×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td> $\mathbf { 1 5 4 . 5 \pm 5 . 2 \ : ( 2 . 0 3 \times ) }$ </td><td> $\mathbf { 1 3 0 . 1 { \pm } 0 . 4 \ ( 1 . 7 4 \times ) }$ </td><td> $\mathbf { 1 1 3 . 3 { \pm } 1 . 7 \left( 1 . 5 5 \times \right)} $ </td><td> $\mathbf { 1 2 0 . 6 { \pm } 8 . 0 \ ( 1 . 7 4 \times ) }$ </td><td>129.6 (1.77×)</td></tr><tr><td colspan="6">α = 0.5</td></tr><tr><td>CD</td><td> $7 0 . 7 { \pm } 2 . 9 \left( 1 . 0 0 \times \right)$ </td><td> $7 2 . 3 { \pm } 2 . 8 \left( 1 . 0 0 \times \right)$ </td><td> $7 3 . 4 { \pm } 3 . 4 \left( 1 . 0 0 { \times } \right)$ </td><td> $7 1 . 6 { \pm } 3 . 6 \left( 1 . 0 0 { \times } \right)$ </td><td>72.0 (1.00×)</td></tr><tr><td>SCD</td><td> $8 4 . 4 { \pm } 3 . 7 \left( 1 . 1 9 \times \right)$ </td><td> $8 2 . 5 { \pm } 0 . 7 \ : ( 1 . 1 4 \times )$ </td><td> $7 0 . 1 \pm 1 . 9 ( 0 . 9 6 \times )$ </td><td> $6 5 . 8 { \pm } 1 . 0 ( 0 . 9 2 { \times } )$ </td><td>75.7 (1.05×)</td></tr><tr><td>CoS</td><td> $9 2 . 8 { \pm } 0 . 6 \left( 1 . 3 1 \times \right)$ </td><td> $8 3 . 2 { \pm } 1 . 1 \left( 1 . 1 5 \times \right)$ </td><td> $7 1 . 6 { \pm } 1 . 8 ( 0 . 9 8 { \times } )$ </td><td> $6 9 . 3 { \pm } 1 . 6 \left( 0 . 9 7 \times \right)$ </td><td>79.2 (1.10×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td> $1 0 9 . 3 { \pm } 0 . 6 \left( 1 . 5 5 \times \right)$ </td><td> $9 7 . 3 { \pm } 0 . 9 \left( 1 . 3 5 \times \right)$ </td><td> $9 6 . 0 { \pm } 1 . 4 \left( 1 . 3 1 \times \right)$ </td><td> $1 5 1 . 1 { \pm } 0 . 9 ( 2 . 1 1 \times )$ </td><td>113.4 (1.58×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td> $\mathbf { 1 4 5 . 5 \pm 5 . 8 \ ( 2 . 0 6 \times ) }$  </td><td> $\mathbf { 1 1 6 . 8 { \pm } 6 . 0 \ ( 1 . 6 2 \times ) }$ </td><td> $\mathbf { 1 1 7 . 9 { \pm } 5 . 1 \ : ( 1 . 6 1 \times ) }$ </td><td> $\mathbf { 1 1 5 . 7 { \pm } 6 . 4 ( 1 . 6 1 \times ) }$ </td><td>124.0 (1.72×)</td></tr><tr><td colspan="6"> $L l a m a \ – 3 \ ( L l a m a \ – 3 . I \mathrm { - } \delta B \mathrm { - } I n s t r u c t / L l a m a \ – 3 . 2 – I B \mathrm { - } I n s t r u c t )$ </td></tr><tr><td colspan="6"> $\alpha = 0 . 1$ </td></tr><tr><td>CD</td><td> $9 2 . 4 { \pm } 3 . 3 \left( 1 . 0 0 \times \right)$ </td><td> $9 0 . 5 { \pm } 6 . 6 \left( 1 . 0 0 { \times } \right)$ </td><td> $9 0 . 3 { \pm } 3 . 3 \left( 1 . 0 0 { \times } \right)$ </td><td> $8 8 . 2 { \pm } 2 . 6 \left( 1 . 0 0 { \times } \right)$ </td><td>90.3 (1.00×)</td></tr><tr><td>SCD</td><td> $1 2 0 . 7 { \pm } 1 . 6 \left( 1 . 3 1 \times \right)$ </td><td> $1 2 5 . 3 { \pm } 0 . 9 \left( 1 . 3 9 \times \right)$ </td><td> $1 0 4 . 3 { \pm } 4 . 9 \ : ( 1 . 1 6 \times )$ </td><td> $1 0 7 . 8 { \pm } 3 . 2 \ : ( 1 . 2 2 \times )$ </td><td>114.6 (1.27×)</td></tr><tr><td>CoS</td><td> $1 5 6 . 3 { \pm } 5 . 9 \left( 1 . 6 9 { \times } \right)$ </td><td> $1 3 9 . 3 { \pm } 2 . 4 \left( 1 . 5 4 \times \right)$ </td><td> $1 0 9 . 2 { \pm } 2 . 3 \ : ( 1 . 2 1 \times )$ </td><td> $1 0 8 . 7 { \pm } 5 . 2 \left( 1 . 2 3 \times \right)$ </td><td>128.4 (1.42×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td> $1 2 8 . 9 { \pm } 2 . 0 \left( 1 . 3 9 \times \right)$ </td><td> $1 1 4 . 4 { \pm } 1 . 5 \ : ( 1 . 2 6 \times )$ </td><td> $1 0 8 . 4 { \pm } 0 . 9 \left( 1 . 2 0 { \times } \right)$ </td><td> $1 4 2 . 1 { \pm } 0 . 9 \left( 1 . 6 1 \times \right)$ </td><td>123.4 (1.37×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td> $\mathbf { 1 6 1 . 3 { \pm } 8 . 9 \ ( 1 . 7 5 \times ) }$ </td><td> ${ \bf 1 3 9 . 5 \pm 0 . 6 \left( 1 . 5 4 \times \right) }$ </td><td> $\mathbf { 1 1 9 . 6 { \pm } 5 . 2 \left( 1 . 3 2 \times \right)} $ </td><td> $\mathbf { 1 4 0 . 9 { \pm } 1 . 7 \ ( 1 . 6 0 \times ) }$ </td><td>140.3 (1.55×)</td></tr><tr><td colspan="6"> $\alpha = 0 . 5$ </td></tr><tr><td>CD</td><td> $9 3 . 3 { \pm } 3 . 3 \left( 1 . 0 0 \times \right)$ </td><td> $9 0 . 8 { \pm } 5 . 0 \left( 1 . 0 0 { \times } \right)$ </td><td> $8 6 . 0 { \pm } 5 . 0 \left( 1 . 0 0 { \times } \right)$ </td><td> $9 3 . 1 { \pm } 2 . 0 \left( 1 . 0 0 \times \right)$ </td><td>90.8 (1.00×)</td></tr><tr><td>SCD</td><td> $1 2 6 . 6 { \pm } 2 . 0 \left( 1 . 3 6 \times \right)$ </td><td> $1 1 6 . 8 { \pm } 2 . 6 \left( 1 . 2 9 \times \right)$ </td><td> $9 1 . 8 { \pm } 0 . 8 \left( 1 . 0 7 { \times } \right)$ </td><td> $9 3 . 5 { \pm } 5 . 1 \ : ( 1 . 0 0 \times )$ </td><td>107.2 (1.18×)</td></tr><tr><td>CoS</td><td> $1 3 4 . 7 { \pm } 9 . 3 \ : ( 1 . 4 4 \times )$ </td><td> $1 1 2 . 4 { \pm } 0 . 4 \left( 1 . 2 4 \times \right)$ </td><td> $8 5 . 0 { \pm } 3 . 1 \ ( 0 . 9 9 { \times } )$ </td><td> $9 6 . 8 { \pm } 5 . 4 ( 1 . 0 4 \times )$ </td><td>107.2 (1.18×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td> $1 2 6 . 7 { \pm } 1 . 3 \ : ( 1 . 3 6 \times )$ </td><td> $1 1 4 . 5 { \pm } 1 . 3 \ : ( 1 . 2 6 \times )$ </td><td> $1 0 4 . 9 { \pm } 1 . 8 \ : ( 1 . 2 2 \times )$ </td><td> $1 4 7 . 4 { \pm } 1 . 3 \left( 1 . 5 8 \times \right)$ </td><td>123.4 (1.36×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td> $\mathbf { 1 8 3 . 1 \pm 2 . 4 \ ( 1 . 9 6 \times ) }$ </td><td> $\mathbf { 1 4 8 . 1 \pm 4 . 4 ( 1 . 6 3 \times ) }$ </td><td> $\mathbf { 1 } 3 2 . 6 { \pm } 1 . 9 \left( \mathbf { 1 . 5 4 } \times \right)$ </td><td> $\mathbf { 1 6 5 . 2 { \pm 0 . 1 ( 1 . 7 7 \times ) } }$ </td><td>157.2 (1.73×)</td></tr><tr><td colspan="6"> $Q w e n 3 \left( Q w e n 3  – 8 B / Q w e n 3 – I . 7 B \right)$ </td></tr><tr><td colspan="6"> $\alpha = 0 . 1$ </td></tr><tr><td>CD</td><td></td><td> $6 3 . 9 { \pm } 0 . 2 \ : ( 1 . 0 0 \times )$ </td><td> $6 4 . 2 { \pm } 2 . 2 \ : ( 1 . 0 0 { \times } )$ </td><td> $6 1 . 5 { \pm } 2 . 1 \ : ( 1 . 0 0 \times )$ </td><td> $6 3 . 7 \left( 1 . 0 0 \times \right)$ </td></tr><tr><td>SCD</td><td> $6 5 . 1 { \pm } 3 . 4 ( 1 . 0 0 { \times } )$   $7 6 . 2 { \pm } 7 . 6 \left( 1 . 1 7 \times \right)$ </td><td> $8 5 . 3 { \pm } 0 . 7 \ : ( 1 . 3 4 \times )$ </td><td> $6 7 . 2 { \pm } 3 . 0 \left( 1 . 0 5 { \times } \right)$ </td><td> $6 3 . 4 { \pm } 2 . 8 \ : ( 1 . 0 3 { \times } )$ </td><td>73.0 (1.15×)</td></tr><tr><td>CoS</td><td> $8 4 . 8 { \pm } 1 . 7 \left( 1 . 3 0 { \times } \right)$ </td><td> $8 9 . 8 { \pm } 1 . 5 \left( 1 . 4 1 \times \right)$ </td><td> $7 3 . 2 { \pm } 0 . 9 \left( 1 . 1 4 \times \right)$ </td><td> $6 6 . 8 { \pm } 1 . 5 \ : ( 1 . 0 9 { \times } )$ </td><td>78.7 (1.24×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td> $8 5 . 0 { \pm } 0 . 9 \left( 1 . 3 1 \times \right)$ </td><td> $8 7 . 6 { \pm } 1 . 2 \left( 1 . 3 7 { \times } \right)$ </td><td> $7 7 . 1 { \pm } 0 . 6 \left( 1 . 2 0 { \times } \right)$ </td><td> $6 9 . 3 { \pm } 0 . 6 \left( 1 . 1 3 \times \right)$ </td><td>79.7 (1.25×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td> $\mathbf { 1 3 3 . 6 { \pm } 1 . 4 \ : ( 2 . 0 5 \times ) }$ </td><td> $\mathbf { 1 2 5 . 2 { \pm } 5 . 6 \left( 1 . 9 6 \times \right) }$ </td><td> ${ \bf 1 0 6 . 9 { \pm } 5 . 4 ( 1 . 6 7 \times ) }$ </td><td> ${ \bf 1 0 3 . 8 { \pm } 2 . 4 \left( 1 . 6 9 \times \right)} $ </td><td>117.4 (1.84×)</td></tr><tr><td colspan="6"> $\alpha = 0 . 5$ </td></tr><tr><td>CD</td><td> $6 4 . 6 { \pm } 2 . 0 \left( 1 . 0 0 { \times } \right)$ </td><td> $6 5 . 9 { \pm } 0 . 3 \ : ( 1 . 0 0 { \times } )$ </td><td> $6 1 . 5 { \pm } 2 . 5 \ : ( 1 . 0 0 { \times } )$ </td><td> $6 0 . 4 { \pm } 0 . 3 \ : ( 1 . 0 0 \times )$ </td><td> $6 3 . 1 \left( 1 . 0 0 \times \right)$ </td></tr><tr><td>SCD</td><td> $6 9 . 0 { \pm } 0 . 8 \left( 1 . 0 7 \times \right)$ </td><td> $7 9 . 7 { \pm } 1 . 5 \ : ( 1 . 2 1 \times )$ </td><td> $5 7 . 7 { \pm } 3 . 7 \ : ( 0 . 9 4 \times )$ </td><td> $4 9 . 3 { \pm } 1 . 5 \ : ( 0 . 8 2 \times )$ </td><td> $6 3 . 9 \left( 1 . 0 1 \times \right)$ </td></tr><tr><td>CoS</td><td> $7 4 . 3 { \pm } 2 . 9 \left( 1 . 1 5 \times \right)$ </td><td> $8 4 . 4 { \pm } 1 . 2 ( 1 . 2 8 \times )$ </td><td> $6 1 . 0 { \pm } 3 . 3 ( 0 . 9 9 { \times } )$ </td><td> $4 8 . 1 { \pm } 2 . 2 \ : ( 0 . 8 0 { \times } )$ </td><td>67.0 (1.06×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td> $8 0 . 4 { \pm } 0 . 7 \ ( 1 . 2 5 { \times } )$ </td><td> $8 5 . 5 { \pm } 0 . 8 \left( 1 . 3 0 { \times } \right)$ </td><td> $7 2 . 9 { \pm } 1 . 4 \left( 1 . 1 9 \times \right)$ </td><td> $6 4 . 8 { \pm } 0 . 2 \ : ( 1 . 0 7 { \times } )$ </td><td>75.9 (1.20×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td> $\mathbf { 1 2 7 . 0 { \pm } 0 . 2 \ ( 1 . 9 7 \times ) }$ </td><td> $\mathbf { 1 1 8 . 0 { \pm } 1 0 . 4 \left( 1 . 7 9 \times \right) }$ </td><td> $\mathbf { 1 0 3 . 3 { \pm } 6 . 5 \left( 1 . 6 8 \times \right)} $ </td><td> $9 2 . 4 { \pm } 3 . 1 \left( 1 . 5 3 \times \right)$ </td><td>110.2 (1.75×)</td></tr><tr><td colspan="6"> $L l a m a { - } E F T ( L l a m a { - } 3 . I { - } 8 B { - } I n s t r u c t / L l a m a { - } 3 . I { - } 8 B )$ </td></tr><tr><td colspan="6">α = 0.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>69.7 (1.00×)</td></tr><tr><td>CD</td><td> $7 0 . 4 { \pm } 2 . 3 \left( 1 . 0 0 \times \right)$ </td><td> $7 2 . 8 { \pm } 2 . 3 \left( 1 . 0 0 { \times } \right)$ </td><td> $6 7 . 7 { \pm } 1 . 0 \ : ( 1 . 0 0 \times )$   $6 9 . 9 { \pm } 2 . 0 \left( 1 . 0 3 { \times } \right)$ </td><td> $6 7 . 7 { \pm } 3 . 9 \ : ( 1 . 0 0 \times )$   $6 1 . 8 { \pm } 0 . 9 ( 0 . 9 1 \times )$ </td><td>72.1 (1.03×)</td></tr><tr><td>SCD CoS</td><td> $7 9 . 2 { \pm } 5 . 6 \left( 1 . 1 2 \times \right)$ </td><td> $7 7 . 4 { \pm } 2 . 4 \left( 1 . 0 6 \times \right)$   $7 8 . 9 { \pm } 4 . 7 \left( 1 . 0 8 \times \right)$ </td><td> $7 1 . 4 { \pm } 5 . 1 \ : ( 1 . 0 6 \times )$ </td><td> $6 9 . 3 { \pm } 4 . 9 \left( 1 . 0 2 \times \right)$ </td><td>76.1 (1.09×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td> $8 4 . 8 { \pm } 1 . 9 \left( 1 . 2 0 { \times } \right)$   $1 0 5 . 3 { \pm } 1 . 0 \left( 1 . 4 9 \times \right)$ </td><td> $9 1 . 4 { \pm } 0 . 6 \left( 1 . 2 5 \times \right)$ </td><td> $8 6 . 1 { \pm } 1 . 2 \left( 1 . 2 7 \times \right)$ </td><td> $1 1 4 . 8 { \pm } 2 . 4 \ : ( 1 . 7 0 { \times } )$ </td><td>99.4 (1.43×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td> $\mathbf { 1 3 7 . 5 { \pm } 7 . 2 \ ( 1 . 9 5 \times ) }$ </td><td> ${ \bf 1 1 5 . 9 { \pm } 0 . 5 \left( 1 . 5 9 \times \right) }$ </td><td> $9 9 . 9 { \pm } 5 . 8 \left( 1 . 4 8 \times \right)$ </td><td> $\mathbf { 1 1 2 . 2 { \pm } 3 . 1 \ : ( 1 . 6 6 \times ) }$ </td><td>116.4 (1.67×)</td></tr><tr><td colspan="6">α = 0.5</td></tr><tr><td></td><td></td><td> $7 5 . 0 { \pm } 0 . 6 \left( 1 . 0 0 { \times } \right)$ </td><td> $7 2 . 8 { \pm } 1 . 0 \left( 1 . 0 0 { \times } \right)$ </td><td> $6 8 . 4 { \pm } 4 . 4 \left( 1 . 0 0 { \times } \right)$ </td><td>70.6 (1.00×)</td></tr><tr><td>CD SCD</td><td> $6 6 . 0 { \pm } 0 . 9 \left( 1 . 0 0 { \times } \right)$   $7 6 . 4 { \pm } 0 . 8 \ : ( 1 . 1 6 \times )$ </td><td> $6 5 . 1 { \pm } 0 . 9 \ : ( 0 . 8 7 { \times } )$ </td><td> $5 7 . 3 { \pm } 0 . 6 ( 0 . 7 9 { \times } )$ </td><td> $6 3 . 6 { \pm } 0 . 6 \left( 0 . 9 3 { \times } \right)$ </td><td>65.6 (0.93×)</td></tr><tr><td>CoS</td><td> $7 8 . 6 { \pm } 1 . 9 \left( 1 . 1 9 \times \right)$ </td><td> $7 3 . 6 { \pm } 1 . 9 \ : ( 0 . 9 8 \times )$ </td><td> $6 2 . 4 { \pm } 2 . 6 ( 0 . 8 6 \times )$ </td><td> $6 3 . 7 { \pm } 2 . 3 \ : ( 0 . 9 3 \times )$ </td><td>69.6 (0.99×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td> $1 0 4 . 5 { \pm } 1 . 2 \ : ( 1 . 5 8 \times )$ </td><td> $9 2 . 8 { \pm } 1 . 1 \ ( 1 . 2 4 \times )$ </td><td> $8 7 . 2 { \pm } 0 . 9 \left( 1 . 2 0 { \times } \right)$ </td><td> $1 2 0 . 9 { \pm } 2 . 2 \left( 1 . 7 7 \times \right)$ </td><td>101.4 (1.44×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td> ${ \bf 1 4 2 . 9 { \pm } 8 . 9 \ ( 2 . 1 7 \times ) }$ </td><td> $\pm 2 9 . 8 \pm 2 . 5 \left( 1 . 7 3 \times \right)$ </td><td> $\mathbf { 1 1 7 . 5 { \pm } 2 . 1 \ : ( 1 . 6 1 \times ) }$ </td><td> $\mathbf { 1 3 1 . 0 { \pm } 8 . 7 \ : ( 1 . 9 1 \times ) }$ </td><td>130.3 (1.85×)</td></tr><tr><td colspan="6">e 12: Per-dataset sampling breakdown for T = 1 and  $\alpha \in \{ 0 . 1 , 0 . 5 \}$  . Each per-dataset cell reports through</td></tr><tr><td colspan="6">kens/sec as mean ± standard deviation across three independent runs, with speedup over vanilla CD at the sa</td></tr><tr><td colspan="6">'he Avg. column reports the unweighted mean of the per-dataset throughputs; its parenthesized multiplier is</td></tr><tr><td colspan="6">of this mean to the corresponding vanilla-CD mean. The  $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td></tr></table>

Table 11: Per-dataset greedy breakdown for $\alpha \in \{ 0 . 1 , 0 . 5 \}$ . Each cell reports throughput in tokens/sec with reported std and speedup over vanilla CD at the same α. The $\mathrm { D C D } _ { \mathrm { N G R A M } }$ rows use the matched greedy N-gram proposer setting.

<table><tr><td>Model</td><td>Method</td><td>Fixed  $\gamma { = } 5$ </td><td> $\mathbf { B e s t } \gamma$ </td><td>Best speedup</td></tr><tr><td>LLAMA-3</td><td>SCD</td><td>1.13×</td><td>3</td><td>1.32×</td></tr><tr><td>LLAMA-3</td><td>CoS</td><td>1.24×</td><td>2</td><td>1.40×</td></tr><tr><td>LLAMA-3</td><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>1.57×</td><td>5</td><td>1.57×</td></tr><tr><td>QWEN3</td><td>SCD</td><td>1.08×</td><td>2</td><td>1.18×</td></tr><tr><td>QWEN3</td><td>CoS</td><td>1.06×</td><td>1</td><td>1.44×</td></tr><tr><td>QWEN3</td><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>1.64×</td><td>3</td><td>1.66×</td></tr><tr><td>LLAMA-EFT</td><td>SCD</td><td>1.01×</td><td>3</td><td>1.18×</td></tr><tr><td>LLAMA-EFT</td><td>CoS</td><td>1.09×</td><td>1</td><td>1.50×</td></tr><tr><td>LLAMA-EFT</td><td> $\mathbf { D C D _ { E A G L E 3 } }$ </td><td>1.60×</td><td>6</td><td>1.70×</td></tr></table>

Table 13: MMLU draft-length tuning check at $T { = } 0 .$ $\alpha { = } 0 . 1$ . Speedups are relative to vanilla CD for the same model pair. Fixed $\gamma { = } 5$ is the main operating point; Best γ selects the highest-throughput setting from $\gamma \in \{ 1 , \ldots , 7 \}$ using the existing chain-drafting implementation.

<table><tr><td colspan="2">Configuration</td><td colspan="2">Throughput (tok/s)</td></tr><tr><td>Draft Tokens</td><td>k</td><td>GSM8K</td><td>MMLU</td></tr><tr><td colspan="4">Steps = 3</td></tr><tr><td>4</td><td>1</td><td>157.06</td><td>136.84</td></tr><tr><td>6</td><td>3</td><td>163.94</td><td>145.23</td></tr><tr><td>6</td><td>5</td><td>160.91</td><td>149.53</td></tr><tr><td>10</td><td>3</td><td>170.73</td><td>154.89</td></tr><tr><td>10</td><td>5</td><td>176.24</td><td>160.24</td></tr><tr><td>20</td><td>3</td><td>178.13</td><td>163.77</td></tr><tr><td>20</td><td>5</td><td>181.14</td><td>168.51</td></tr><tr><td colspan="4">Steps = 5</td></tr><tr><td>6</td><td>1</td><td>158.44</td><td>134.78</td></tr><tr><td>6</td><td>3</td><td>160.45</td><td>142.24</td></tr><tr><td>6</td><td>5</td><td>158.72</td><td>147.62</td></tr><tr><td>10</td><td>3</td><td>178.15</td><td>156.10</td></tr><tr><td>10</td><td>5</td><td>169.37</td><td>157.24</td></tr><tr><td>20</td><td>3</td><td>188.02</td><td>165.43</td></tr><tr><td>20</td><td>5</td><td>191.84</td><td>170.47</td></tr><tr><td colspan="4">Steps = 8</td></tr><tr><td>9</td><td>1</td><td>144.19</td><td>126.62</td></tr><tr><td>9</td><td>3</td><td>152.58</td><td>137.00</td></tr><tr><td>9</td><td>5</td><td>154.11</td><td>138.35</td></tr><tr><td>10</td><td>3</td><td>156.26</td><td>139.18</td></tr><tr><td>10</td><td>5</td><td>157.35</td><td>139.96</td></tr><tr><td>20</td><td>3</td><td>169.32</td><td>148.55</td></tr><tr><td>20</td><td>5</td><td>173.45</td><td>155.76</td></tr></table>

Table 14: $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ throughput grid at $T { = } 0$ and $\alpha { = } 0 . 1$ . Blocks fix speculative steps; rows vary draft tokens and top-k. Bold marks the block-wise best throughput for each dataset, and shading denotes the configuration that is best on both GSM8K and MMLU within a block.

<table><tr><td>Method</td><td>2</td><td>4</td><td>8</td><td>16</td><td>24</td><td>32</td><td>48</td><td>56</td></tr><tr><td>SCD</td><td>1.29</td><td>1.33</td><td>1.32</td><td>1.28</td><td>1.16</td><td>1.44</td><td>1.21</td><td>0.87</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>1.37</td><td>1.38</td><td>1.19</td><td>0.91</td><td>1.04</td><td>0.99</td><td>1.11</td><td>0.72</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ </td><td>1.56</td><td>1.51</td><td>1.29</td><td>1.48</td><td>1.36</td><td>1.52</td><td>1.32</td><td>1.37</td></tr></table>

Table 15: Configuration-matched multi-request serving speedups on the 1000-question GSM8K sweep. Columns denote the number of concurrent requests at 2, 4, 8, 16, 24, 32, 48, and 56. $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ , SCD, and $\mathrm { D C D } _ { \mathrm { N G R A M } }$ all use 5 speculative steps with chain-style greedy drafting and 6 draft tokens per speculative round. Each entry reports the throughput of the corresponding method divided by the throughput of vanilla CD at the same request count; therefore, the vanilla CD baseline is 1.0 in every column.
<table><tr><td>Method</td><td>MT-Bench Translation</td><td>Summ.</td><td>QA</td><td>Math</td><td>RAG</td><td>Avg. (Speedup)</td></tr><tr><td colspan="7">Llama-3 (8B-Ins / 1B-Ins)</td></tr><tr><td>CD</td><td>95.1</td><td>92.7</td><td>95.1</td><td>94.2</td><td>95.9</td><td>88.2 93.8 (1.00×)</td></tr><tr><td>SCD</td><td>132.6</td><td>105.7</td><td>114.7</td><td>122.6</td><td>151.8 119.3</td><td>125.6 (1.34×)</td></tr><tr><td>CoS</td><td>138.0</td><td>107.2</td><td>116.9 125.6</td><td>159.0</td><td>123.0</td><td>129.7 (1.38×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>118.3</td><td>104.5</td><td>92.9</td><td>107.4 123.8</td><td>96.8</td><td>108.9 (1.16×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>156.3</td><td>109.4</td><td>146.4</td><td>121.7 166.3</td><td>142.3</td><td>142.6 (1.52×)</td></tr><tr><td colspan="7">Qwen3 (8B / 1.7B)</td></tr><tr><td>CD</td><td>65.9</td><td>62.4</td><td>65.8 66.8</td><td>66.9</td><td>65.7</td><td>65.6 (1.00×)</td></tr><tr><td>SCD</td><td>76.2</td><td>75.5</td><td>66.2</td><td>66.0 97.2</td><td>69.8</td><td>75.3 (1.15×)</td></tr><tr><td>CoS</td><td>77.9</td><td>76.6</td><td>66.1</td><td>65.7</td><td>101.8 70.0</td><td>76.6 (1.17×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>77.6</td><td>58.3</td><td>63.1</td><td>71.7</td><td>92.6 72.5</td><td>73.4 (1.12×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>128.0</td><td>99.9</td><td>110.3</td><td>114.6</td><td>152.1 124.7</td><td>122.5 (1.87×)</td></tr></table>

Table 16: Spec-Bench evaluation results (tokens/sec). Spec-Bench (Xia et al., 2024) is a standardized benchmark for speculative decoding with six sub-tasks. All methods use $\alpha { = } 0 . 1 , \gamma { = } 5 .$ , and greedy decoding. Speedup is relative to vanilla CD. The matched $\mathrm { D C D } _ { \mathrm { N G R A M } }$ reference uses the same setting as an appendix row, while $\mathrm { D C D } _ { \mathrm { E A G L E 3 } }$ has the highest average throughput for both model families and leads on most sub-tasks, but not every individual column.
<table><tr><td>Method</td><td>α</td><td>HumanEval</td><td>GSM8K</td><td>MMLU</td><td>CNN/DM</td><td>Avg.</td></tr><tr><td></td><td colspan="6">Llama-3.3-70B-Instruct / Llama-3.1-8B-Instruct, greedy decoding</td></tr><tr><td colspan="7"> $\alpha = 0 . 0$ </td></tr><tr><td>CD</td><td>0.0</td><td>30.4 (1.00×)</td><td>30.2 (1.00×)</td><td>30.2 (1.00×)</td><td>29.8 (1.00×)</td><td>30.2 (1.00×)</td></tr><tr><td>SCD</td><td>0.0</td><td>61.6 (2.02×)</td><td>59.3 (1.96×) 62.6 (2.07×)</td><td>49.5 (1.64×)</td><td>50.8 (1.70×)</td><td>55.3 (1.83×)</td></tr><tr><td>CoS</td><td>0.0</td><td>66.0 (2.17×)</td><td></td><td>50.3 (1.67×)</td><td>53.0 (1.78×)</td><td>58.0 (1.92×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>0.0</td><td>46.8 (1.54×)</td><td>42.0 (1.39×)</td><td>38.0 (1.26×)</td><td>50.2 (1.69×)</td><td>44.3 (1.47×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>0.0</td><td>79.6 (2.62×)</td><td>62.7 (2.07×)</td><td>47.4 (1.57×)</td><td>55.5 (1.86×)</td><td>61.3 (2.03×)</td></tr><tr><td colspan="7">α = 0.1</td></tr><tr><td>CD</td><td>0.1</td><td>30.2 (1.00×)</td><td>30.1 (1.00×)</td><td>30.3 (1.00×)</td><td>29.8 (1.00×)</td><td>30.1 (1.00×)</td></tr><tr><td>SCD</td><td>0.1</td><td>59.3 (1.96×)</td><td>58.2 (1.93×)</td><td>48.7 (1.61×)</td><td>49.8 (1.67×)</td><td>54.0 (1.79×)</td></tr><tr><td>CoS</td><td>0.1</td><td>62.3 (2.06×)</td><td>62.2 (2.07×)</td><td>50.4 (1.66×)</td><td>52.2 (1.75×)</td><td>56.8 (1.89×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>0.1</td><td>46.5 (1.54×)</td><td>40.5 (1.35×)</td><td>38.0 (1.26×)</td><td>50.3 (1.69×)</td><td>43.8 (1.46×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>0.1</td><td>78.1 (2.58×)</td><td>62.0 (2.06×)</td><td>47.5 (1.57×)</td><td>55.3 (1.85×)</td><td>60.7 (2.02×)</td></tr><tr><td colspan="7">α = 0.5</td></tr><tr><td>CD</td><td>0.5</td><td>30.2 (1.00×)</td><td>30.3 (1.00×)</td><td>30.5 (1.00×)</td><td>29.8 (1.00×)</td><td>30.2 (1.00×)</td></tr><tr><td>SCD</td><td>0.5</td><td>52.2 (1.73×)</td><td>57.2 (1.89×)</td><td>45.6 (1.50×)</td><td>49.0 (1.64×)</td><td>51.0 (1.69×)</td></tr><tr><td>CoS</td><td>0.5</td><td>54.2 (1.79×)</td><td>60.0 (1.98×)</td><td>47.4 (1.56×)</td><td>50.0 (1.68×)</td><td>52.9 (1.75×)</td></tr><tr><td> $\mathrm { D C D } _ { \mathrm { N G R A M } }$ </td><td>0.5</td><td>46.1 (1.52×)</td><td>41.2 (1.36×)</td><td>37.3 (1.22×)</td><td>49.5 (1.66×)</td><td>43.5 (1.44×)</td></tr><tr><td> $\mathbf { D C D } _ { \mathbf { E A G L E 3 } }$ </td><td>0.5</td><td></td><td>73.6 (2.44×) 60.8 (2.01×)</td><td>46.6 (1.53×)</td><td>55.1 (1.85×)</td><td>59.1 (1.96×)</td></tr></table>

Table 17: 70B-class extension under greedy decoding. Values report throughput (tokens/sec); speedup in parentheses is relative to vanilla CD at the same α. All results are single-run diagnostics using the Llama-3.3-70B-Instruct expert, the Llama-3.1-8B-Instruct amateur, and either the corresponding EAGLE3 70B drafter or the matched N-gram proposer setting.

<table><tr><td>Method</td><td>Rel.@0.1</td><td>Rel.@0.5</td><td>Drop to 0.5</td></tr><tr><td>SCD</td><td>97.1%</td><td>89.3%</td><td>-10.7%</td></tr><tr><td>CoS</td><td>97.1%</td><td>89.3%</td><td>-10.7%</td></tr><tr><td>DCDNGRAM</td><td>99.2%</td><td>97.0%</td><td>-3.0%</td></tr><tr><td> $\mathbf { D C D _ { E A G L E 3 } }$ </td><td>98.5%</td><td>93.7%</td><td>-6.3%</td></tr></table>

Extra drop vs. $\mathrm { D C D } _ { \mathrm { E A G L E 3 } } \colon$ $\mathrm { S C D } = + 4 . 4$ pts, $\mathrm { C o S } = + 4 . 4$ pts.

Table 18: Relative mean accepted length as contrastive strength increases in the 70B-class extension. Each ratio is normalized to the same method at α=0.0 and averaged across the four datasets under greedy decoding, including the matched N-gram proposer setting. Drop is the relative decrease from α=0.0 to α=0.5.
<table><tr><td>Configuration</td><td>Component</td><td>Model</td></tr><tr><td rowspan="3">LLAMA-3</td><td>Expert</td><td>Llama-3.1-8B-Instruct</td></tr><tr><td>Amateur</td><td>Llama-3.2-1B-Instruct</td></tr><tr><td>Drafter</td><td>EAGLE3-LLaMA3.1- Instruct-8Bª</td></tr><tr><td rowspan="3">QWEN3</td><td>Expert</td><td>Qwen3-8B</td></tr><tr><td>Amateur</td><td>Qwen3-1.7B</td></tr><tr><td>Drafter</td><td>Qwen3-8B-EAGLE3b</td></tr><tr><td rowspan="3">LLAMA-EFT</td><td>Expert</td><td>Llama-3.1-8B-Instruct</td></tr><tr><td>Amateur</td><td>Llama-3.1-8B</td></tr><tr><td>Drafter</td><td>EAGLE3-LLaMA3.1- Instruct-8Bª</td></tr><tr><td rowspan="3">Llama-3 70B Extension</td><td>Expert</td><td>Llama-3.3-70B- Instruct</td></tr><tr><td>Amateur</td><td>Llama-3.1-8B-Instruct</td></tr><tr><td>Drafter</td><td>EAGLE3-LLaMA3.3- Instruct-70Bc</td></tr></table>

<sup>a</sup> yuhuili/EAGLE3-LLaMA3.1-Instruct-8B  
<sup>b</sup> Tengyunw/qwen3\_8b\_eagle3  
<sup>c</sup> yuhuili/EAGLE3-LLaMA3.3-Instruct-70B

Table 19: Off-the-shelf model configurations used in the deployment experiments.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>max_running_ requests</td><td>1</td></tr><tr><td>disable_cuda_ graph</td><td>True</td></tr><tr><td>dtype</td><td>bfloat16</td></tr><tr><td>context_ length</td><td>4096</td></tr><tr><td>speculative- num-steps</td><td>5</td></tr><tr><td>speculative-</td><td>1</td></tr></table>

Table 20: Main single-request inference configuration.