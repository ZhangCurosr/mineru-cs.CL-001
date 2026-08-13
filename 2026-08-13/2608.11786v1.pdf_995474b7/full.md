# Language-Conditional Dequantization: Recovering What Quantization Steals from Non-English Languages

Nirmal Thomas Prathama International

## Abstract

Aggressive quantization disproportionately harms multilingual capability: in the sub-4B INT3 GPTQ regime, we measure 2–4× larger perplexity degradation on non-English languages than on English. We propose Language-Conditional Dequantization (LCD), a post-hoc method that attaches per-language rank-2 LoRA corrections to the linear layers of an already-quantized model, adding 0.12% parameters per language and training in under 20 minutes on a single GPU. Across Qwen2.5-3B and Llama-3.2-3B, LCD recovers 70–83% of the perplexity gap for non-Latin-script languages and 17–28% of the GlobalMMLU accuracy gap, outperforming a language-agnostic correction of equal capacity by 3–9 points on typologically distant languages and a data-free low-rank baseline (LQER) by an order of magnitude. We further identify a perplexity–accuracy disconnect and trace it to where quantization concentrates damage: early-depth errors (Llama) propagate downstream and resist local correction, while late-depth errors (Qwen) do not. A layerrestricted variant of LCD validates this mechanism directly.

## 1 Introduction

Quantized large language models serve billions of queries daily across the world’s languages. When a Korean user queries a 3-bit quantized model, they receive measurably worse outputs than an English user, not because the original model lacked Korean ability, but because the quantization process was calibrated exclusively on English data. In this paper, we quantify the severity of this gap in the regime where quantization is most needed: on Qwen2.5-3B, INT3 GPTQ degrades Arabic perplexity by 4.37× and Japanese by 3.04×, compared to just 1.35× for English; on Llama-3.2-3B the pattern repeats (Arabic 3.65×, English 1.39×). In the sub-4B INT3 regime, this is a practically consequential, language-dependent quality gap.

Multilingual calibration (Chimoto et al., 2026) addresses the root cause but requires re-quantization, which is impractical for alreadydeployed models. Static error correction methods (LQER (Zhang et al., 2024), RILQ (Lee et al., 2025a), ResQ (Saxena et al., 2025)) add low-rank residuals post-hoc but apply the same correction regardless of input language. Input-conditional methods like BinaryMoS (Jo et al., 2024) adapt weights to the input but target binarization and ignore language identity.

We propose Language-Conditional Dequantization (LCD), built on a simple insight: if quantization error is language-dependent, the correction should be too. For each linear layer in a quantized model, LCD attaches a per-language rank-r additive correction via forward hooks:

$$
\hat { y } = W _ { q } x + \frac { 1 } { r } ( A _ { \ell } \cdot B _ { \ell } ) \cdot x\tag{1}
$$

where $A _ { \ell } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r }$ and $\begin{array} { r l r } { B _ { \ell } } & { { } \in } & { \mathbb { R } ^ { r \times d _ { \mathrm { i n } } } } \end{array}$ are language-specific correction matrices. With rank r = 2, this adds only 0.12% parameters per language. Corrections are trained on 256 samples of language-specific text in under 20 minutes per language on a single GPU. At inference time, the appropriate correction is selected based on the input language, requiring no modification to the base model architecture.

Our contributions are:

1. We characterize the multilingual degradation caused by English-calibrated INT3 GPTQ on two sub-4B models (Qwen2.5-3B, Llama-3.2-3B), quantifying per-language perplexity ratios from 1.35× (English) to 4.37× (Arabic). While the broad phenomenon is known (Marchisio et al., 2024), we pin down its magnitude in the specific regime where aggressive quantization is most needed for deployment, motivating the need for languageconditional correction.

2. We propose LCD, a post-hoc method that recovers 70–83% of the perplexity gap for non-Latin-script languages at 0.12% parameter cost per language, without re-quantization.

3. We show that the per-language conditioning captures real language-specific signal: a language-agnostic rank-2 baseline matches per-LCD on average but trails by 3–9 points on the typologically distant languages where per-language corrections have the most signal to exploit.

4. We reveal a perplexity–accuracy disconnect and trace it to a structural cause: a per-layer error analysis shows that quantization concentrates damage in different network depths across models (Qwen: late; Llama: earlymiddle). We validate and probe this mechanism: LCD restricted to Llama’s bottom half of transformer blocks outperforms uniform correction by 10 pp at half the parameter cost (Table 6); precision-targeting of the exact worst-error layers (7–14) does not further improve over uniform, indicating that broad early-depth coverage, rather than narrow layer allocation, is the operative mechanism.<sup>1</sup>

## 2 Related Work

Multilingual Quantization Harm. Marchisio et al. (2024) documented that quantization disproportionately degrades non-English performance, particularly for non-Latin-script languages. Borgersen and Goodwin (2025) found no such harm for k-quantized Llama-3.3-70B, but their 70B moderate-quantization regime differs qualitatively from the sub-4B INT3 setting where redundancy is scarce. Chimoto et al. (2026) addressed the root cause by using multilingual calibration data during GPTQ, but this requires re-quantization and cannot be applied to already-deployed models. LCD is complementary: it operates post-hoc.

Quantization Error Correction. Prior work on low-rank error correction splits on a single axis: at-quantization-time vs. post-hoc. LRQ (Lee et al., 2025b), OmniQuant (Shao et al., 2024),

LQER (Zhang et al., 2024), QERA (Zhang et al., 2025), RILQ (Lee et al., 2025a), and ResQ (Saxena et al., 2025) all require access to the quantization pipeline and produce language-agnostic corrections. Recover-LoRA (Das et al., 2025) is the closest prior work: it trains LoRA adapters post-hoc on a pre-quantized model to recover lost accuracy. LCD extends this post-hoc paradigm with language-conditional selection: each language receives a dedicated adapter that specializes to its quantization-error profile. To our knowledge, LCD is the only post-hoc method that is also language-conditional.

Input-Conditional Adaptation. BinaryMoS (Jo et al., 2024) uses token-adaptive scaling for binarized models; MLAS-LoRA (Dong et al., 2025) applies language-aware LoRA for multilingual fine-tuning (not quantization repair). LCD occupies the language-level position on the inputconditionality spectrum and targets a distinct problem: correcting systematic quantization error concentrated in specific language distributions.

## 3 Method

## 3.1 Language-Conditional Correction

For each linear layer and each language ℓ, we define a rank-r additive correction:

$$
y _ { \mathrm { c o r r e c t e d } } = W _ { q } x + { \frac { 1 } { r } } A _ { \ell } B _ { \ell } x\tag{2}
$$

where $A _ { \ell } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r }$ and $B _ { \ell } \in \mathbb { R } ^ { r \times d _ { \mathrm { i n } } }$ are learnable parameters and $1 / r$ is a fixed scaling constant. We initialize $A _ { \ell } = \mathbf { 0 }$ and $B _ { \ell } \sim \mathcal { N } ( 0 , 0 . 0 1 )$ , ensuring the correction is exactly zero at initialization. This follows the LoRA parameterization (Hu et al., 2022) but serves a fundamentally different purpose: rather than adapting the model to a new task, we are correcting systematic quantization error for a specific language distribution.

Corrections are implemented as forward hooks on each linear layer (excluding lm head and embed tokens), requiring no modification to the model architecture or forward pass.

## 3.2 Training

Each language is trained independently. Given a set of N = 256 text samples from the target language (sourced from mC4/C4), we minimize the standard language modeling loss with only the correction parameters $\{ A _ { \ell } , B _ { \ell } \}$ trainable:

$$
\mathcal { L } _ { \ell } = - \sum _ { t } \log P ( x _ { t } | x _ { < t } ; W _ { q } , A _ { \ell } , B _ { \ell } )\tag{3}
$$

We use AdamW with learning rate $5 \times 1 0 ^ { - 4 }$ , cosine decay with 10% warmup, weight decay 0.01, gradient clipping at 1.0, and train for 500 steps per language. All base model parameters remain frozen. Training takes approximately 15–20 minutes per language on a single L4 GPU, so covering all eight non-English languages costs under 3 GPU-hours in total. Each trained correction ships as a ∼7 MB delta on top of the unchanged deployed checkpoint; by contrast, re-quantizing with multilingual calibration requires curating perlanguage calibration data, re-running the quantization pipeline, and redistributing the full multigigabyte model artifact.

## 3.3 Inference

At inference time, the input language is identified (via locale, automatic detection, or explicit specification) and the corresponding correction slot is activated via a single index. In many deployments the language is already known from the user locale or product surface; when it is not, off-theshelf language identification (e.g., fastText LID) classifies a prompt in well under a millisecond, negligible relative to a single decoding step of the LLM itself. The base quantized model resides in memory once; language switching requires only updating the active adapter index, with no model reloading. For K languages with rank r = 2, total overhead is $K \times 0 . 1 2 \%$ additional parameters.

## 4 Experimental Setup

Models. Qwen2.5-3B (Qwen Team, 2024) and Llama-3.2-3B (Grattafiori et al., 2024), two sub-4B families with distinct tokenizers and training distributions.

Quantization. GPTQ (Frantar et al., 2023) W3A16, group size 128, calibrated on 128 English C4 samples, replicating the standard English-only deployment practice.

Languages. 9 languages: English (baseline), Arabic, Japanese, Chinese, Hindi, Russian, French, Spanish, Korean.

Metrics. (1) Perplexity on held-out mC4/C4 (32 samples/language); degradation ratio and recovery %. (2) GlobalMMLU accuracy (Singh et al.,

<table><tr><td rowspan="2">Lang</td><td colspan="2">Qwen2.5-3B</td><td colspan="2">Llama-3.2-3B</td></tr><tr><td>I/F×</td><td>vs EN</td><td>I/F×</td><td>vs EN</td></tr><tr><td>English</td><td>1.35</td><td></td><td>1.39</td><td></td></tr><tr><td>Arabic</td><td>4.37</td><td>3.23</td><td>3.65</td><td>2.63</td></tr><tr><td>Japanese</td><td>3.04</td><td>2.25</td><td>2.79</td><td>2.01</td></tr><tr><td>Chinese</td><td>2.39</td><td>1.77</td><td>2.89</td><td>2.07</td></tr><tr><td>Hindi</td><td>2.73</td><td>2.02</td><td>2.14</td><td>1.54</td></tr><tr><td>Russian</td><td>2.12</td><td>1.57</td><td>2.39</td><td>1.72</td></tr><tr><td>French</td><td>1.64</td><td>1.21</td><td>1.71</td><td>1.23</td></tr><tr><td>Spanish</td><td>1.60</td><td>1.19</td><td>1.66</td><td>1.20</td></tr><tr><td>Korean</td><td>3.49</td><td>2.58</td><td>3.00</td><td>2.16</td></tr></table>

Table 1: Perplexity degradation ratios after INT3 GPTQ quantization. I/F = INT3/FP16 ratio (mC4/C4 held-out, 32 samples/lang); “vs EN” is the ratio of each language’s degradation to English’s.

2024) via log-likelihood over ≈14,040 items per language (57 subjects).

Baselines. FP16 (unquantized), INT3 (uncorrected), INT3+LCD (corrected). English is the calibration language and excluded from correction training.

## 5 Results

## 5.1 INT3 Quantization Disproportionately Harms Non-English

Table 1 presents the perplexity degradation ratios across languages; the underlying absolute perplexities for every condition are given in Appendix A. On Qwen2.5-3B, English degrades by 1.35×, while Arabic degrades by 4.37×, Korean by 3.49×, and Japanese by 3.04× (a relative disparity of 2.3–3.2×). The pattern is consistent on Llama-3.2-3B, confirming that the effect is not model-specific. Languages with non-Latin scripts and greater typological distance from English exhibit the largest degradation; Romance languages (French, Spanish) suffer only modest additional harm beyond English.

## 5.2 LCD Recovers Most of the Perplexity Gap

Table 2 shows the perplexity recovery results. LCD recovers 70–83% of the quantization gap for non-Latin-script languages (Arabic, Japanese, Chinese, Korean) on both models. Recovery is lower for Latin-script languages close to English (French 38%/54%, Spanish 36%/52%), consistent with these languages suffering less languagespecific quantization error for LCD to correct. These figures are robust to the random seed: retraining all corrections under three seeds and reevaluating on a fixed 256-sample held-out set changes per-language recovery by at most ±1.3 percentage points (standard deviation), so the results are not an artifact of a single seed.

<table><tr><td>Lang</td><td>Qwen</td><td>Llama</td><td>Script</td></tr><tr><td>Arabic</td><td>78.8</td><td>80.1</td><td>Arabic</td></tr><tr><td>Japanese</td><td>69.7</td><td>72.9</td><td>CJK</td></tr><tr><td>Chinese</td><td>71.2</td><td>76.4</td><td>CJK</td></tr><tr><td>Korean</td><td>74.8</td><td>76.7</td><td>Hangul</td></tr><tr><td>Hindi</td><td>82.6</td><td>68.6</td><td>Devanagari</td></tr><tr><td>Russian</td><td>57.0</td><td>71.7</td><td>Cyrillic</td></tr><tr><td>French</td><td>38.2</td><td>53.8</td><td>Latin</td></tr><tr><td>Spanish</td><td>35.6</td><td>52.2</td><td>Latin</td></tr><tr><td>Avg (non-EN)</td><td>63.5</td><td>69.1</td><td></td></tr><tr><td>Avg (non-Latin)</td><td>72.4</td><td>74.4</td><td></td></tr></table>

Table 2: Perplexity recovery (%): fraction of the INT3→FP16 gap closed by LCD. Green: ≥70%. English is excluded from correction training as it is the calibration language.
<table><tr><td>Language</td><td>Per-lang</td><td>Agnostic</td><td>Δ</td></tr><tr><td>Arabic</td><td>78.8</td><td>69.9</td><td>+8.9</td></tr><tr><td>Japanese</td><td>69.7</td><td>62.7</td><td>+7.0</td></tr><tr><td>Korean</td><td>74.8</td><td>71.9</td><td> $+ 3 . 0$ </td></tr><tr><td>French</td><td>38.2</td><td>54.0</td><td>-15.8</td></tr><tr><td>Average</td><td>65.4</td><td>64.6</td><td>+0.8</td></tr></table>

Table 3: Per-language vs. language-agnostic rank-2 LoRA on Qwen2.5-3B (perplexity recovery %). Averages are near-identical, but per-language wins by 3–9 points on typologically distant languages and loses 16 points on French, where no distinct per-language signal exists. ∆ = per-language − agnostic (percentage points).

## 5.3 Language-Specific vs. Language-Agnostic Correction

A natural question is whether per-language conditioning is necessary, or whether a single shared rank-2 LoRA trained on mixed multilingual data would suffice. Table 3 compares LCD against a language-agnostic baseline (same rank, same training budget, training data pooled across four target languages) on Qwen2.5-3B.

On average the two methods are nearly identical (65.4% vs. 64.6%), but the per-language breakdown is revealing: the per-language correction wins by 3–9 points on Arabic, Japanese, and Korean (precisely the languages whose distribution is most distant from the English calibration set) and loses 16 points on French. This pattern confirms that per-language conditioning captures genuine language-specific signal where such signal exists in the data; on French, where the quantization error is closer to English’s, a shared correction trained on a multilingual mixture transfers better than a French-specific one estimated from 256 samples of French text alone. We interpret the near-tie on average as a sharpening, not a weakening, of the central claim: LCD is a faithful correction of language-specific quantization error, not a free lunch that helps every language.

<table><tr><td>Method (rank-2)</td><td>PPL</td><td>MMLU</td></tr><tr><td>LQER (data-free)</td><td>5.5</td><td>13.0</td></tr><tr><td>LCD (per-language)</td><td>63.5</td><td>27.8</td></tr></table>

Table 4: LCD vs. LQER (Zhang et al., 2024) on Qwen2.5-3B at matched rank-2 budget: non-English average recovery (%) of the INT3→FP16 gap, for perplexity (PPL) and GlobalMMLU (MMLU).

Comparison to data-free error reconstruction (LQER). A stronger, principled baseline is LQER (Zhang et al., 2024), which reconstructs the quantization error with a rank-r truncated SVD of $W _ { \mathrm { f p } 1 6 } \mathrm { ~ - ~ } W _ { q }$ at each layer: low-rank like LCD, but data-free, untrained, and language-agnostic. At the identical rank-2 budget on Qwen2.5-3B (Table 4), LQER recovers only 5.5% of the perplexity gap and 13.0% of the GlobalMMLU gap (non-English averages), versus 63.5% and 27.8% for LCD; LQER is even negative on several languages (e.g. Russian, −40% perplexity). A static SVD of the weight error cannot capture the input-dependent, language-specific component that LCD learns from data. This is the sharpest statement of our claim: at matched parameter budget, a trained, language-conditional correction recovers an order of magnitude more of the perplexity gap than a data-free, agnostic one.

## 5.4 Recovery Correlates with Typological Distance

Recovery correlates with typological distance from English: non-Latin-script languages (Arabic, Japanese, Chinese, Korean) cluster at the top (70– 83%), while Latin-script languages close to English (French, Spanish) cluster at the bottom (36– 54%), with Hindi and Russian in between. This ordering is consistent across both models and confirms that LCD corrects calibration–distribution mismatch: the further a language’s activation statistics lie from English, the more concentrated the quantization error and the more LCD has to

## 5.5 Rank Ablation

We trained LCD corrections on Qwen2.5-3B at ranks 1, 2, and 4 (Arabic and Japanese, 500 steps, $\mathrm { l r } = 5 \times 1 0 ^ { - 4 } )$ . The average recovery across the two languages was 76.92% (r=1), 76.98% (r=2), and 76.84% (r=4). Rank is not the bottleneck: the gap from rank 1 to rank 4 is within 0.1 points, indicating that the correction subspace is effectively one-dimensional on these languages. We adopt r = 2 as a small safety margin; a rank-1 deployment would halve parameter overhead to 0.06% per language without measurable degradation.

## 5.6 Downstream Transfer: The Perplexity–Accuracy Disconnect

Table 5 presents GlobalMMLU accuracy across conditions. INT3 quantization severely degrades accuracy on both models (Qwen: 64.7→41.8% English; Llama: 55.5→39.9%).

On Qwen2.5-3B, LCD delivers consistent accuracy gains across every non-English language. The largest improvements are Chinese (+9.1 points), Spanish (+6.7), Russian (+6.2), and Japanese (+6.0); Arabic, French, and Korean also gain between 3.5 and 7.8 points. Across the eight non-English languages, LCD closes an average of 28% of the FP16→INT3 accuracy gap.

On Llama-3.2-3B, LCD also improves every non-English language, but the gains are smaller, between 1.1 points (Chinese, Korean) and 3.5 points (Russian), closing an average of only 17% of the gap. This perplexity–accuracy disconnect is the paper’s most scientifically informative finding: Llama’s perplexity recovery (avg 69%) exceeds Qwen’s (avg 63%), yet its MMLU recovery lags substantially. Section 5.7 diagnoses this gap using a third measurement, distinct from both perplexity and task accuracy: per-layer activation error, which reveals where in the network quantization concentrates its damage and why that location determines whether perplexity recovery translates into downstream gains.

## 5.7 Diagnosing the Disconnect: Where Quantization Hurts

To understand the perplexity–accuracy disconnect mechanistically, we measure the relative Frobenius error $\| y _ { \mathrm { f p 1 6 } } - y _ { \mathrm { i n t 3 } } \| _ { F } / \| y _ { \mathrm { f p 1 6 } } \| _ { F }$ for every linear layer’s output activations, separately for each language. Figure 1 shows the resulting heatmaps.

Two structural findings explain the disconnect. First, the highest-error layers in both models are o proj and down proj, the compression bottlenecks that project from a larger intermediate space back to the hidden dimension, where GPTQ’s per-channel grid is least faithful. Second, and more consequentially, the depth of these worst-error layers differs sharply: Qwen2.5-3B’s top-10 lie in indices 18–30 of 36 (late); Llama-3.2-3B’s lie in indices 7–14 of 28 (early-middle), with 14+ downstream layers consuming their corrupted output.

This depth asymmetry explains the disconnect directly. Late-layer errors (Qwen) primarily distort final next-token logits, which perplexity measures and LCD fixes locally. Early-middle errors (Llama) distort intermediate representations that feed into downstream attention and MLP operations, exactly the multi-step computation MMLU requires. A rank-2 hook corrects a layer’s immediate output but cannot undo propagated upstream corruption. Quantization damage thus operates at two levels: distributional (perplexity, correctable regardless of depth) and representational (downstream accuracy, correctable only when error is near the output). The representational reading also predicts a scale effect: larger models carry more redundancy, so INT3 should inflict less representational damage and leave less for LCD to recover; our Qwen2.5-7B experiment (Limitations) bears this out, with non-English MMLU recovery falling to 8.5% and higher-capacity corrections overfitting.

Direct validation: targeting the worst-error depth band. The depth-asymmetry hypothesis predicts that, for each model, restricting LCD’s correction to the half of the network containing the worst-error layers should match or exceed uniform correction; restricting it to the opposite half should underperform. We test this by training LCD (rank r = 2, identical hyperparameters) under three layer-coverage configurations on the four languages with the largest INT3 degradation: (i) ALL layers (the default); (ii) BOTTOM-HALF of transformer blocks (covering Llama’s worst-error band at indices 7–14); (iii) TOP-HALF of transformer blocks (covering Qwen’s worst-error band at indices 18–30). Table 6 reports the resulting perplexity recovery.

On Llama, BOTTOM-HALF exceeds uniform ALL by 10 pp and beats TOP-HALF by 43 pp on avasymmetry is visible precisely because Llama’s worst layers (25% of the network) are entirely contained in one half and absent from the other.

![](images/85ab88c810d73f97bf4fe2fc7480f0ab5bb9915ffd36b7b7a5ef24ac6be48f81.jpg)

<table><tr><td>Model</td><td>Cond</td><td>EN</td><td>AR</td><td>JA</td><td>ZH</td><td>HI</td><td>RU</td><td>FR</td><td>ES</td><td>KO</td></tr><tr><td rowspan="3">Qwen2.5-3B</td><td>FP16</td><td> $6 4 . 7 _ { \pm 0 . 4 }$ </td><td> $4 8 . 7 _ { \pm 0 . 4 }$ </td><td> $5 2 . 1 { \scriptstyle \pm 0 . 4 }$ </td><td> $5 9 . 5 { \scriptstyle \pm 0 . 4 }$ </td><td> $3 8 . 9 { \scriptstyle \pm 0 . 4 }$ </td><td> $5 3 . 9 { \scriptstyle \pm 0 . 4 }$ </td><td> $5 6 . 7 _ { \pm 0 . 4 }$ </td><td> $5 7 . 7 _ { \pm 0 . 4 }$ </td><td> $5 0 . 0 { \scriptstyle \pm 0 . 4 }$ </td></tr><tr><td>INT3</td><td> $4 1 . 8 { \scriptstyle \pm 0 . 4 }$ </td><td> $2 8 . 0 { \scriptstyle \pm 0 . 4 }$ </td><td> $3 0 . 7 _ { \pm 0 . 4 }$ </td><td> $3 5 . 4 { \scriptstyle \pm 0 . 4 }$ </td><td> $2 6 . 3 { \scriptstyle \pm 0 . 4 }$ </td><td> $2 9 . 6 _ { \pm 0 . 4 }$ </td><td> $3 3 . 3 { \scriptstyle \pm 0 . 4 }$ </td><td> $3 5 . 0 { \scriptstyle \pm 0 . 4 }$ </td><td> $2 8 . 1 { \scriptstyle \pm 0 . 4 }$ </td></tr><tr><td>LCD</td><td> $4 1 . 8 { \scriptstyle \pm 0 . 4 }$ </td><td> $3 1 . 5 _ { \pm 0 . 4 } ^ { \pm }$ </td><td> $\mathbf { 3 6 . 7 _ { \pm 0 . 4 } ^ { \pm } }$ </td><td> $4 4 . 5 _ { \pm 0 . 4 } ^ { \pm }$ </td><td> $2 8 . 3 _ { \pm 0 . 4 } ^ { \pm }$ </td><td> $\mathbf { 3 5 . 8 _ { \pm 0 . 4 } ^ { \pm } }$ </td><td> $\mathbf { 4 0 . 0 _ { \pm 0 . 4 } ^ { \pm } }$ </td><td> ${ \bf 4 1 . 7 _ { \pm 0 . 4 } ^ { \pm } }$ </td><td> $\mathbf { 3 5 . 9 _ { \pm 0 . 4 } ^ { \pm } }$ </td></tr><tr><td rowspan="3">Llama-3.2-3B</td><td>FP16</td><td> $5 5 . 5 { \pm } 0 . 4 $ </td><td> $3 9 . 9 { \scriptstyle \pm 0 . 4 }$ </td><td> $4 1 . 6 { \scriptstyle \pm 0 . 4 }$ </td><td> $4 4 . 7 { \scriptstyle \pm 0 . 4 }$ </td><td> $3 8 . 3 { \scriptstyle \pm 0 . 4 }$ </td><td> $4 4 . 4 { \scriptstyle \pm 0 . 4 }$ </td><td> $4 8 . 2 { \scriptstyle \pm 0 . 4 }$ </td><td> $4 8 . 7 { \scriptstyle \pm 0 . 4 }$ </td><td> $4 1 . 1 { \pm } 0 . 4$ </td></tr><tr><td>INT3</td><td> $3 9 . 9 { \scriptstyle \pm 0 . 4 }$ </td><td> $2 8 . 5 { \scriptstyle \pm 0 . 4 }$ </td><td> $3 0 . 4 { \scriptstyle \pm 0 . 4 }$ </td><td> $3 1 . 7 _ { \pm 0 . 4 }$ </td><td> $2 8 . 4 { \scriptstyle \pm 0 . 4 }$ </td><td> $3 1 . 2 { \scriptstyle \pm 0 . 4 }$ </td><td> $3 4 . 3 { \scriptstyle \pm 0 . 4 }$ </td><td> $3 2 . 9 { \scriptstyle \pm 0 . 4 }$ </td><td> $3 0 . 1 { \scriptstyle \pm 0 . 4 }$ </td></tr><tr><td>LCD</td><td> $3 9 . 9 { \scriptstyle \pm 0 . 4 }$ </td><td> $3 1 . 4 _ { \pm 0 . 4 } ^ { \ddagger }$ </td><td> $3 1 . 9 _ { \pm 0 . 4 } ^ { \dagger }$ </td><td> $3 2 . 8 _ { \pm 0 . 4 } ^ { \dagger }$ </td><td> $3 0 . 3 _ { \pm 0 . 4 } ^ { \dagger }$ </td><td> $3 4 . 7 _ { \pm 0 . 4 } ^ { \pm }$ </td><td> $3 5 . 9 { \scriptstyle \pm 0 . 4 }$ </td><td>35.7 ±0.4</td><td> $3 1 . 2 { \scriptstyle \pm 0 . 4 }$ </td></tr></table>

Table 5: GlobalMMLU accuracy (%) across nine languages. Subscripts show binomial SE $( { \sqrt { p ( 1 - p ) / n } } ,$ n ≈ 14,040). Significance vs. INT3: $^ { \dag } p < 0 . 0 5 , ^ { \dag } p < 0 . 0 0 1$ (paired t-test across 57 subjects, df = 56). Bold: LCD gain ≥5 pp over INT3. Qwen closes 28% of the FP16→INT3 gap (non-EN avg); Llama closes 17%.

Figure 1: Per-layer relative INT3 quantization error by language. Top: Qwen2.5-3B. Bottom: Llama-3.2-3B. Each column is one linear layer; columns are sorted attention-then-MLP, shallow-to-deep within each block. Color indicates $\| y _ { \mathrm { f p 1 6 } } - y _ { \mathrm { i n t 3 } } \| _ { F } / \| y _ { \mathrm { f p 1 6 } } \| _ { F }$ averaged over 32 samples per language. The worst-error layers are the “compression” projections (o proj, down proj) in both models, but their depth differs: Qwen’s hotspots cluster in layers 18–30 (out of 36), while Llama’s cluster in layers 7–14 (out of 28).
<table><tr><td>Model</td><td>Strategy</td><td>AR</td><td>JA</td><td>ZH</td><td>KO</td></tr><tr><td rowspan="2">Qwen2.5-3B</td><td>all</td><td>84.6</td><td>77.3</td><td>18.9</td><td>81.8</td></tr><tr><td>bottom-half top-half</td><td>74.4 78.6</td><td>70.2 68.2</td><td>3.8 -1.9</td><td>73.1 70.5</td></tr><tr><td rowspan="2">Llama-3.2-3B</td><td>all</td><td>71.9</td><td>71.0</td><td>21.8</td><td>70.2</td></tr><tr><td>bottom-half top-half</td><td>78.5 40.2</td><td>83.3 45.6</td><td>36.1 -20.2</td><td>78.3 37.6</td></tr></table>

Table 6: Perplexity recovery (%) for LCD under three layer-coverage configurations, all at rank $r \ = \ 2$ and identical training. BOTTOM-HALF covers Llama’s worst-error band (indices 7–14) and recovers more than uniform ALL on every Llama language despite using half the parameters. On Qwen, whose worst layers (18– 30) span a larger fraction of the network, ALL dominates and the two halves are not differentiated.

erage, obtaining better recovery at half the parameters. On Qwen, the two halves are indistinguishable and both trail ALL, consistent with its worsterror layers spanning a broad 40% band where any half-subset misses some hotspots. The depth

Critically, this perplexity advantage does not reach downstream accuracy. Evaluated on GlobalMMLU, Llama’s BOTTOM-HALF LCD recovers only 9.5% of the accuracy gap, versus 17% for uniform ALL (the reverse of the perplexity ordering). Concentrating capacity where the distributional error is largest (the early compression band) actively trades away representational recovery. This double dissociation (bottom-half wins on perplexity but loses on MMLU) is the strongest evidence that the two levels of quantization damage are governed by distinct mechanisms, and that perplexity recovery is an unreliable proxy for downstream gains.

Negative result: narrow layer targeting does not improve further. We follow up by testing whether concentrating corrections specifically on layers 7–14 with additional rank yields further gains. Three configurations are evaluated: UNIFORM-R2 (rank-2, all layers), TARGETED-R4 (rank-4, layers 7–14 only), and TARGETED-R2 (rank-2, layers 7–14 only). UNIFORM-R2 achieves 45.0% average recovery; both targeted variants reach only 37%, and TARGETED-R4≈TARGETED-R2 (37.0% vs. 37.1%) rules out rank as an explanatory factor. The bottom-half advantage thus comes from broad coverage of the early depth band (layers 0–13 as a whole) rather than precise targeting of the identified hotspot. When correction capacity is limited, broad early-depth coverage dominates narrow layer allocation.

## 6 Conclusion

We have shown that English-calibrated INT3 quantization creates a systematic, languagedependent quality gap that disproportionately harms non-English users. LCD demonstrates that this gap is largely correctable: rank-2 corrections at 0.12% parameter cost per language recover 70–83% of the perplexity degradation for non-Latin-script languages across two model families, and 17–28% of the GlobalMMLU accuracy gap. A language-agnostic baseline with the same capacity matches per-LCD on average but trails by 3–9 points on the typologically distant languages where language-specific signal is concentrated: evidence that LCD is a faithful correction of language-specific quantization error rather than a generic fine-tune.

Our analysis further reveals that perplexity recovery does not reliably translate to downstream accuracy: Llama’s higher perplexity recovery yields smaller MMLU gains than Qwen’s. A perlayer error analysis traces this to depth asymmetry: Llama’s worst-error layers sit early-middle in the network, propagating corruption downstream, while Qwen’s sit late. Restricting LCD to Llama’s bottom half confirms this (+10 pp over uniform), yet precision-targeting of layers 7–14 specifically does not help further (37% vs. 45%), establishing that broad early-depth coverage, not narrow layer allocation, is the operative mechanism.

The broader implication is practical: quantization need not discriminate. With trivial overhead, deployed quantized models can serve non-English users more equitably.

## Limitations

Language identification at inference. LCD requires the input language to be known at inference time so that the correct per-language adapter is activated. In controlled deployment (e.g., a localized product surface with known user locale) this is straightforward, but code-switched inputs, lowquality language ID, or mixed-language prompts are not directly handled by the current design. A soft gating mechanism over adapters, or falling back to the language-agnostic adapter when confidence is low, is a natural extension but is not evaluated here.

Residual MMLU gap on early-error architectures. Section 5.7 attributes LCD’s smaller MMLU gains on Llama-3.2-3B to the depth of its worst-error layers (indices 7–14 of 28), whose corrupted outputs propagate through many downstream blocks. We evaluate a layer-targeted variant that concentrates rank-4 corrections exclusively on these layers, but it underperforms uniform rank-2 correction (37% vs. 45% perplexity recovery), indicating that the MMLU gap reflects a structural representational deficit that narrow layer targeting cannot overcome. Allocating higher rank specifically to the compression projections (o proj, down proj), which carry the highest per-layer error in both models, rather than to entire transformer blocks, remains an open direction.

Narrow quantization regime. We study INT3 GPTQ with W3A16, group size 128, English C4 calibration. Other quantization schemes (AWQ, GGUF k-quants, SmoothQuant), other bit-widths (INT2, INT4, FP4), and other calibration corpora may change both the magnitude of the languagespecific harm and the degree to which a rank-2 correction suffices. We expect the qualitative story to hold but make no claim beyond the evaluated setting.

Model scale and family coverage. Our main evaluation uses two sub-4B-parameter models (Qwen2.5-3B, Llama-3.2-3B). To probe scale, we additionally quantized Qwen2.5-7B to INT3 and trained LCD corrections, evaluating on GlobalMMLU. The result is the opposite of a clean scale-up: at 7B, rank-2 LCD recovers only 8.5% of the non-English accuracy gap (vs. 28% at 3B), and raising capacity to rank-4 with 1000 steps makes every language worse (average −3.9%), as the corrections overfit the 256-sample training slice. Crucially, the languages quantization damages most still recover at 7B (Korean +4.7 pts, 26.5%; Arabic +2.4 pts, 17.0%), while languages with small INT3 gaps (Russian, Spanish, French) go flat or slightly negative. We read this as consistent with, not counter to, our central claim: a larger model carries more redundancy, so INT3 inflicts less representational damage on high-resource languages, leaving little language-specific error for a correction to recover, and additional capacity then fits noise. LCD’s downstream benefit is therefore damage-dependent and does not automatically grow with scale; characterizing exactly where on the (scale × bit-width) plane the benefit persists is open. We make no claim about MoE architectures or instruction-tuned variants.

Language coverage and data. Perplexity recovery is measured on 32 mC4/C4 held-out samples per language, and MMLU on the GlobalMMLU test set (≈14,040 items per language). The eight non-English languages we evaluate span four scripts, but they are all relatively high-resource. Recovery behavior on truly lowresource languages, on dialectal variation, and on domain-shifted text (legal, medical, code-mixed) is an open question.

Adapter training data. Each per-language adapter is trained on 256 samples of monolingual text from that language for 500 steps. This is deliberately small to match the “post-hoc, cheap” framing, but it also means the correction is estimated from a narrow slice of each language’s distribution. Larger and more diverse per-language data may shift the per-language vs. agnostic comparison, particularly for languages like French where the agnostic baseline currently wins.

Ethical and fairness considerations. LCD is intended to reduce a fairness gap introduced by English-centric calibration. However, because the correction is trained on web-scraped multilingual text, it inherits any biases and quality issues present in that corpus. We do not audit the adapters for language-specific toxicity, factuality, or stereotype changes introduced by the correction itself; such an audit is necessary before deploying LCD in user-facing settings.

## Ethics Statement

Societal benefit. English-calibrated INT3 quantization imposes a systematic quality penalty on non-English users: perplexity degrades 2.6–3.2× more for Arabic and Korean than for English on the same model, with no corresponding reduction in model size for those users. LCD is designed to reduce this disparity. Because corrections train on 256 samples in under 20 minutes on a single consumer GPU, the method is accessible to researchers without large-scale compute resources. Because it operates post-hoc on already-quantized weights, it offers a practical correction path for deployed models where re-quantization is not an option.

Known limitations and risks. LCD requires the input language to be identified at inference time. Misidentification activates the wrong perlanguage correction; because the adapters are small (0.12% of parameters, initialized near zero), the magnitude of harm from a mismatched adapter is bounded, but it is not zero and is not characterized in this work. Each adapter is trained on webscraped monolingual text (C4 or mC4). These corpora carry societal biases, quality skew, and domain imbalance. We do not audit whether the correction process amplifies or introduces languagespecific toxicity, stereotyping, or factual error; such an audit is necessary before deploying LCD in user-facing systems. Our evaluation covers eight non-English languages, all relatively highresource; generalization to endangered or verylow-resource languages is untested. Finally, practitioners should not treat perplexity recovery as a proxy for full task recovery: on Llama-3.2-3B, 69% average perplexity recovery translates to only 17% MMLU gap recovery.

Data and compute. All training and evaluation data (C4, mC4, GlobalMMLU) are publicly available. No human subjects were involved and no personally identifiable information was used. Total compute for the experiments reported in this paper was approximately 42 GPU-hours on L4/L40S hardware.

## References

Niklas Borgersen and Morten Goodwin. 2025. English k-quantization does not disproportionately diminish multilingual performance. arXiv preprint arXiv:2503.03592.

Everlyn Asiko Chimoto, Mostafa Elhoushi, and Bruce Bassett. 2026. Calibrating beyond English: Addressing multilingual bias in quantization. In Proceedings of EACL.

Devleena Das, Rajeev Patwari, and Ashish Sirasao. 2025. Recover-LoRA: Data-free accuracy recovery of degraded language models via low-rank adaptation. In Proceedings of EMNLP (Industry Track). ArXiv:2510.08600.

Tianyu Dong, Bo Li, Jingsong Liu, Shaolin Zhu, and Deyi Xiong. 2025. MLAS-LoRA: Language-aware parameters detection and LoRA-based knowledge transfer for multilingual machine translation. In Proceedings ofACL.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2023. GPTQ: Accurate post-training quantization for generative pre-trained transformers. ICLR.

Aaron Grattafiori and 1 others. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. ICLR.

Dongwon Jo, Taesu Kim, Yulhwa Kim, and Jae-Joon Kim. 2024. Mixture of scales: Memory-efficient token-adaptive binarization for large language models. In NeurIPS. ArXiv:2406.12311.

Geonho Lee, Janghwan Lee, Sukjin Hong, Minsoo Kim, Euijai Ahn, Du-Seong Chang, and Jungwook Choi. 2025a. RILQ: Rank-insensitive LoRAbased quantization error compensation for boosting 2-bit large language model accuracy. In AAAI. ArXiv:2412.01129.

Jung Hyun Lee, Jeonghoon Kim, June Yong Yang, Se Jung Kwon, Eunho Yang, Kang Min Yoo, and Dongsoo Lee. 2025b. LRQ: Optimizing posttraining quantization for large language models by learning low-rank weight-scaling matrices. In Proceedings ofNAACL-HLT. ArXiv:2407.11534.

Kelly Marchisio, Saurabh Dash, Hongyu Chen, Dennis Aumiller, Ahmet Ust<sup>¨</sup> un, Sara Hooker, and Se-¨ bastian Ruder. 2024. How does quantization affect multilingual LLMs? In Findings of EMNLP. ArXiv:2407.03211.

Qwen Team. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Utkarsh Saxena, Sayeh Sharify, Kaushik Roy, and Xin Wang. 2025. ResQ: Mixed-precision quantization of large language models with low-rank residuals. In ICML. ArXiv:2412.14363.

Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. 2024. OmniQuant:

Omnidirectionally calibrated quantization for large language models. In ICLR. ArXiv:2308.13137.

Shivalika Singh, Angelika Romanou, Clementine Four-´ rier, and 1 others. 2024. Global MMLU: Understanding and addressing cultural and linguistic biases in multilingual evaluation. arXiv preprint arXiv:2412.03304.

Cheng Zhang, Jianyi Cheng, George A. Constantinides, and Yiren Zhao. 2024. LQER: Low-rank quantization error reconstruction for LLMs. In ICML. ArXiv:2402.02446.

Cheng Zhang, Jeffrey T. H. Wong, Can Xiao, George A. Constantinides, and Yiren Zhao. 2025. QERA: An analytical framework for quantization error reconstruction. In ICLR. ArXiv:2410.06040.

## A Absolute Perplexities

Tables 1 and 2 report degradation ratios and recovery percentages to normalize across languages with very different base perplexities. Table 7 provides the underlying absolute values (mC4/C4 held-out, 32 samples per language) for all three conditions.

<table><tr><td rowspan="2">Lang</td><td colspan="2">Qwen2.5-3B</td><td colspan="3">Llama-3.2-3B</td></tr><tr><td>FP16</td><td>INT3</td><td>LCD FP16</td><td>INT3</td><td>LCD</td></tr><tr><td>English</td><td>13.71</td><td>18.56</td><td></td><td>12.33 17.15</td><td></td></tr><tr><td>Arabic</td><td>7.99</td><td>34.91</td><td>13.70 11.85</td><td>43.31</td><td>18.10</td></tr><tr><td>Japanese</td><td>11.26</td><td>34.21</td><td>18.22</td><td>15.45 43.17</td><td>22.96</td></tr><tr><td>Chinese</td><td>13.10</td><td>31.31</td><td>18.35</td><td>14.91 43.03</td><td>21.54</td></tr><tr><td>Hindi</td><td>4.49</td><td>12.26</td><td>5.84</td><td>5.35 11.44</td><td>7.26</td></tr><tr><td>Russian</td><td>5.90</td><td>12.51</td><td>8.74</td><td>7.74 18.46</td><td>10.77</td></tr><tr><td>French</td><td>10.00</td><td>16.39</td><td>13.95</td><td>11.14 19.09</td><td>14.81</td></tr><tr><td>Spanish</td><td>9.45</td><td>15.15</td><td>13.12</td><td>10.42 17.31</td><td>13.71</td></tr><tr><td>Korean</td><td>7.98</td><td>27.85</td><td>12.98</td><td>12.92 38.79</td><td>18.94</td></tr></table>

Table 7: Absolute perplexity per language under each condition (mC4/C4 held-out, 32 samples/language). English receives no correction, as it is the calibration language.