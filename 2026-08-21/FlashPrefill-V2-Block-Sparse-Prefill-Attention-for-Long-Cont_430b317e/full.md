# FlashPrefill V2: Block-Sparse Prefill Attention for Long-Context LLM Serving

Qihang Fan<sup>1,2,3,∗</sup>, Huaibo Huang<sup>1,2,†</sup>, Zhiying Wu<sup>3</sup>,

Bingning Wang<sup>3,‡</sup>, Ran He<sup>1,2</sup>

<sup>1</sup>MAIS&NLPR, CASIA <sup>2</sup>UCAS <sup>3</sup>WeChat, Tencent

<sup>§</sup> https://github.com/qhfan/FlashPrefillv2

## Abstract

Long-context modeling is a pivotal capability for Large Language Models, yet the quadratic complexity of attention remains a critical bottleneck, particularly during the compute-intensive prefilling phase. Our previous work, FlashPrefill [8], mitigates this cost through instantaneous pattern discovery and max-based dynamic thresholding; however, it remains an algorithmic prototype that is still distant from production deployment. In this paper, we present FlashPrefill V2, which evolves FlashPrefill from a prototype toward practical long-context serving along three dimensions. First, we introduce a mean correction term that effectively suppresses the approximation error, keeping performance degradation manageable even at extreme sparsity levels. Second, we redesign the sparse attention operator with PackGQA memory access, warp specialization, and pingpong pipelining, fully aligning with the latest FlashAttention-3/4 implementations [22, 38] and supporting FP8 inference to meet practical quantization requirements. Third, FlashPrefill V2 natively supports paged KV cache and continuous batching, allowing integration as an attention backend in modern inference frameworks such as SGLang [42]. Extensive evaluations on NVIDIA H20 GPUs—among the most widely deployed inference accelerators—demonstrate that FlashPrefill V2 delivers up to 47.26× and 27.19× speedups over FlashAttention-2 at 128K context length under FP8 and BF16 precision, respectively, and, in FP8, still achieves a 30.49× speedup against an FA3/4-aligned dense baseline.

## 1. Introduction

Recent years have witnessed the rapid evolution of Large Language Models (LLMs) [9, 27, 28]. However, the quadratic complexity of self-attention incurs prohibitive overhead on long-context sequences [29], a bottleneck that is particularly salient during the compute-intensive prefill stage. To mitigate this cost, various sparse attention mechanisms have been proposed [11, 14, 35]: they coarsely estimate attention scores to identify salient blocks with Top-�/Top-� selection, and restrict fine-grained computation to these critical tokens.

![](images/723445204850c385f2e30fe257693e6639f06d400e6b9fbc4eca5aa902660c91.jpg)

![](images/a2b080760774ec4e3f16ce21d82013229fc091974ab3158b8c8e5493218d9141.jpg)

![](images/7c606621e994552c6a5d2fd934d0b1df52779a36b507a92851691fcf0faf7989.jpg)  
Figure 1 | FlashPrefill V2 evolves FlashPrefill [8] from an algorithmic prototype toward practical long-context serving along three dimensions: (1) a mean correction term that preserves model accuracy under extreme sparsity; (2) an FA3/4-aligned sparse attention kernel with PackGQA memory access, warp specialization, pingpong pipelining, and FP8 support; and (3) native compatibility with paged KV cache and continuous batching for integration as an attention backend in modern serving frameworks such as SGLang [42].

Our previous work, FlashPrefill [8], pushes this line of research further with instantaneous pattern discovery and max-based dynamic thresholding, eliminating the sorting latency of Top-�/Top-� selection. Nevertheless, it remains an algorithmic prototype that is still distant from production deployment: its accuracy can degrade uncontrollably under aggressive sparsity; its kernel is built upon FlashAttention-2 [5] and lags behind state-of-the-art dense implementations such as FlashAttention-3/4 [22, 38], a gap that is particularly pronounced on Hopper-architecture GPUs where TMA and asynchronous pipelines dominate kernel efficiency; and its contiguous KV layout is incompatible with the paged KV cache and continuous batching that underpin modern serving frameworks [13, 42].

In this paper, we present FlashPrefill V2, which closes these gaps as illustrated in Fig. 1. (1) We introduce a mean correction term that compensates pruned blocks with their pooled K/V statistics inside the attention computation, keeping accuracy degradation manageable even at extreme sparsity levels. (2) We redesign the sparse attention operator for Hopper-architecture GPUs with PackGQA memory access, warp specialization, and pingpong pipelining, fully aligning with the latest FlashAttention-3/4 implementations and supporting FP8 precision for quantized deployment. (3) FlashPrefill V2 natively supports paged KV cache and continuous batching, and can be integrated as an attention backend in modern inference frameworks such as SGLang.

As shown in Fig. 2, on NVIDIA H20, one of the most widely deployed inference GPUs and a representative of the Hopper architecture, FlashPrefill V2 attains a 27.19× speedup over FlashAttention-2 at a 128K context length under BF16 precision, surpassing all competing sparse operators, while its FP8 variant further reaches 47.26×. Even against the much stronger FA3/4- aligned dense baseline, FlashPrefill V2 still delivers speedups of 17.54× (BF16) and 30.49× (FP8), and maintains acceleration at sequence lengths as short as 4K.

Our contributions can be summarized as follows:

• A mean correction term for block-wise score estimation that ensures controlled accuracy degradation even under extreme sparsity.

• A redesigned sparse attention operator featuring PackGQA memory access, warp specialization, and pingpong pipelining, fully aligned with FlashAttention-3/4 and extended to FP8 inference, attaining up to 47.26× speedup over FlashAttention-2 at 128K context.

![](images/1605cee74156c2715c80a214fa381ffd2a228bb7028267df95f1923fddd16391.jpg)  
Figure 2 | Speedup of various attention operators relative to FlashAttention-2 [5] on NVIDIA H20 GPUs, including an FA3/4-aligned dense baseline [22, 38]. All results are measured with a batch size of 4. FlashPrefill V2 exhibits a dominant advantage, particularly in long-context scenarios, and its FP8 variant further amplifies the gains.

• Native compatibility with paged KV cache and continuous batching, demonstrated through integration into SGLang with up to 4.8× lower end-to-end time-to-first-token at 128K with marginal accuracy loss.

## 2. Related Works

Sparse Attention. Reducing the quadratic cost of attention via sparsity has been extensively studied. Early designs fix the sparse layout in advance, via local windows with global tokens [3, 39] or hashing- and clustering-based routing [12, 30], or evict KV entries under a preset budget [4, 16, 31, 32, 40]. Learnable variants integrate sparsity into training or fine-tuning, coupling the pattern with the weights [18, 34, 37, 41]. Training-free methods instead estimate which regions of the attention map are worth computing and skip the rest at inference [11, 14, 35], extend to query-aware KV selection during decoding [21, 23, 25, 33, 44], and to multimodal inputs [17]. A recent variant compensates pruned blocks with surrogate contributions [15], and UniPrefill [7] lifts block-wise sparsification to token-level computation for hybrid architectures. FlashPrefill V2 belongs to the training-free category: compared with prior training-free methods, it further emphasizes controlled accuracy under extreme sparsity and system-level deployability, rather than treating the sparse pattern as the sole design axis. A complementary line replaces softmax attention altogether with linear or recurrent formulations [19, 24], e.g., rectifying its neglect of query magnitude [6].

Efficient Attention Kernels. Orthogonal to sparsity, a parallel line of work accelerates dense attention through careful hardware-aware kernel design. FlashAttention-2 reformulates the computation around better parallelism and work partitioning [5], while FlashAttention-3 exploits asynchrony and low-precision arithmetic on Hopper GPUs [22], and FlashAttention-4 co-designs the algorithm with kernel pipelining to accommodate asymmetric hardware scaling [38]. Our sparse operator inherits this design philosophy: built upon CUTLASS/CuTe with TMA and warp-specialized producer-consumer pipelines, it adopts PackGQA memory access so that block sparsity translates into wall-clock speedup rather than being offset by kernel inefficiency, and it further supports FP8 execution.

LLM Serving Systems. Production-grade inference increasingly relies on system-level innovations in memory management and scheduling. vLLM popularized paged KV cache management with PagedAttention [13], SGLang optimizes the execution of structured language model programs [42], and disaggregated architectures further separate prefill from decoding to satisfy latency objectives [20, 43]; continuous batching across concurrent requests has since become a standard technique for maximizing throughput. However, most research-oriented sparse attention kernels assume a contiguous KV layout, which prevents direct adoption in these frameworks. FlashPrefill V2 is instead designed around paged KV cache and continuous batching from the outset, and can be integrated as an attention backend in modern serving engines.

![](images/3182f73550618b683d1f01e42ac8b7d39e4ba4c69ce3f73e7a2eb3ecd5f8ec88.jpg)  
Figure 3 | The FlashPrefill V2 prefill pipeline. Stage 1 packs queries with PackGQA, pools block-mean statistics, and produces a CSR index of selected blocks in a single fused pass; Stage 2 runs the warp-specialized sparse attention kernel over the selected blocks (exact path) while the unselected blocks are compensated through the mean correction path (Sec. 3.2), and the two streams merge inside the online softmax.

## 3. Method

## 3.1. Preliminaries: The FlashPrefill Framework

We first recap the core computation of FlashPrefill [8], upon which FlashPrefill V2 is built. Consider the prefilling stage with � tokens, where $Q , K , V \in \mathbb { R } ^ { L \times d }$ denote the query, key, and value matrices of one attention head, and the causal output is $O = \mathrm { s o f t m a x } ( Q K ^ { \top } / \sqrt { d } + { \cal M } ) V$ FlashPrefill partitions � and � into blocks of size � and accelerates prefilling in two stages.

Block-level Score Estimation. FlashPrefill identifies the salient blocks with uniformly distributed probe queries, which simultaneously resolve vertical, slash, and block-sparse patterns. Each key block $\mathcal { B } _ { J }$ is represented by its mean-pooled key $\begin{array} { r } { \bar { k } _ { J } = \frac { 1 } { B } \sum _ { k _ { i } \in \mathcal { B } _ { J } } } \end{array}$ ��. Since exp $( q \cdot \bar { k } _ { J } )$ is the geometric mean of the per-token scores $\{ \exp ( q \cdot k _ { i } ) \}$ , by the $\mathrm { A M - G M }$ inequality it lower-bounds the block’s true contribution $\begin{array} { r } { \frac { 1 } { B } \sum _ { i } \exp ( q \cdot \hat { k _ { i } } ) } \end{array}$ , while the low intra-block variance of attention distributions keeps the cross-block ranking preserved. To avoid materializing the $L \times ( L / B )$ score matrix, a fused kernel computes, for each query tile � and key block $J ,$

$$
m _ { I , J } = \operatorname * { m a x } _ { q _ { i } \in I } ( q _ { i } \cdot \bar { k } _ { J } ) , \qquad S _ { I , J } = \sum _ { q _ { i } \in I } \exp ( q _ { i } \cdot \bar { k } _ { J } - m _ { I , J } ) ,\tag{1}
$$

followed by a global rescaling with $\ M _ { I } = \operatorname* { m a x } _ { J } m _ { I , J }$ and row normalization:

$$
\mathrm { S c o r e } _ { I , J } = \frac { S _ { I , J } \cdot \exp ( m _ { I , J } - M _ { I } ) } { \sum _ { J ^ { \prime } } S _ { I , J ^ { \prime } } \cdot \exp ( m _ { I , J ^ { \prime } } - M _ { I } ) + \varepsilon } ,\tag{2}
$$

reducing the memory footprint from $O ( L ^ { 2 } / B )$ to $O ( ( L / B ) ^ { 2 } )$

Max-based Dynamic Thresholding. Instead of Top-�/Top-� selection, which requires global sorting or cumulative summation, FlashPrefill derives the pruning threshold for each query

![](images/3aa084620620bb78a33bd413605499c53dd05152ad1c75da08ec0dc9bad0f898.jpg)  
Figure 4 | Mean correction. Selected blocks (red) are computed exactly, while each pruned block is pooled into its mean statistics $( \bar { k } _ { J } , \bar { \upsilon } _ { J } )$ and contributes a surrogate term $| \mathcal { B } _ { J } | e ^ { \bar { s } _ { J } } ( \bar { \nu } _ { J } , 1 )$ to the softmax numerator and denominator, recovering the discarded probability mass without per-token computation.

block � from its peak score,

$$
\mathrm { \ t h r e s h } _ { I } = \alpha \cdot \mathrm { m a x } { \cal S } \mathrm { c o r e } _ { I , J } ,\tag{3}
$$

and retains every key block with Score�<sub>,</sub>� ≥ thresh�, together with the first sink blocks, a local sliding-window band, and the most recent blocks, subject to the causal constraint. Block-sparse attention is then executed only over the selected blocks.

## 3.2. Mean Correction under Extreme Sparsity

For a query � with per-token logits $s _ { i } = q \cdot k _ { i } / { \sqrt { d } } ,$ , let S denote the selected block set. FlashPrefill approximates the exact output by truncating both sums of the softmax to S:

$$
O = \frac { \sum _ { J } \sum _ { i \in \mathcal { B } _ { J } } e ^ { s _ { i } } \upsilon _ { i } } { \sum _ { J } \sum _ { i \in \mathcal { B } _ { J } } e ^ { s _ { i } } } \approx \frac { \sum _ { J \in S } \sum _ { i \in \mathcal { B } _ { J } } e ^ { s _ { i } } \upsilon _ { i } } { \sum _ { J \in S } \sum _ { i \in \mathcal { B } _ { J } } e ^ { s _ { i } } } .\tag{4}
$$

Under extreme sparsity, however, the discarded mass $\textstyle \sum _ { J \not \in S } \sum _ { i \in { \mathcal { B } } _ { J } } e ^ { s _ { i } }$ can no longer be neglected. We recover it with a zero-order mean correction (a surrogate-contribution idea also adopted in [15]), as illustrated in Fig. 4: for each key block �, we precompute

$$
\bar { k } _ { J } = \frac { 1 } { | \mathcal { B } _ { J } | } \sum _ { i \in \mathcal { B } _ { J } } k _ { i } , \qquad \bar { \nu } _ { J } = \frac { 1 } { | \mathcal { B } _ { J } | } \sum _ { i \in \mathcal { B } _ { J } } \nu _ { i } , \qquad \bar { s } _ { J } = q \cdot \bar { k } _ { J } / \sqrt { d } ,\tag{5}
$$

and the corrected output reads

$$
\hat { O } = \frac { { \displaystyle \sum _ { J \in S } \sum _ { i \in \mathcal { B } _ { J } } e ^ { s _ { i } } \upsilon _ { i } \ } + \ \sum _ { J \notin S } { \left| \mathcal { B } _ { J } \right| e ^ { \bar { s } _ { J } } \bar { \upsilon } _ { J } } }  { { \displaystyle \sum _ { J \in S } \sum _ { i \in \mathcal { B } _ { J } } e ^ { s _ { i } } \ } + \ \sum _ { J \notin S } { \left| \mathcal { B } _ { J } \right| e ^ { \bar { s } _ { J } } } } .\tag{6}
$$

Approximation error. Writing $\delta _ { i } = s _ { i } - \bar { s } _ { J }$ with $\textstyle \sum _ { i } \delta _ { i } = 0$ and denoting within-block averages by (·), the exact mass and weighted value of block � expand around $\overline { { s } } _ { J }$ as

$$
\sum _ { i \in \mathcal { B } _ { J } } e ^ { s _ { i } } = \left| \mathcal { B } _ { J } \right| e ^ { \bar { s } _ { J } } \Big ( 1 + \frac { \overline { { \delta ^ { 2 } } } } { 2 } + O ( \overline { { \delta ^ { 3 } } } ) \Big ) , \qquad \sum _ { i \in \mathcal { B } _ { J } } e ^ { s _ { i } } \upsilon _ { i } = \left| \mathcal { B } _ { J } \right| e ^ { \bar { s } _ { J } } \Big ( \bar { \upsilon } _ { J } + \overline { { \delta \nu } } + O ( \overline { { \delta ^ { 2 } \upsilon } } ) \Big ) ,\tag{7}
$$

where $\begin{array} { r } { \overline { { \delta \nu } } = \frac { 1 } { | \mathcal { B } _ { I } | } \sum _ { i } \delta _ { i } ( \nu _ { i } - \bar { \nu } _ { J } ) } \end{array}$ is the within-block covariance between logit deviations and values: the mass surrogate is second-order accurate, whereas the numerator surrogate retains a firstorder covariance term. With block mass $\begin{array} { r } { { w _ { J } } = \sum _ { i \in \mathcal { B } _ { I } } { e ^ { s _ { i } } } } \end{array}$ , total mass $\begin{array} { r } { D = \sum _ { J } w _ { J , } } \end{array}$ , and exact block mean $\begin{array} { r } { \mu _ { J } = w _ { J } ^ { - 1 } \sum _ { i \in \mathcal { B } _ { J } } e ^ { s _ { i } } \nu _ { i } , } \end{array}$ substituting the surrogate for a pruned block shifts the output by

$$
\Delta _ { J } \ = \ \frac { w _ { J } } { D } \big ( \mu _ { J } - \bar { \nu } _ { J } \big ) + O \bigg ( \frac { w _ { J } } { D } \overline { { { \delta ^ { 2 } } } } \bigg ) , \qquad \mu _ { J } - \bar { \nu } _ { J } \ = \ \overline { { { \delta \nu } } } + O \big ( \overline { { { \delta ^ { 2 } \nu } } } \big ) .\tag{8}
$$

Eq. 8 is controlled by three bounds. (i) Max-based thresholding (Sec. 3.1) caps the mass share of every pruned block, $w _ { J } / D \lesssim \alpha ,$ so the total shift is a small-share signed sum,

$$
\bigg \| \sum _ { J \notin { S } } \Delta _ { J } \bigg \| \leq \rho _ { \mathrm { p r } } \cdot \operatorname* { m a x } _ { J \notin { S } } \Big \| \mu _ { J } - \bar { \nu } _ { J } \Big \| , \qquad \rho _ { \mathrm { p r } } = \sum _ { J \notin { S } } \frac { w _ { J } } { D } .\tag{9}
$$

(ii) The per-block deviation is a covariance, vanishing under within-block independence and bounded by dispersion via Cauchy–Schwarz,

$$
\overline { { \delta \nu } } = 0 \mathrm { i f } \delta \perp \big ( \nu - \bar { \nu } _ { J } \big ) , \qquad \left\| \overline { { \delta \nu } } \right\| \ \leq \ \delta _ { \mathrm { r m s } } \cdot \sigma _ { \nu } ,\tag{10}
$$

where $\sigma _ { \nu }$ is the within-block value dispersion, itself small under the low intra-block variation that justifies block-mean score estimation (Sec. 3.1). (iii) Relative to dropping the block outright, $\begin{array} { r } { \Delta _ { J } ^ { \mathrm { n o n e } } = \frac { w _ { J } } { D } \big ( \mu _ { J } - O \big ) } \end{array}$ , the correction cancels the zeroth order and retains only the covariance:

$$
\frac { \| \Delta _ { J } \| } { \| \Delta _ { J } ^ { \mathrm { n o n e } } \| } \ = \ \frac { \| \mu _ { J } - { \bar { \nu } _ { J } } \| } { \| \mu _ { J } - O \| } \ \sim \ O ( \delta _ { \mathrm { r m s } } ) ,\tag{11}
$$

since $\mu _ { J } - O$ is an inter-block deviation while $\mu _ { J } - \bar { \nu } _ { J }$ is an intra-block one. Salient blocks always follow the exact path, so only the small-�� tail enters Eq. 9. Empirically (Tab. 8), removing the correction costs up to 6.2 points at 128K in FP8, while the corrected pipeline stays within 1.5 points of full attention even at 128K in BF16.

Implementation. As illustrated by the green correction path in Fig. 3, only blocks fully visible to the whole query tile are corrected; blocks intersecting the diagonal band are covered by the exact sink/window/recent selections. Within the kernel, each corrected block enters the pipeline as one extra iteration whose score is shifted by log $| \mathcal { B } _ { J } | ,$ , so that a single mean vector represents $\lvert \mathcal { B } _ { J } \rvert$ tokens in both the numerator and the denominator of Eq. 6, reusing the same MMA pipeline and online-softmax rescaling without any additional kernel launch. Concretely, a two-pointer scan over the selected-block list aggregates the correction targets of each chunk into a bitmask, todo = valid & ¬ selected, and chunks in which every block is selected are skipped identically by the producer and the consumer, so the pipeline stays in lockstep without extra synchronization.

## 3.3. Hopper-Aligned Sparse Attention Operator

The V1 sparse kernel inherits the execution model of FlashAttention-2, whose tile-based synchronous pipeline under-utilizes Hopper GPUs: the Tensor Memory Accelerator (TMA) and asynchronous warpgroup MMAs (wgmma) only reach full throughput when data movement and computation are decoupled. We therefore redesign the operator on top of a CUT-LASS/CuTe warp-specialized pipeline (Stage 2 of Fig. 3). Our implementation builds upon the Hopper-specific techniques established by FlashAttention-3/4 themselves [22, 38]—TMAbased asynchronous data movement, warp-specialized producer-consumer pipelines, and intra-warpgroup GEMM-softmax overlap—and our contribution lies in extending this execution model to block-sparse computation (index-driven traversal, correction chunk injection) and to the GQA/paged/FP8 regimes that deployment requires, as detailed below.

PackGQA Memory Layout. Under grouped-query attention with ratio $g = H _ { q } / H _ { k \nu } , \mathfrak { a }$ naive mapping assigns one thread block to each (query head, �-tile) pair, so that $g$ thread blocks repeatedly load the same KV block. We instead reshape the query matrix as

$$
{ \cal Q } \in \mathbb R ^ { L \times H _ { q } \times d } \longrightarrow { \cal Q } ^ { \prime } \in \mathbb R ^ { ( g \cdot L ) \times H _ { k \nu } \times d } ,\tag{12}
$$

where the packed row index � maps to position and head via

$$
\mathrm { p o s } ( r ) = \lfloor r / g \rfloor , \qquad h ( r ) = h _ { k \nu } \cdot g + ( r \bmod g ) ,\tag{13}
$$

implemented with a single fast divmod per row. Each �-tile of $Q ^ { \prime }$ covers �������/� positions of all � heads of one KV group, yielding three properties: (i) each staged KV block is consumed by all rows of the tile, so group-wide sharing is guaranteed in shared memory and never relies on cross-CTA L2 retention; (ii) the sparse index of Sec. 3.1 is defined per KV head, shrinking the index metadata by a factor of �; and (iii) tiles stay fully occupied for any $L \colon$ the wasted-row fraction is ( ⌈�/�������⌉������� − �)/( ⌈�/�������⌉�������) per head under the naive mapping but only ( ⌈��/�������⌉������� − ��)/( ⌈��/�������⌉�������) once per group under packing, which for $L \leq k B l o c k M / g$ (e.g., decode) cuts KV stagings by up to the full factor of �. Q is staged with vectorized cp.async loads, with threads of the same row sharing one warp to amortize page-table lookups.

Warp-Specialized Producer-Consumer Pipeline. The persistent kernel partitions its warpgroups into one producer warpgroup issuing asynchronous TMA/cp.async loads of $\mathrm { K } / \mathrm { V }$ (and of the correction chunks of Sec. 3.2) into a multi-stage shared-memory pipeline, and two consumer warpgroups executing wgmma on staged tiles, with registers reallocated between the two roles (warpgroup\_reg\_dealloc/alloc). On top of this, GEMM and softmax proceed in a pingpong fashion within each consumer warpgroup: each step issues the QK GEMM of block �, the PV GEMM of block �+1, and the online softmax of block �,

$$
\underbrace { S ^ { ( n ) } = Q K _ { ( n ) } ^ { \top } } _ { \mathrm { G E M M } _ { 0 } } ~ \lVert ~ \underbrace { O + = P ^ { ( n + 1 ) } V _ { ( n + 1 ) } } _ { \mathrm { G E M M } _ { 1 } } ~ \rVert ~ \underbrace { \tilde { P } ^ { ( n ) } = \mathrm { o n l i n e - s o f t m a x } ( S ^ { ( n ) } ) } _ { \mathrm { s o f t m a x } } ,\tag{14}
$$

so that MMA and non-MMA (exponentiation, rescaling) units overlap instead of stalling each other. This matches the intra-warpgroup overlap schedule of state-of-the-art dense implementations [22, 38], which our dense path replicates to provide a strong baseline.

FP8 Execution. With per-tensor scales $c _ { \boldsymbol { q } } , c _ { \boldsymbol { k } } , c _ { \nu . }$ , FP8-e4m3 operands are dequantized on the fly inside the online softmax:

$$
S = c _ { q } c _ { k } \cdot ( \tilde { Q } \tilde { K } ^ { \top } ) , \qquad \tilde { P } = \left\lfloor 2 ^ { 8 } \cdot \exp ( S - m ) \right\rceil _ { \mathrm { e 4 m 3 } } \in [ 0 , 2 5 6 ] , \qquad O = c _ { v } \cdot \frac { \sum _ { t } \tilde { P } ^ { ( t ) } \tilde { V } ^ { ( t ) } } { \sum _ { t } \tilde { P } ^ { ( t ) } } ,\tag{15}
$$

where $\lfloor \cdot \rceil _ { \mathrm { e } 4 \mathrm { m } 3 }$ denotes FP8 quantization and � the running row maximum. The $2 ^ { 8 }$ offset maps probabilities onto [0, 256], fully utilizing the e4m3 dynamic range; it cancels in the softmax ratio and is recovered only in the log-sum-exp at finalize.

The second GEMM requires special handling because FP8 wgmma mandates a K-major (transposed) layout for the B operand, which constrains how $\tilde { V }$ is fed:

• Column-major $\tilde { V }$ already satisfies the layout and is consumed with zero copying; the probability fragment $\tilde { P }$ is instead re-arranged into the wgmma A-operand register layout purely in registers via warp shuffles and byte permutation, and the output columns are un-permuted symmetrically in the epilogue:

$$
\tilde { P } \xrightarrow { \mathrm { \ s h f l { + } b y t e \mathrm { - } p e r m } } \tilde { P } ^ { \prime } , \qquad O ^ { \prime } = \tilde { P } ^ { \prime } \tilde { V } \xrightarrow { \mathrm { \ e p i l o g u e } } O .\tag{16}
$$

• Row-major $\tilde { V }$ is transposed on the fly by the producer inside shared memory with an LDSM.T→byte-permute→STSM sequence; to avoid bank conflicts we permute the columns of $\tilde { V }$ during this transpose and restore the column order of � in the epilogue, rather than transposing conflict-free at full price.

In both cases the layout conversions involve no global-memory traffic, and the PV GEMM consumes �<sup>˜</sup> directly from registers (RS-mode wgmma).

Index-Driven Sparse Traversal. The selection result is stored as a CSR index: for each (batch, head, tile) segment $g$ with offsets $\mathrm { c u } [ g ]$ and $T = \mathtt { c u } [ g + 1 ] - \mathtt { c u } [ g ]$ , the ascending index list $\{ \mathsf { i d } \mathbf { \boldsymbol { x } } [ \mathsf { c u } [ g ] + i \mathsf { \bar { l } } \} _ { i = 0 } ^ { T - 1 }$ defines the iterated block sequence

$$
n ^ { ( t ) } = \operatorname { i d } \mathbf { x } \left[ \mathbf { c u } [ g ] + T - 1 - t \right] , \qquad t = 0 , \ldots , T - 1 ,\tag{17}
$$

traversed backwards to match the descending order of the dense schedule. Producer and consumer evaluate the same sequence $\{ n ^ { ( t ) } \}$ independently, so their control flows are identical by construction and the pipeline advances in lockstep without any synchronization for block skipping; the next index $\mathbf { \hat { \Pi } } _ { n } ^ { \bullet ( t + 1 ) }$ is prefetched while the load and GEMM of $n ^ { ( t ) }$ are still in flight, keeping index resolution off the critical path. Under KV splitting, split � of � receives the contiguous subsequence

$$
t \in \Big [ u \cdot \lceil T / U \rceil , \ \operatorname* { m i n } \big ( ( u { + } 1 ) \lceil T / U \rceil , T \big ) \Big ) ,\tag{18}
$$

i. $\mathrm { e . , }$ the CSR list is partitioned by selected-block count rather than by position range, equalizing actual work across splits.

Single-Pass Score and Selection. Scoring and thresholding are fused into one pass over the pooled keys. With $s _ { i J } = q _ { i } \cdot \bar { k } _ { J } / \sqrt { d }$ , the scoring kernel maintains a running tile maximum � and block energies online,

$$
M \gets \mathrm { m a x } ( M , M _ { \mathrm { t i l e } } ) , \qquad E _ { J } = \sum _ { q _ { i } \in I } 2 ^ { s _ { i J } \log _ { 2 } e - M } ,\tag{19}
$$

then writes $E _ { J }$ (bitcast to integer) into the index buffer itself together with the per-chunk maximum, eliminating both a separate score buffer and a second GEMM pass. Selection then only rescales and thresholds:

$$
\hat { E } _ { J } = E _ { J } \cdot 2 ^ { M _ { c } - M _ { \mathrm { f i n a l } } } , \qquad J \in { \cal S } \iff \hat { E } _ { J } \ \geq \ \alpha \cdot \operatorname* { m a x } _ { \iota ^ { \prime } } \hat { E } _ { J ^ { \prime } } \ \vee \ J \in \{ \mathrm { s i n k } \} \cup \{ \mathrm { w i n d o w } \} \cup \{ \mathrm { d i a g } \} ,\tag{20}
$$

and survivors are compacted in place, which is safe since the write position never exceeds the scan position.

Sparsity-Aware Load Balancing. After PackGQA, the cost of tile � is proportional to its selected-block count,

$$
C ( I ) \ \propto \ T _ { I } \ = \ O ( L ) \ \mathrm { f o r \ t h e \ l a s t \ t i l e s } , O ( 1 ) \ \mathrm { o t h e r w i s e } ,\tag{21}
$$

so the dense heuristic that disables splitting once tiles cover the SMs leaves the heavy tail tiles serialized. We therefore always admit splits on the block-sparse path: requests are scheduled in descending order of their causal workload by a device-side sort in the scheduler preparation pass, and each tile’s selected-block list is partitioned evenly across splits by count (Eq. 18), so that every split receives an equal share of the actual sparse work; tile assignment is resolved by warp-level parallel prefix scans over per-request lengths.

Algorithm 1 FlashPrefill V2 prefill pipeline (one attention layer)   
Require: packed queries �, paged KV cache $( K , V )$ with page table, lengths $\{ ( L _ { b } ^ { q } , L _ { b } ^ { k \nu } ) \}$ , threshold   
�   
Ensure: attention output �   
1: �<sup>′</sup> ← PackGQA(�), with row mapping of Eq. 13   
2: $( \bar { k } _ { J } , \bar { \upsilon } _ { J } )$ ← block means of $( K , V ) .$ , one Triton pass (fp8 descaled if needed)   
3: $E _ { J } , M$ ← fused scoring of Eq. 19 over $( Q ^ { \prime } , \bar { k } )$ $/ /$ single GEMM pass   
4: S ← thresholding and in-place compaction of Eq. 20, forcing sink/window/diag blocks   
5: (cu, idx) ← CSR index of S, logical blocks expanded to $B / 6 4$ physical tiles   
6: for each work unit $( b , h , I )$ from the persistent scheduler (descending workload order) do   
7: producer: stage $K , V$ tiles of $\{ \boldsymbol { n } ^ { ( t ) } \}$ (Eq. 17) via TMA/cp.async through the page table   
(Eq. 22)   
8: consumer: wgmma + online softmax in pingpong order (Eq. 14)   
9: correction: append mean chunks of unselected blocks with logit shift log $| \mathcal { B } _ { J } | \left( \mathrm { E q . } \theta \right)$   
10: end for   
11: epilogue: rescale by the accumulated softmax denominator (and un-permute for $\mathrm { f p } 8 )$   
12: return �

Paged and Variable-Length Execution. K/V addressing goes through the page table: the global address of key token � of request � is

$$
\operatorname { a d d r } ( b , j ) = \operatorname { p a g e \_ t a b l e } \left[ b , \lfloor j / p \rfloor \right] \cdot p + ( j \operatorname { m o d } p ) ,\tag{22}
$$

with page size $p ,$ resolved on the fly with cp.async; a varlen-aware persistent scheduler enumerates $( b , h , I )$ work units from per-request lengths $( L _ { b } ^ { q } , L _ { b } ^ { k \nu } )$ , which is exactly the execution model of continuous batching. The CSR index format is compatible with SGLang, allowing direct integration as an attention backend (Sec. 4).

Implementation Details. All instantiations share one tile configuration $( 1 2 8 \times 6 4 ,$ two-stage smem pipeline, �/64 physical tiles per logical block); registers are statically reallocated between roles: the producer holds 24 registers with TMA-based KV (40 on the cp.async paged path), and each consumer warpgroup holds 240 (232 on the paged path).

## 3.4. End-to-End Prefill Pipeline

Fig. 3 and Algorithm 1 put the pieces together. Given a batch of requests with new-token lengths $\{ L _ { b } ^ { \bar { q } } \}$ and cached KV lengths $\{ L _ { b } ^ { k \nu } \}$ , FlashPrefill V2 executes two stages per layer: an index stage (Sec. 3.1) that packs queries, scores pooled blocks, and compacts the survivors into a CSR index, and an attention stage (Sec. 3.2, 3.3) that runs the warp-specialized sparse operator over the selected blocks with mean correction.

Overhead Analysis. For one KV head with group size �, query length $L _ { q } ,$ and KV length �, the FLOP costs of the three components are

$$
\begin{array} { r } { C _ { \mathrm { i d x } } = O \Big ( g L _ { q } \cdot \frac { L } { B } \cdot d \Big ) , \qquad C _ { \mathrm { c o r r } } = O \Big ( g L _ { q } \cdot \frac { ( 1 - \rho ) L } { B } \cdot d \Big ) , \qquad C _ { \mathrm { a t t n } } = O \big ( g L _ { q } \cdot \rho L \cdot 2 d \big ) , } \end{array}\tag{23}
$$

where $\rho$ is the retained density, � the block size, and the factor of two accounts for the QK and PV GEMMs. Both the index stage and the correction operate on one pooled vector per block, hence a factor of � below the dense reference; relative to the sparse attention itself,

$$
\frac { C _ { \mathrm { i d x } } + C _ { \mathrm { c o r r } } } { C _ { \mathrm { a t t n } } } = O \Big ( \frac { 2 - \rho } { 2 \rho B } \Big ) ,\tag{24}
$$

<table><tr><td rowspan="2">Method</td><td colspan="6">RULER Score</td><td colspan="6">Operator Speedup over FA2</td></tr><tr><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>64K 128K</td><td>Avg.</td><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>64K</td><td>128K</td></tr><tr><td colspan="10">Llama-3.1-8B-Instruct</td><td></td><td></td><td></td></tr><tr><td>Full</td><td>95.98</td><td>94.74</td><td>93.07</td><td>89.06 86.27</td><td>73.82</td><td>88.82</td><td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td></tr><tr><td>MInference</td><td>95.06</td><td>94.23</td><td>92.18</td><td>87.68</td><td>83.01 71.61</td><td>87.30</td><td>0.12x</td><td>0.18×</td><td>0.46x</td><td>0.83x</td><td>1.34×</td><td>2.45x</td></tr><tr><td>FlexPrefill</td><td>95.18</td><td>94.09</td><td>92.35</td><td>86.79</td><td>84.08</td><td>72.01 87.42</td><td>0.12x</td><td>0.33×</td><td>0.98×</td><td>2.21×</td><td>4.16×</td><td>5.18x</td></tr><tr><td>XAttention</td><td>95.24</td><td>94.37</td><td>91.61</td><td>87.03</td><td>83.65</td><td>72.33 87.37</td><td>0.79×</td><td>1.29×</td><td>1.83x</td><td>2.34x</td><td>3.19x</td><td>3.48x</td></tr><tr><td>FlashPrefill</td><td>95.16</td><td>94.29</td><td>92.11</td><td>87.49</td><td>84.18</td><td>70.91 87.36</td><td>1.21×</td><td>2.38×</td><td>4.31×</td><td>7.46×</td><td>13.62×</td><td>22.67×</td></tr><tr><td>FlashPrefill V2</td><td>95.52</td><td>94.65</td><td>92.95</td><td>88.48</td><td>83.07</td><td>72.08 87.79</td><td>1.66×</td><td>2.62×</td><td>4.28×</td><td>7.63×</td><td>14.39×</td><td>27.21×</td></tr><tr><td>FlashPrefill V2-FP8*</td><td>95.28</td><td>94.48 92.78</td><td>88.80</td><td>80.28</td><td>67.78</td><td>86.57</td><td>2.67×</td><td>4.56×</td><td>7.79×</td><td>13.89×</td><td>25.76×</td><td>47.33×</td></tr><tr><td colspan="10">Qwen3-4B-Instruct-2507</td><td></td><td></td><td></td></tr><tr><td>Full</td><td>94.82</td><td>92.98</td><td>91.68 88.86</td><td>82.81</td><td>71.22</td><td>87.06</td><td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td></tr><tr><td>MInference</td><td>94.06</td><td>92.18</td><td>91.09</td><td>88.07</td><td>82.08 68.01</td><td>85.92</td><td>0.11×</td><td>0.15×</td><td>0.42x</td><td>0.83x</td><td>1.32×</td><td>2.53x</td></tr><tr><td>FlexPrefill</td><td>93.81</td><td>90.96</td><td>91.41</td><td>87.62</td><td>81.37</td><td>67.24 85.40</td><td>0.12x</td><td>0.33×</td><td>0.99×</td><td>2.28×</td><td>4.01×</td><td>5.27×</td></tr><tr><td>XAttention FlashPrefill</td><td>94.42</td><td>91.49</td><td>91.42</td><td>87.31</td><td>81.56</td><td>68.24 85.74</td><td>0.81×</td><td>1.35×</td><td>1.92x</td><td>2.38×</td><td>3.07×</td><td>3.43×</td></tr><tr><td>FlashPrefill V2</td><td>94.71</td><td>92.04</td><td>91.28</td><td>87.29</td><td>81.08</td><td>69.02 85.90</td><td>1.21×</td><td>2.40×</td><td>4.36×</td><td>6.91×</td><td>11.54x</td><td>19.26×</td></tr><tr><td></td><td>94.49</td><td>92.42 92.37</td><td>91.56</td><td>87.48</td><td>81.67</td><td>69.76 86.23</td><td>1.72x</td><td>2.82×</td><td>4.63×</td><td>7.92×</td><td>13.93×</td><td>27.45×</td></tr><tr><td>FlashPrefill V2-FP8</td><td>94.13</td><td>91.36</td><td>87.47</td><td>80.59</td><td>68.97</td><td>85.82</td><td>2.75x</td><td>4.87×</td><td>8.40×</td><td>14.44×</td><td>25.06×</td><td>47.59×</td></tr><tr><td colspan="10">Qwen3-30B-A3B-Instruct-2507</td><td></td><td></td><td></td></tr><tr><td>Full</td><td>95.78</td><td>95.23</td><td>94.36 92.72</td><td>90.11</td><td>84.12</td><td>92.05</td><td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td></tr><tr><td>MInference</td><td>95.71</td><td>94.33</td><td>93.81</td><td>92.01</td><td>89.02</td><td>82.09 91.16</td><td>0.11×</td><td>0.17×</td><td>0.46x</td><td>0.83x</td><td>1.43×</td><td>2.76x</td></tr><tr><td>FlexPrefill</td><td>95.32</td><td>94.71</td><td>93.52</td><td>91.81</td><td>88.09</td><td>83.11 91.09</td><td>0.12×</td><td>0.34×</td><td>0.98x</td><td>2.31×</td><td>3.92x</td><td>5.43x</td></tr><tr><td>XAttention</td><td>95.43</td><td>94.12</td><td>92.78</td><td>91.06</td><td>88.39</td><td>82.23 90.67</td><td>0.86x</td><td>1.44×</td><td>2.01×</td><td>2.42×</td><td>2.95×</td><td>3.42x</td></tr><tr><td>FlashPrefill</td><td>95.62</td><td>94.47</td><td>93.28</td><td>92.17</td><td>88.14</td><td>82.96 91.11</td><td>1.22x</td><td>2.43×</td><td>4.36×</td><td>6.92×</td><td>11.45×</td><td>18.67×</td></tr><tr><td>FlashPrefill V2</td><td>95.35</td><td>94.85</td><td>94.19</td><td>92.48</td><td>89.88</td><td>83.83 91.76</td><td>1.75×</td><td>2.95×</td><td>4.62×</td><td>8.21×</td><td>14.11×</td><td>27.19×</td></tr><tr><td>FlashPrefill V2-FP8</td><td>95.28</td><td>94.53</td><td>94.12 92.34</td><td>89.54</td><td>82.55</td><td>91.39</td><td>2.78×</td><td>5.12×</td><td>8.41×</td><td>14.93×</td><td>25.36×</td><td>47.26×</td></tr></table>

Table 1 | RULER scores (left) and attention operator speedups over FlashAttention-2 (right) at sequence lengths from 4K to 128K. The speedups are measured end-to-end on needle-ina-haystack inputs, including pattern discovery, thresholding, and mean correction. Results marked with <sup>∗</sup> (in gray) are measured with direct online FP8 quantization without corrected weights.

i.e., the auxiliary FLOPs vanish as the retained density grows, and even at � = 4% with � = 128 they are bounded by roughly 19%, consistent with the gap between the attention-only and end-to-end speedups at long contexts in Fig. 2.

Serving Integration. FlashPrefill V2 is integrated into SGLang as a standard attention backend. The two-stage pipeline runs in the extend (prefill) phase, while decode falls back to dense attention, since a single query token per step leaves no room for block-level sparsity. The page-level index required by our kernels is derived from SGLang’s token-level mapping by striding every � entries, and per-forward metadata is built once per step and shared by all layers. Selection hyperparameters (�, sink/window sizes) are exposed as server arguments, and the index stage reuses a per-stream workspace cache whose capacity-matched reuse avoids any host synchronization or allocator round trip in steady state. No changes to the model definition, the KV cache layout, or the scheduling logic are required.

## 4. Experiments

We evaluate FlashPrefill V2 on two widely-recognized long-context benchmarks, RULER [10] and LongBench [2], across representative LLMs. We first describe the experimental setup, then report accuracy and efficiency results, and finally present ablation studies on the individual components.

<table><tr><td>Method</td><td colspan="3">Single-Document QA</td><td colspan="4">Multi-Document QA</td><td colspan="4">Summarization</td><td colspan="3">Few-shot Learning</td><td>Synthetic</td><td colspan="3">Code</td><td></td><td></td></tr><tr><td></td><td rowspan="2">NarrQA</td><td rowspan="2">Qasper</td><td rowspan="2">MF-en</td><td rowspan="2">MF-zh Hotpot</td><td rowspan="2">2wi</td><td rowspan="2">MuSiQue</td><td rowspan="2">DuRead</td><td rowspan="2">GovRep</td><td rowspan="2">QMSum</td><td rowspan="2">MNews</td><td rowspan="2">VCSum</td><td rowspan="2">TREC</td><td rowspan="2">TOA</td><td rowspan="2">SAMSum</td><td rowspan="2">LSHT</td><td rowspan="2">Count PR-en</td><td rowspan="2">PR-zh</td><td rowspan="2"></td><td rowspan="2">8</td><td rowspan="2">RepB</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Llama-3.1-8B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Full</td><td>29.39</td><td>44.61</td><td>56.34</td><td>63.63 57.48</td><td>48.08</td><td>32.28</td><td>34.43</td><td>34.51</td><td>25.29</td><td>26.80</td><td>17.37 72.50</td><td>91.48</td><td>43.59</td><td>46.50</td><td>11.10</td><td>100.00</td><td>90.45</td><td>62.93</td><td>56.19</td><td>49.76</td></tr><tr><td>MInference</td><td>28.13</td><td>42.36</td><td>53.67</td><td>62.08 56.28</td><td>46.72</td><td>32.01</td><td>33.41</td><td>33.28</td><td>25.16</td><td>27.02</td><td>16.84</td><td>69.50 90.83</td><td>42.87</td><td>43.00</td><td>6.00</td><td>96.00</td><td>84.00</td><td>60.41</td><td>55.92</td><td>47.88</td></tr><tr><td>FlexPrefill</td><td>27.62</td><td>42.67</td><td>52.98</td><td>62.34 57.14</td><td>45.32</td><td>30.97</td><td>33.72</td><td>33.87</td><td>24.58</td><td>26.34</td><td>17.02</td><td>67.50 90.36</td><td>44.72</td><td>43.00</td><td>6.00</td><td>94.00</td><td>83.33</td><td>61.72</td><td>55.83</td><td>47.67</td></tr><tr><td>XAttention FlashPrefill</td><td>28.52 28.31</td><td>43.12 42.89</td><td>54.01 54.23</td><td>60.19 56.73</td><td>45.98</td><td>32.17</td><td>33.15</td><td>34.55</td><td>25.37</td><td>27.21</td><td>17.41</td><td>71.00 91.37</td><td>44.05</td><td>44.00</td><td>6.88</td><td>95.50</td><td>85.68</td><td>61.05</td><td>55.36</td><td>48.25</td></tr><tr><td>FlashPrefill V2</td><td>30.21</td><td>44.56</td><td>55.97</td><td>61.97 57.01 62.37</td><td>45.76</td><td>31.96</td><td>33.60</td><td>34.02</td><td>24.83</td><td>26.47</td><td>17.13 71.00</td><td>91.12</td><td>44.21</td><td>46.50</td><td>7.00</td><td>95.50</td><td>83.45</td><td>62.33</td><td>53.41</td><td>48.22</td></tr><tr><td>FlashPrefill V2-FP8*</td><td>29.08</td><td></td><td>54.70 62.59</td><td>58.02 55.41</td><td>48.84 46.40</td><td>32.57 32.03</td><td>33.78 32.97 34.70</td><td>34.90</td><td>25.24 24.88</td><td>27.13 26.97</td><td>17.37 72.50 17.61 70.00</td><td>91.45 91.16</td><td>44.76 44.80</td><td>45.50</td><td>7.39</td><td>97.00 96.50</td><td>89.47</td><td>62.68 63.65</td><td>53.86</td><td>49.31 48.76</td></tr><tr><td></td><td colspan="4">45.51</td><td colspan="8"></td><td colspan="2">45.00</td><td colspan="2">6.88</td><td colspan="2">85.68</td><td colspan="2">57.39</td></tr><tr><td>Full</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Qwen3-4B-Instruct-2507</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MInference</td><td>27.97 25.89</td><td>44.98 41.69</td><td>51.12 48.26</td><td>64.59 59.39 62.16 54.17</td><td>44.18</td><td>24.85 22.16</td><td>26.17</td><td>30.68</td><td>22.62</td><td>23.93</td><td>11.73</td><td>74.00</td><td>87.74</td><td>45.55</td><td>44.00 0.75</td><td>100.00</td><td></td><td>98.04</td><td>62.99</td><td>55.56</td><td>47.66 45.48</td></tr><tr><td>FlexPrefill</td><td>25.26</td><td>41.48</td><td>48.02</td><td>61.98 54.96</td><td>42.76 43.21</td><td>22.71</td><td>24.89 25.43</td><td>29.12</td><td>21.98</td><td>23.15</td><td>11.28</td><td>72.00</td><td>87.62</td><td>44.32</td><td>42.50</td><td>0.00</td><td>91.00</td><td>94.67</td><td>60.92</td><td>54.48</td><td>45.82</td></tr><tr><td>XAttention</td><td>24.31</td><td>42.67</td><td>49.01</td><td>63.02 54.14</td><td>43.14</td><td>22.56</td><td>25.12</td><td>30.94</td><td>22.15</td><td>24.47</td><td>11.47</td><td>70.50</td><td>86.18</td><td>45.87</td><td>41.00</td><td>0.50</td><td>94.00</td><td>94.20</td><td>61.67</td><td>56.19</td><td>45.90</td></tr><tr><td>FlashPrefill</td><td>25.06</td><td>42.78</td><td>49.22</td><td>62.31 55.07</td><td>42.97</td><td>23.42</td><td></td><td>29.63</td><td>22.41</td><td>23.68</td><td>11.56</td><td>72.50</td><td>86.47</td><td>45.13</td><td>41.50</td><td>0.00</td><td>96.00</td><td>93.89</td><td>62.24</td><td>54.92</td><td>45.84</td></tr><tr><td>FlashPrefill V2</td><td>27.56</td><td>43.73</td><td>51.07</td><td>63.72 56.39</td><td>43.11</td><td>23.18</td><td>25.66</td><td>29.80</td><td>22.09</td><td>23.32</td><td>11.62</td><td>72.00</td><td>87.95</td><td>45.42</td><td>42.00</td><td>0.00</td><td>92.00</td><td>94.00</td><td>62.71</td><td>53.23</td><td></td></tr><tr><td>FlashPrefill V2-FP8</td><td>26.55</td><td>42.70</td><td>50.28</td><td>61.71 55.21</td><td>41.95</td><td>24.50</td><td>26.36</td><td>30.48 30.74</td><td>22.67</td><td>23.88</td><td>11.76</td><td>74.00</td><td>87.74</td><td>45.78 45.89</td><td>41.25 41.25</td><td>0.50</td><td>96.00</td><td>97.17</td><td>63.50</td><td>56.38</td><td>46.96 46.07</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>26.03</td><td>Qwen3-30B-A3B-Instruct-2507</td><td>22.47</td><td>24.07</td><td>11.48</td><td>73.50</td><td>88.16</td><td></td><td></td><td>0.50</td><td>93.00</td><td>94.67</td><td>60.85</td><td>52.06</td><td></td></tr><tr><td>Full 31.29 44.09</td><td></td><td></td><td>54.79 66.48</td></table>

Table 2 | Performance comparison on all 21 tasks of LongBench [2], grouped by task category. Results marked with <sup>∗</sup> (in gray) are measured with direct online FP8 quantization without corrected weights.

## 4.1. Setup

Benchmarks and Models. RULER [10] stresses long-context retrieval and reasoning at controlled sequence lengths, and LongBench [2] covers a diverse suite of realistic long-context tasks. We evaluate on Llama-3.1-8B-Instruct [9], Qwen3-4B-Instruct-2507 [36], and Qwen3-30B-A3B-Instruct-2507 [36].

Baselines. We compare against Full Attention [29], MInference [11], FlexPrefill [14], and XAttention [35]. All methods are evaluated under identical hyperparameter settings on NVIDIA H20 GPUs, the most widely deployed inference GPU in our target deployment environments; all measurements in this section are collected on H20 unless otherwise noted. As dense references, we use FlashAttention-2 [5] and an FA3/4-aligned dense kernel [22, 38] that shares the same warp-specialized pipeline as our sparse operator, so that the measured speedups isolate the benefit of sparsity rather than kernel engineering.

Implementation Details. The block-sparse operator is compiled with fixed tile sizes: a query tile of 128 rows (kBlockM) and a key tile of 64 columns (kBlockN). For all accuracy evaluations, every model shares a single configuration without per-model tuning: selection block size � = 128, 256 sink tokens, a local window of 512 tokens, and � = 0.1 in the max-based thresholding. All operator-level results are measured on a single H20 GPU. End-to-end results (Tabs. 4, 5, and 6) are collected from a single SGLang server instance per configuration with tensor parallelism ��=4 on four H20 GPUs and no data parallelism; the decode attention backend is fixed to FA3/4 across all configurations, and only the prefill attention backend varies. The software stack comprises CUDA 12.9 (driver 535.161.08), PyTorch 2.9.1, CUTLASS 4.3 for the attention kernels, and SGLang 0.5.10 [42] for serving.

## 4.2. Accuracy Results

RULER. Tab. 1 reports RULER scores at controlled sequence lengths (left) together with the end-to-end attention operator speedups (right). Across all three models, FlashPrefill V2 stays within 1.1 points of Full attention on average (87.79 vs 88.82 on Llama-3.1-8B, 86.23 vs 87.06 on Qwen3-4B, and 91.76 vs 92.05 on Qwen3-30B-A3B), and the FP8 variant costs only an additional 0.4 to 1.2 points. The gap is largest at 128K where the density drops below 5% (Tab. 3 lists the measured densities at each sequence length), yet V2 remains within 1.8 points of Full attention. Meanwhile, the sparse operator including index selection already outperforms FA2 at 4K and reaches 27.2× to 27.5× at 128K in BF16 and 47.3× to 47.6× in FP8.

<table><tr><td>Model</td><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>64K</td><td>128K</td></tr><tr><td>Llama-3.1-8B</td><td>76.0%</td><td>54.2%</td><td>33.6%</td><td>18.7%</td><td>9.0%</td><td>4.6%</td></tr><tr><td>Qwen3-4B</td><td>72.0%</td><td>48.5%</td><td>29.8%</td><td>17.6%</td><td>9.2%</td><td>4.9%</td></tr><tr><td>Qwen3-30B-A3B</td><td>70.4%</td><td>46.0%</td><td>29.6%</td><td>16.2%</td><td>9.4%</td><td>4.9%</td></tr></table>

Table 3 | Attention density of FlashPrefill V2 measured on needle-in-a-haystack inputs at each sequence length.
<table><tr><td rowspan="2">BSZ</td><td colspan="6">FA3/4</td><td colspan="6">FlashPrefill V2</td><td colspan="6">FlashPrefill V2-FP8</td></tr><tr><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>64K</td><td>128K</td><td>4K 8K</td><td>16K</td><td>32K</td><td>64K</td><td>128K</td><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>64K</td><td>128K</td></tr><tr><td colspan="10"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>0.15</td><td>0.29</td><td>0.63</td><td>1.49</td><td>3.95</td><td>11.81</td><td>0.15 0.29 (1.02×) (1.02×)</td><td>0.58 (1.09×)</td><td>1.17 (1.28×)</td><td>2.46 (1.60×)</td><td>5.49 (2.15×)</td><td>0.10 (1.53×)</td><td>0.18 (1.63×)</td><td>0.35 (1.81×)</td><td>0.71 (2.09×)</td><td>1.48 (2.66×)</td><td>3.23 (3.66×)</td></tr><tr><td>4</td><td>0.58</td><td>1.17</td><td>2.54 5.94</td><td></td><td>12.30</td><td>32.94</td><td>0.57 1.14 (1.01×) (1.03×)</td><td>2.32 (1.09×)</td><td>4.79 (1.24×)</td><td>7.69 (1.60×)</td><td>15.39 (2.14×)</td><td>0.35 (1.66×)</td><td>0.68 (1.73×)</td><td>1.36 (1.87×)</td><td>2.80 (2.12×)</td><td>4.49 (2.74×)</td><td>8.84 (3.73×)</td></tr><tr><td>8</td><td>1.07</td><td>2.21</td><td>4.87</td><td>9.83</td><td>21.23</td><td>56.67</td><td>1.06 (1.01×)</td><td>2.14 4.39 (1.03×) (1.11×)</td><td>7.77 (1.27×)</td><td>13.18 (1.61×)</td><td>26.13 (2.17×)</td><td>0.67 (1.58×)</td><td>1.39 (1.59×)</td><td>2.81 (1.73×)</td><td>4.51 (2.18×)</td><td>7.66 (2.77×)</td><td>14.93 (3.80×)</td></tr><tr><td>16</td><td>2.60</td><td>3.85</td><td>8.14</td><td>16.10</td><td>37.12</td><td>102.84</td><td>2.58 3.59 (1.01×) (1.07×)</td><td>7.31 (1.11×)</td><td>12.56 (1.28×)</td><td>22.93 (1.62×)</td><td>46.96 (2.19×)</td><td>1.44 (1.80×)</td><td>2.97 (1.30×)</td><td>4.62 (1.76×)</td><td>7.28 (2.21×)</td><td>13.25 (2.80×)</td><td>26.82 (3.83×)</td></tr><tr><td colspan="10">Qwen3-4B-Instruct-2507</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.09</td><td>0.19</td><td>0.43</td><td>1.13</td><td>3.35</td><td>11.10</td><td>0.09 0.18 (1.02×) (1.07×) 0.35 0.69</td><td>0.36 (1.19×)</td><td>0.77 (1.47×)</td><td>1.71 (1.96×)</td><td>4.12 (2.70×)</td><td>0.07 (1.42×)</td><td>0.12 (1.61×)</td><td>0.23 (1.87×)</td><td>0.49 (2.33×)</td><td>1.08 (3.09×)</td><td>2.44 (4.55×)</td></tr><tr><td>4</td><td>0.35</td><td>0.74</td><td>1.69</td><td>4.35</td><td>10.61</td><td>31.47</td><td>(1.01×) (1.06×)</td><td>1.43 (1.18×)</td><td>3.06 (1.42×)</td><td>5.38 (1.97×)</td><td>11.55 (2.72×)</td><td>0.22 (1.59×)</td><td>0.43 (1.72×)</td><td>0.88 (1.93×)</td><td>1.86 (2.34×)</td><td>3.26 (3.25×)</td><td>6.73 (4.67×)</td></tr><tr><td>8</td><td>0.64</td><td>1.38</td><td>3.25</td><td>7.40</td><td>18.25</td><td>53.84</td><td>0.63 1.29 (1.02×) (1.07×)</td><td>2.70 (1.20×)</td><td>5.05 (1.47×)</td><td>9.19 (1.99×)</td><td>19.35 (2.78×)</td><td>0.43 (1.49×)</td><td>0.87 (1.59×)</td><td>1.78 (1.82×)</td><td>3.06 (2.42×)</td><td>5.51 (3.31×)</td><td>11.28 (4.77×)</td></tr><tr><td>16</td><td>1.07</td><td>2.31</td><td>5.53</td><td>12.20</td><td>31.73</td><td>97.26</td><td>1.05 2.15 (1.02×) (1.08×)</td><td>4.56 (1.21×)</td><td>8.18 (1.49×)</td><td>15.84</td><td>34.48</td><td>0.90 (1.18×)</td><td>1.83 (1.26×)</td><td>2.95 (1.87×)</td><td>4.92 (2.48×)</td><td>9.43 (3.37×)</td><td>20.13 (4.83×)</td></tr><tr><td colspan="10"></td><td>(2.00×) (2.82×) Qwen3-30B-A3B-Instruct-2507</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>0.10</td><td>0.19</td><td>0.45</td><td>1.26</td><td>4.00</td><td>0.09 13.78 (1.02×)</td><td>0.18 (1.09×)</td><td>0.36 (1.26×)</td><td>0.78 (1.61×)</td><td>1.75 (2.29×)</td><td>4.17 (3.30×)</td><td>0.09 (1.06×)</td><td>0.15 (1.24×)</td><td>0.30 (1.50×)</td><td>0.61 (2.07×)</td><td>1.32 (3.03×)</td><td>3.00 (4.59×)</td></tr><tr><td>4</td><td>0.26</td><td>0.73</td><td>1.76</td><td>4.86</td><td>13.05</td><td>0.24 40.44 (1.09×)</td><td>0.67 (1.10×)</td><td>1.42 (1.24×)</td><td>3.11 (1.56×)</td><td>5.70 (2.29×)</td><td>12.16 (3.33×)</td><td>0.22 (1.18×)</td><td>0.57 (1.29×)</td><td>1.15 (1.53×)</td><td>2.41 (2.01×)</td><td>4.21 (3.10×)</td><td>8.59 (4.71×)</td></tr><tr><td>8</td><td>0.64</td><td>1.35</td><td>3.36</td><td>8.47</td><td>22.37</td><td>0.62 68.56 (1.03×)</td><td>1.22 (1.11×)</td><td>2.65 (1.27×)</td><td>5.25 (1.61×)</td><td>9.69 (2.31×)</td><td>20.35 (3.37×)</td><td>0.55 (1.15×)</td><td>1.02 (1.32×)</td><td>2.11 (1.59×)</td><td>4.03 (2.10×)</td><td>7.13 (3.14×)</td><td>14.35 (4.78×)</td></tr><tr><td>16</td><td>1.16</td><td>3.14</td><td>5.90</td><td>13.98</td><td>38.50</td><td>123.23</td><td>1.15 2.92 (1.01×) (1.08×)</td><td>4.60 (1.28×)</td><td>8.49 (1.65×)</td><td>16.55 (2.33×)</td><td>36.21 (3.40×)</td><td>1.04 (1.12×)</td><td>1.73 (1.81×)</td><td>3.66 (1.61×)</td><td>6.47 (2.16×)</td><td>12.16 (3.17×)</td><td>25.51 (4.83×)</td></tr></table>

Table 4 | End-to-end time-to-first-token (TTFT, seconds) of Llama-3.1-8B-Instruct, Qwen3-4B-Instruct-2507, and Qwen3-30B-A3B-Instruct-2507 served by SGLang on the needle-in-a-haystack workload. Each entry is averaged over 30 repeated measurements. Parenthesized numbers are speedups over the FA3/4 backend.

LongBench. Tab. 2 reports the per-task results on LongBench. FlashPrefill V2 achieves the highest average among all sparse operators on every model: 49.31 on Llama-3.1-8B (vs. 48.25 for the strongest baseline, XAttention), 46.96 on Qwen3-4B (vs. 45.90), and 50.73 on Qwen3-30B-A3B (vs. 49.61), staying within 0.9 points of full attention. The advantage is most pronounced on retrieval-sensitive synthetic tasks: on English passage retrieval, V2 retains 96.0 to 99.5 points while the baselines drop to 91.0 to 97.0, and on the counting task V2 remains the closest to full attention on all three models. Summarization tasks are nearly lossless, with V2 occasionally exceeding full attention (e.g., 44.76 vs. 43.59 on SAMSum for Llama-3.1-8B). The FP8 variant, even with direct online quantization, still outperforms every baseline on average (48.76/46.07/50.29). Tab. 3 lists the attention densities measured on needle-in-a-haystack inputs.

<table><tr><td rowspan="2">Rate (req/s)</td><td rowspan="2">Backend</td><td colspan="2">TTFT (s)</td><td colspan="2">TPOT (ms)</td><td colspan="2">Throughput</td></tr><tr><td>P50</td><td>P99</td><td>P50</td><td>P99</td><td>(tok/s)</td><td>(req/s)</td></tr><tr><td colspan="8">Qwen3-4B-Instruct-2507</td></tr><tr><td>1</td><td>FA3/4 FlashPrefill V2 FlashPrefill V2-FP8</td><td>76.73 17.23 (4.45x) 2.79 (27.47x)</td><td>154.64 48.73 (3.17x) 17.27 (8.95x)</td><td>419 258 (1.62x) 47 (8.84x)</td><td>1170 601 (1.94x) 487 (2.40x)</td><td>24.0 45.5 (1.90x) 58.4 (2.44x)</td><td>0.37 0.71 0.91</td></tr><tr><td>4</td><td>FA3/4 FlashPrefill V2 FlashPrefill V2-FP8</td><td>79.05 42.28 (1.87x) 24.47 (3.23x)</td><td>161.54 80.34 (2.01x) 46.05 (3.51x)</td><td>513 258 (1.99x) 165 (3.10x)</td><td>1102 509 (2.16x) 331 (3.33x)</td><td>23.8 47.3 (1.99x) 82.3 (3.46x)</td><td>0.37 0.74 1.29</td></tr><tr><td>8</td><td>FA3/4 FlashPrefill V2 FlashPrefill V2-FP8</td><td>79.55 43.83 (1.81x) 25.83 (3.08x)</td><td>161.01 80.12 (2.01x) 46.24 (3.48x)</td><td>463 236 (1.96x) 152 (3.05x)</td><td>1110 523 (2.12x) 302 (3.68x)</td><td>23.9 48.3 (2.03x) 82.2 (3.45x)</td><td>0.37 0.76 1.28</td></tr><tr><td>16</td><td>FA3/4 FlashPrefill V2 FlashPrefill V2-FP8</td><td>81.91 43.22 (1.90x) 25.14 (3.26x)</td><td>160.99 78.68 (2.05x) 45.16 (3.56x)</td><td>481 263 (1.83x) 152 (3.17x)</td><td>1110 501 (2.21x) 285 (3.89x)</td><td>23.8 48.9 (2.05x) 85.5 (3.59x)</td><td>0.37 0.76 1.34</td></tr><tr><td colspan="8">Qwen3-30B-A3B-Instruct-2507 FA3/4</td></tr><tr><td>1</td><td>FlashPrefill V2 FlashPrefill V2-FP8 FA3/4</td><td>90.69 17.63 (5.14x) 7.93 (11.44x) 101.44</td><td>195.43 52.15 (3.75x) 21.92 (8.91x) 195.33</td><td>582 241 (2.42x) 102 (5.69x) 462</td><td>1335 604 (2.21x) 637 (2.10x) 1312</td><td>19.9 44.5 (2.24x) 56.1 (2.82x) 19.9</td><td>0.31 0.70 0.88 0.31</td></tr><tr><td>4</td><td>FlashPrefill V2 FlashPrefill V2-FP8 FA3/4</td><td>46.26 (2.19x) 30.62 (3.31x) 105.52</td><td>85.57 (2.28x) 57.43 (3.40x) 195.34</td><td>255 (1.81x) 198 (2.33x) 481</td><td>520 (2.52x) 407 (3.22x) 1333</td><td>45.5 (2.29x) 64.6 (3.25x) 19.9</td><td>0.71 1.01</td></tr><tr><td>8</td><td>FlashPrefill V2 FlashPrefill V2-FP8</td><td>45.73 (2.31x) 33.17 (3.18x)</td><td>83.04 (2.35x) 58.46 (3.34x) 195.20</td><td>234 (2.05x) 165 (2.92x)</td><td>529 (2.52x) 369 (3.62x)</td><td>47.0 (2.36x) 66.9 (3.36x)</td><td>0.31 0.73 1.05</td></tr><tr><td>16</td><td>FA3/4 FlashPrefill V2 FlashPrefill V2-FP8</td><td>99.68 45.20 (2.21x) 32.47 (3.07x)</td><td>81.95 (2.38x) 57.33 (3.40x)</td><td>547 266 (2.06x) 185 (2.95x)</td><td>1347 514 (2.62x) 359 (3.75x)</td><td>19.9 47.4 (2.38x) 67.7 (3.41x)</td><td>0.31 0.74 1.06</td></tr></table>

Table 5 | Open-loop serving results on Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B-Instruct-2507 under Poisson arrivals with mixed prompt lengths (4K–128K RULER needle-in-a-haystack, 100 requests per cell, 64 output tokens each). Parenthesized numbers are speedups over the FA3/4 backend.

## 4.3. Efficiency Results

The operator-level speedups over FA2 are reported on the right side of Tab. 1. Fig. 2 further compares against the FA3/4-aligned dense kernel and prior sparse operators at batch size 4: at 128K context, our operator attains 27.2× (BF16) and 47.3× (FP8) over FA2, 17.5× and 30.5× over the FA3/4-aligned dense kernel, and clearly surpasses prior sparse operators (18.7× for FlashPrefill, 5.4× for FlexPrefill, 3.4× for XAttention, and 2.8× for MInference). The density stays around 70% at 4K and drops to about 5% at 128K (Tab. 3), which is the source of the growing speedup.

End-to-End Serving Latency. Tab. 4 reports the end-to-end time-to-first-token of FlashPrefill V2 deployed in SGLang, with batch sizes from 1 to 16 under continuous batching. The operatorlevel gains translate directly into serving latency: at 128K context, FlashPrefill V2 reduces TTFT by 2.1× to 3.4× in BF16 and 3.7× to 4.8× in FP8 across all three models and batch sizes (e.g., from 123.2 s to 36.2 s in BF16 and 25.5 s in FP8 at batch size 16 on Qwen3-30B-A3B). The speedup is nearly batch-invariant, indicating that the sparse operator scales with continuous batching in the same way as the dense backend. As expected, the end-to-end speedup is bounded by the attention share of total prefill compute and is therefore smaller than the operator-level one; it grows with sequence length as attention becomes dominant, and is largest on Qwen3- 30B-A3B, where attention accounts for a larger fraction of prefill compute. At 4K, where the density remains around 70%, BF16 stays at parity with FA3/4 (1.01× to 1.09×), where the modest attention savings are largely offset by the index stage overhead, while FP8 already delivers 1.1× to 1.8×.

<table><tr><td rowspan=1 colspan=1>Chunksize</td><td rowspan=1 colspan=1>Backend</td><td rowspan=1 colspan=1>TTFT (s)P50           P99</td><td rowspan=1 colspan=1>TPOT (ms)P50         P99</td><td rowspan=1 colspan=2>Throughput(tok/s)   (req/s)</td></tr><tr><td rowspan=1 colspan=6>Qwen3-4B-Instruct-2507</td></tr><tr><td rowspan=1 colspan=1>8K</td><td rowspan=1 colspan=1>FA3/4FlashPrefill V2FlashPrefill V2-FP8</td><td rowspan=1 colspan=1>80.98         165.1450.15 (1.61x)  95.23 (1.73x)33.47 (2.42x)  61.20 (2.70x)</td><td rowspan=1 colspan=1>597         1228360 (1.66x)  684 (1.80x)259 (2.30x)  529 (2.32x)</td><td rowspan=1 colspan=1>22.538.7 (1.72x)60.7 (2.70x)</td><td rowspan=1 colspan=1>0.350.600.95</td></tr><tr><td rowspan=1 colspan=1>16K</td><td rowspan=1 colspan=1>FA3/4FlashPrefill V2FlashPrefill V2-FP8</td><td rowspan=1 colspan=1>79.42         161.9645.09 (1.76x)  85.47 (1.89x)26.80 (2.96x)  50.34 (3.22x)</td><td rowspan=1 colspan=1>580         1203315 (1.84x)  615 (1.96x)193 (3.00x)  382 (3.15x)</td><td rowspan=1 colspan=1>22.943.9 (1.91x)73.3 (3.20x)</td><td rowspan=1 colspan=1>0.360.691.15</td></tr><tr><td rowspan=1 colspan=6>Qwen3-30B-A3B-Instruct-2507</td></tr><tr><td rowspan=1 colspan=1>8K</td><td rowspan=1 colspan=1>FA3/4FlashPrefill V2FlashPrefill V2-FP8</td><td rowspan=1 colspan=1>96.61         199.3979.81 (1.21x) 149.52 (1.33x)63.23 (1.53x) 121.53 (1.64x)</td><td rowspan=1 colspan=1>693         1494574 (1.21x) 1200 (1.24x)505 (1.37x) 1009 (1.48x)</td><td rowspan=1 colspan=1>18.724.6 (1.32x)29.2 (1.56x)</td><td rowspan=1 colspan=1>0.290.380.46</td></tr><tr><td rowspan=1 colspan=1>16K</td><td rowspan=1 colspan=1>FA3/4FlashPrefill V2FlashPrefill V2-FP8</td><td rowspan=1 colspan=1>94.78         195.8446.47 (2.04x)  88.15 (2.22x)33.72 (2.81x)  62.19 (3.15x)</td><td rowspan=1 colspan=1>672         1466323 (2.08x)  652 (2.25x)250 (2.69x)  464 (3.16x)</td><td rowspan=1 colspan=1>19.042.1 (2.21x)57.1 (3.00x)</td><td rowspan=1 colspan=1>0.300.660.89</td></tr></table>

Table 6 | Open-loop serving at 16 req/s with chunked prefill at chunk sizes of 8K and 16K (same workload as Tab. 5, where chunking is disabled). Parenthesized numbers are speedups over the FA3/4 backend at the same chunk size.

Open-Loop Serving. Tab. 5 moves from fixed batches to an open-loop setting: requests arrive as a Poisson process at 1 to 16 req/s with prompt lengths mixed over 4K–128K, so prefills and decodes of different requests continuously share the engine. The FA3/4 backend is saturated at every rate (P50 TTFT of 77–106 s, request throughput capped at 0.31–0.37 req/s), whereas FlashPrefill V2 sustains about twice the request throughput (0.70–0.76 req/s) and cuts the P50 TTFT to 17–46 s; with FP8 the request throughput rises to 0.88–1.34 req/s and the P50 TTFT drops further to 2.8–33 s. The speedup is largest at the lowest rate (4.5×–5.1× P50 at 1 req/s in BF16 and 11.4×–27.5× in FP8), where faster prefills shorten queueing much more than at higher rates, and remains 1.8×–2.3× in BF16 and 3.1×–3.3× in FP8 once the system is loaded. Shorter prefills also benefit decoding: because long dense prefills block the decode steps of concurrently running requests, P50 TPOT improves by 1.6×–2.4× in BF16 and 2.3×–8.8× in FP8, and output throughput rises by 1.9×–2.4× and 2.4×–3.6× respectively.

Compatibility with Chunked Prefill. Chunked prefill [1], which bounds the number of tokens processed per scheduling step, is the standard mechanism in production engines for protecting inter-token latency from long prefills. Tab. 6 repeats the open-loop experiment at 16 req/s with chunk sizes of 8K and 16K. Chunking barely affects the dense backend’s TTFT, but it erodes our speedup because the index selection is re-run on every chunk and the mandatory tail blocks raise the effective density of short chunks: the P50 TTFT speedup in BF16 drops from 1.9× without chunking to 1.6× at 8K on Qwen3-4B, and from 2.2× to 1.2× on Qwen3-30B-A3B. Even so, FlashPrefill V2 remains faster than the dense backend at 8K (1.2×–1.6× in BF16 and 1.5×–2.4× in FP8), and a 16K chunk recovers most of the margin (1.8×–2.0× and 2.8×–3.0× respectively), so we recommend chunk sizes of at least 8K when deploying FlashPrefill V2 together with chunked prefill.

![](images/a34fd477fc95623209dafabb31d8de741a6eb9dc4acd39e044877b3aa043b7ba.jpg)

![](images/f4414959c8bfee14657878783ef9dc02a93ce965475b2dfb14c340821bd18e9e.jpg)  
Figure 5 | Block-sparse attention latency at 64K sequence length in FP8. HPC-BSA does not support BF16, so only FP8 is compared. Both sparse operators track the theoretical bound; FlashPrefill V2 is 6 to 7% faster than HPC-BSA at every sparsity level.  
Figure 6 | Overhead of mean correction at 64K sequence length. The corrected variants (dashed) stay close to the uncorrected ones (solid) in both BF16 and FP8, and the absolute overhead shrinks as sparsity increases.

Comparison with the Production Kernel. Fig. 5 compares our block-sparse operator against the block-sparse attention (BSA) kernel in HPC-Ops [26], a production inference operator library. Since the HPC-Ops BSA kernel only supports FP8, the comparison is conducted in FP8 at 64K sequence length. Our operator is consistently faster than HPC-BSA across all sparsity levels (by 6 to 7%), and both dense references are comparable (123.7 ms vs 133.4 ms), confirming that the gain comes from the sparse execution path rather than the dense backbone. Tab. 7 summarizes the design differences behind this gap.

Overhead of Mean Correction. Fig. 6 quantifies the runtime cost of the correction term at 64K. Because all pruned blocks within a query tile are replaced by a single pass over their pooled K/V means, the extra latency is small and nearly constant in absolute terms: at most 12.2 ms in the dense limit and only 3.6 to 4.5 ms at 90% sparsity. The relative overhead therefore stays within 5% to 14% up to 50% sparsity, and reaches 18% (BF16) to 27% (FP8) only at 90% sparsity, the regime where the correction is most needed for accuracy.

## 4.4. Ablation Study

Effect of Mean Correction. Tab. 8 ablates the mean correction term (Sec. 3.2) on RULER with Qwen3-4B-Instruct-2507, comparing the full FlashPrefill V2 pipeline against a variant in which all unselected blocks are simply discarded. Without correction, the dropped probability mass grows as the density falls, so the accuracy gap widens with sequence length: in BF16 the variant trails the corrected pipeline by only 0.3 points on average up to 32K, but by 0.8 points at 64K and 0.9 points at 128K, where fewer than 5% of the blocks survive selection (Tab. 3). Restoring the correction term recovers nearly all of this degradation at a negligible runtime cost (Fig. 6), keeping FlashPrefill V2 within 0.9 points of full attention on average and within 1.5 points even at 128K. The correction becomes substantially more important in FP8: without it, the FP8 variant loses 2.3 points on average and 6.2 points at 128K relative to its corrected counterpart, compared with only 0.5 and 0.9 points in BF16. We attribute this to the compressed score margins under quantization, which let the blocks discarded by the max-based threshold carry a larger share of the softmax mass, so recovering them matters more. With correction enabled, the FP8 pipeline tracks BF16 closely (85.82 vs. 86.23 on average), confirming that the correction remains effective even when the pooled statistics are quantized.

<table><tr><td></td><td>HPC-Ops BSA</td><td>FlashPrefill V2</td></tr><tr><td>Sparse index format Index memory @ 64K / 128K GEMM-softmax overlap Paged KV addressing</td><td>dense mask, padded, per Q head 8 MB / 32 MB (any density) serial per-page TMA (page size | 128)</td><td>CSR of selected blocks, per KV head 3 MB / 6 MB (9% / 5% density) intra-warpgroup pingpong cp.async, arbitrary page size</td></tr></table>

Table 7 | Design comparison with the HPC-Ops block-sparse attention kernel. Index memory is evaluated at batch size 1 with 32 query heads and 4 KV heads: the dense mask stores one byte per (query head, 128-token Q tile, 128-token KV tile) regardless of density, whereas the CSR index stores only the selected 64-token tiles as int32 at the measured needle-in-a-haystack densities (Tab. 3), is organized per KV head under PackGQA, and matches the SGLang index format. The fused correction path has no counterpart in HPC-Ops BSA.
<table><tr><td>Method</td><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>64K</td><td>128K</td><td>Avg.</td></tr><tr><td>Full Attention V2 w/o correction</td><td>94.82</td><td>92.98</td><td>91.68</td><td>88.86</td><td>82.81</td><td>71.22</td><td>87.06</td></tr><tr><td>FlashPrefill V2</td><td>94.38 94.49</td><td>92.06 92.42</td><td>91.18 91.56</td><td>87.21 87.48</td><td>80.92 81.67</td><td>68.86 69.76</td><td>85.77 86.23</td></tr><tr><td>V2-FP8 w/o correction</td><td>93.78</td><td>91.82</td><td>90.68</td><td>85.04</td><td>76.86</td><td>62.78</td><td></td></tr><tr><td>FlashPrefill V2-FP8</td><td>94.13</td><td>92.37</td><td>91.36</td><td>87.47</td><td>80.59</td><td>68.97</td><td>83.49 85.82</td></tr></table>

Table 8 | Ablation of the mean correction term on RULER with Qwen3-4B-Instruct-2507, in both BF16 and FP8. Discarding unselected blocks without compensation degrades accuracy increasingly toward longer contexts, while the zero-order correction stays close to full attention at all lengths in both precisions.

Effect of the Selection Threshold. Tab. 9 sweeps the threshold � of the max-based selection on RULER at 64K with Qwen3-4B-Instruct-2507 in FP8, and reports the measured attention density together with the scores with and without mean correction. As � decreases from 0.2 to 0.0125, the density rises from 5.2% to 23.6%. The corrected FP8 pipeline is remarkably flat across the entire sweep, staying within 2.7 points of full attention (82.81) even at 5.2% density and within 2.3 points at the default � = 0.1, whereas removing the correction consistently costs 2.7 to 5.4 further points, with the gap widening as the density falls. Mean correction therefore enlarges the usable sparsity range: with compensation, � can be pushed to 0.2 at 64K—halving the density relative to the default—while retaining 80.12 points.

<table><tr><td>α</td><td>0.2</td><td>0.1</td><td>0.05</td><td>0.025</td><td>0.0125</td></tr><tr><td>Density (%)</td><td>5.2</td><td>9.2</td><td>14.1</td><td>19.8</td><td>23.6</td></tr><tr><td>V2-FP8 w/o correction</td><td>74.68(-5.44)</td><td>76.86(-3.73)</td><td>77.16(-3.51)</td><td>77.41(-3.31)</td><td>78.02(-2.73)</td></tr><tr><td>FlashPrefill V2-FP8</td><td>80.12</td><td>80.59</td><td>80.67</td><td>80.72</td><td>80.75</td></tr></table>

Table 9 | Ablation of the selection threshold � on RULER at 64K with Qwen3-4B-Instruct-2507 in FP8. Density is the fraction of selected attention blocks. Red numbers are the score drops of the uncorrected variant relative to the corrected pipeline at the same �. Full attention scores 82.81.

## 5. Conclusion

In this paper, we present FlashPrefill V2, which evolves FlashPrefill from an algorithmic prototype toward practical long-context prefilling. A zero-order mean correction term compensates pruned blocks with their pooled K/V statistics inside the softmax computation, confining the accuracy loss to within about one point on RULER and LongBench averages, and to within 1.8 points at 128K on RULER where fewer than 5% of the blocks are computed, while proving particularly effective under FP8 quantization. The redesigned sparse operator aligns with the FlashAttention-3/4 execution model—PackGQA memory access, warp-specialized producerconsumer pipelines, intra-warpgroup pingpong overlap, and FP8 execution—attaining up to 27.19× (BF16) and 47.26× (FP8) speedups over FlashAttention-2 at 128K, and a 30.49× speedup over an FA3/4-aligned dense baseline. With native support for paged KV cache and continuous batching, FlashPrefill V2 integrates into SGLang as a standard attention backend without any model-side changes, reducing end-to-end time-to-first-token by up to 3.4× (BF16) and 4.8× (FP8) at 128K under continuous batching. We hope this work helps bring sparse attention from algorithmic exploration toward practical serving.

## References

[1] A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. Gulavani, A. Tumanov, and R. Ramjee. Taming throughput-latency tradeoff in llm inference with sarathi-serve. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2024.

[2] Y. Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu, A. Zeng, L. Hou, Y. Dong, J. Tang, and J. Li. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024.

[3] I. Beltagy, M. E. Peters, and A. Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020. URL https://arxiv.org/abs/2004.05150.

[4] Z. Cai, Y. Zhang, B. Gao, Y. Liu, Y. Li, T. Liu, K. Lu, W. Xiong, Y. Dong, J. Hu, and W. Xiao. Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling. arXiv preprint arXiv:2406.02069, 2024.

[5] T. Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023. URL https://arxiv.org/abs/2307.08691.

[6] Q. Fan, H. Huang, Y. Ai, and R. He. Rectifying magnitude neglect in linear attention. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[7] Q. Fan, H. Huang, Z. Wu, B. Wang, and R. He. Uniprefill: Universal long-context prefill acceleration via block-wise dynamic sparsification. arXiv preprint arXiv:2605.06221, 2026.

[8] Q. Fan, H. Huang, Z. Wu, J. Wang, B. Wang, and R. He. Flashprefill: Instantaneous pattern discovery and thresholding for ultra-fast long-context prefilling. arXiv preprint arXiv:2603.06199, 2026.

[9] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[10] C.-P. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Jia, Y. Zhang, and B. Ginsburg. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024. URL https://arxiv.org/abs/2404.06654.

[11] H. Jiang, Y. Li, C. Zhang, Q. Wu, X. Luo, S. Ahn, Z. Han, A. H. Abdi, D. Li, C.-Y. Lin, Y. Yang, and L. Qiu. MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention. In Advances in Neural Information Processing Systems, 2024.

[12] N. Kitaev, Ł. Kaiser, and A. Levskaya. Reformer: The efficient transformer. arXiv preprint arXiv:2001.04451, 2020.

[13] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles (SOSP), 2023.

[14] X. Lai, J. Lu, Y. Luo, Y. Ma, and X. Zhou. Flexprefill: A context-aware sparse attention mechanism for efficient long-sequence inference. In International Conference on Learning Representations, 2025.

[15] H. Li, Y. Li, J. Chen, T. Ye, H. Liu, J. Yu, D. Wang, R. Zhang, Z. Xie, E. Xie, and S. Han. Sol-attn: Accelerating video generation inference via on-the-fly attention sparsification. arXiv preprint arXiv:2607.24027, 2026.

[16] Y. Li, Y. Huang, B. Yang, B. Venkitesh, A. Locatelli, H. Ye, T. Cai, P. Lewis, and D. Chen. Snapkv: Llm knows what you are looking for before generation. arXiv preprint arXiv:2404.14469, 2024.

[17] Y. Li, H. Jiang, C. Zhang, Q. Wu, X. Luo, S. Ahn, A. H. Abdi, D. Li, J. Gao, Y. Yang, and L. Qiu. Mminference: Accelerating pre-filling for long-context vlms via modality-aware permutation sparse attention. arXiv preprint arXiv:2504.16083, 2025.

[18] E. Lu, Z. Jiang, J. Liu, Y. Du, T. Jiang, C. Hong, S. Liu, W. He, E. Yuan, Y. Wang, Z. Huang, H. Yuan, S. Xu, X. Xu, G. Lai, Y. Chen, H. Zheng, J. Yan, J. Su, Y. Wu, N. Y. Zhang, Z. Yang, X. Zhou, M. Zhang, and J. Qiu. Moba: Mixture of block attention for long-context llms. arXiv preprint arXiv:2502.13189, 2025. URL https://arxiv.org/abs/2502.13189.

[19] B. Peng, R. Zhang, D. Goldstein, E. Alcaide, X. Du, H. Hou, J. Lin, J. Liu, J. Lu, W. Merrill, G. Song, K. Tan, S. Utpala, N. Wilce, J. S. Wind, T. Wu, D. Wuttke, and C. Zhou-Zheng. Rwkv-7 "goose" with expressive dynamic state evolution. arXiv preprint arXiv:2503.14456, 2025.

[20] R. Qin, Z. Li, W. He, M. Zhang, Y. Wu, W. Zheng, and X. Xu. Mooncake: A kvcache-centric disaggregated architecture for llm serving. arXiv preprint arXiv:2407.00079, 2024.

[21] L. Ribar, I. Chelombiev, L. Hudlass-Galley, C. Blake, C. Luschi, and D. Orr. Sparq attention: Bandwidth-efficient llm inference. arXiv preprint arXiv:2312.04985, 2023.

[22] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramade, and T. Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. arXiv preprint arXiv:2407.08608, 2024.

[23] H. Sun, Z. Chen, X. Yang, Y. Tian, and B. Chen. Triforce: Lossless acceleration of long sequence generation with hierarchical speculative decoding. arXiv preprint arXiv:2404.11912, 2024.

[24] Y. Sun, L. Dong, S. Huang, S. Ma, Y. Xia, J. Xue, J. Wang, and F. Wei. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.

[25] J. Tang, Y. Zhao, K. Zhu, G. Xiao, B. Kasikci, and S. Han. Quest: Query-aware sparsity for efficient long-context llm inference. arXiv preprint arXiv:2406.10774, 2024.

[26] Tencent Hunyuan AI Infra Team. HPC-Ops: High performance computing operators for large language model inference. https://github.com/Tencent/hpc-ops, 2026. Accessed: 2026-08-17.

[27] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[28] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[29] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.

[30] A. Vyas, A. Katharopoulos, and F. Fleuret. Fast transformers with clustered attention. arXiv preprint arXiv:2007.04825, 2020.

[31] C. Xiao, P. Zhang, X. Han, G. Xiao, Y. Lin, Z. Zhang, Z. Liu, and M. Sun. Infllm: Trainingfree long-context extrapolation for llms with an efficient context memory. arXiv preprint arXiv:2402.04617, 2024.

[32] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.

[33] G. Xiao, J. Tang, J. Zuo, J. Guo, S. Yang, H. Tang, Y. Fu, and S. Han. Duoattention: Efficient long-context llm inference with retrieval and streaming heads. arXiv preprint arXiv:2410.10819, 2024.

[34] G. Xiao, J. Guo, K. Mazaheri, and S. Han. Optimizing mixture of block attention. arXiv preprint arXiv:2511.11571, 2025. URL https://arxiv.org/abs/2511.11571.

[35] R. Xu, G. Xiao, H. Huang, J. Guo, and S. Han. Xattention: Block sparse attention with antidiagonal scoring. In International Conference on Machine Learning, 2025.

[36] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https://arxiv.org/abs/2505.09388.

[37] J. Yuan, H. Gao, D. Dai, J. Luo, L. Zhao, Z. Zhang, Z. Xie, Y. Wei, L. Wang, Z. Xiao, Y. Wang, C. Ruan, M. Zhang, W. Liang, and W. Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention. In Annual Meeting of the Association for Computational Linguistics, 2025.

[38] T. Zadouri, M. Hoehnerbach, J. Shah, T. Liu, V. Thakkar, and T. Dao. Flashattention-4: Algorithm and kernel pipelining co-design for asymmetric hardware scaling. arXiv preprint arXiv:2603.05451, 2026.

[39] M. Zaheer, G. Guruganesh, A. Dubey, J. Ainslie, C. Alberti, S. Ontanon, P. Pham, A. Ravula, Q. Wang, L. Yang, and A. Ahmed. Big bird: Transformers for longer sequences. arXiv preprint arXiv:2007.14062, 2020.

[40] Z. Zhang, Y. Sheng, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. In Advances in Neural Information Processing Systems, 2023.

[41] W. Zhao, Z. Zhou, Z. Su, C. Xiao, Y. Li, Y. Li, Y. Zhang, W. Zhao, Z. Li, Y. Huang, A. Sun, X. Han, and Z. Liu. Infllm-v2: Dense-sparse switchable attention for seamless short-to-long adaptation. arXiv preprint arXiv:2509.24663, 2025. URL https://arxiv.org/abs/2509 .24663.

[42] L. Zheng, L. Yin, Z. Xie, C. Sun, J. Huang, C. H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez, C. Barrett, and Y. Sheng. Sglang: Efficient execution of structured language model programs. In Advances in Neural Information Processing Systems, 2024.

[43] Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, and H. Zhang. Distserve: Disaggregating prefill and decoding for goodput-optimized large language model serving. arXiv preprint arXiv:2401.09670, 2024.

[44] K. Zhu, T. Tang, Q. Xu, Y. Gu, Z. Zeng, R. Kadekodi, L. Zhao, A. Li, A. Krishnamurthy, and B. Kasikci. Tactic: Adaptive sparse attention with clustering and distribution fitting for long-context llms. arXiv preprint arXiv:2502.12216, 2025.