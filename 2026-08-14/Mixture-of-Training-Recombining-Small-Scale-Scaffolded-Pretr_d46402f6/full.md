# Mixture of Training: Recombining Small-Scale Scaffolded Pretraining Runs into a Larger Language Model

Mohammed Sabry <sup>1</sup> <sup>2</sup> Sean Augenstein <sup>1</sup> Keith Rush <sup>1</sup> Lucio Dery <sup>1</sup>

## Abstract

We ask whether language-model pre-training can be decomposed into smaller, independently trainable jobs that can later be recomposed into a coherent larger model. We introduce Mixture of Training (MoT), a scaffolded modular pre-training procedure that partitions a target Transformer into contiguous layer blocks, trains each block inside a frozen pretrained aligner scaffold, and then recomposes the trained blocks with an optional short end-to-end adaptation pass. On a 1.3B-parameter Gemma-style model trained on C4, MoT provides a small-scale proof of mechanism: independently trained depth slices can be recomposed into a usable language model, and a quality-parity schedule reaches the same reported perplexity as the monolithic baseline. This parity setting processes more aggregate tokens and has a shorter idealized layer-equivalent critical path after aligner preparation; its effective compute advantage depends on reusing the aligner across runs. We therefore present MoT not as a general replacement for monolithic pre-training, but as a small-scale framework for studying whether scaffolded sub-runs can act as reusable training units.

## 1. Introduction

Large language model training is usually organized as one monolithic end-to-end optimization run. This creates a scaling bottleneck: all layers must be trained together, failures affect the whole run, and every improvement requires restarting or continuing a large coupled system. We ask whether pre-training can instead be decomposed into smaller, independentl trainable jobs that can later be recomposed into a coherent larger model. This question is especially natural in the small-scale regime targeted by MOSS: if small runs can serve as reusable scientific units, they may make training research cheaper, more reproducible, and easier to iterate on.

We introduce Mixture ofTraining (MoT), a scaffolded modular pre-training method. MoT partitions a target Transformer into contiguous layer blocks, trains each block inside a frozen pretrained aligner scaffold, and then recomposes the trained blocks into the full target model. The aligner supplies a stable representational interface during independent block training, so the blocks can be optimized without exchanging gradients while still remaining compatible at composition time.

Our study asks whether this decomposition can preserve language-model quality while changing the compute, tokenexposure, and critical-path trade-off. On a 12-layer, 1.3B-parameter Gemma-style language model trained on C4, MoT provides a small-scale proof of mechanism: independently trained depth slices can be recomposed into a coherent model, and a quality-parity schedule reaches the same reported perplexity as the monolithic baseline after end-to-end adaptation. This setting processes more aggregate tokens and has a shorter idealized layer-equivalent critical path after aligner preparation, but its effective compute advantage depends on amortizing the pretrained aligner across reuse cases. A lower-compute MoT schedule remains below the monolithic EFLOP budget even when the aligner is fully charged, but reaches slightly worse perplexity. Ablations show that the aligner is essential; under the tested aligner setting, disjoint submodel data streams improve cold composition; and increasing the number of splits trades quality for additional efficiency.

Our contributions are: (i) a scaffolded modular training procedure that trains target depth slices independently while preserving a shared representational interface; (ii) a small-scale proof-of-mechanism study showing that independently trained slices can be recomposed into a 1.3B-parameter language model; (iii) compute, token-exposure, and estimated critical-path accounting for cold composition, adapted composition, and quality-parity schedules, including explicit treatment of aligner amortization; and (iv) ablations showing how the aligner, data partitioning, and split count shape the observed quality–efficiency trade-off in this setting.

Positioning. MoT most directly adapts Deep Incubation’s scaffolded module-training pattern (Ni et al., 2023) to autoregressive language-model pre-training, examining flexible aligner depth, separately sampled submodel streams, end-to-end adaptation, and compute amortization. Unlike progressive growth, its target blocks train in parallel; unlike model averaging, distinct depth slices form a larger model; and unlike post-hoc stitching, compatibility is encouraged during training. DiffusionBlocks (Shing et al., 2026) also trains blocks independently but uses a diffusion-denoising interpretation, whereas MoT retains next-token prediction and an aligner scaffold. Appendix A provides broader comparisons.

## 2. Mixture of Training

Let the target Transformer be written as a composition of K contiguous layer blocks,

$$
F = f _ { K } \circ f _ { K - 1 } \circ \cdot \cdot \cdot \circ f _ { 1 } .
$$

Each $f _ { i }$ is a target submodel. MoT trains these submodels independently but not in isolation. It introduces a pretrained aligner $A = a _ { K } \circ \cdots \circ a _ { 1 }$ , sliced into the same number of blocks and shape-compatible with the target.

For each target block $f _ { i }$ , MoT constructs a scaffolded network

$$
S _ { i } = a _ { K } \circ \cdots \circ a _ { i + 1 } \circ f _ { i } \circ a _ { i - 1 } \circ \cdots \circ a _ { 1 } ,
$$

where only $f _ { i }$ is trainable and all aligner slices are frozen. Each scaffold is optimized with the standard next-token prediction loss. Thus, every target block learns inside the representational context supplied by the same aligner, but no gradients are exchanged between target blocks during scaffolded training.

Interface constraints. The aligner is required to be shape-compatible with the target model: it uses the same global width, attention-head dimensionality, feed-forward width, token embedding space, and output head. This lets aligner slices and target slices be swapped without adding projection or stitching layers. In this first study we restrict $f _ { i }$ to equal or near-equa contiguous layer blocks. Non-contiguous, neuron-level, or expert-level splits are possible, but they introduce additiona permutation or projection problems and are left to future work.

Embedding and output handling. Each scaffold reuses the aligner’s token embeddings and output head; only $f _ { i }$ is updated. The recomposed model retains these shared components but replaces the aligner’s Transformer slices with the trained target blocks, and all reported perplexities are measured on this recomposed model.

MoT has three stages. Stage 0 prepares or selects the aligner and partitions both aligner and target into corresponding blocks. Stage 1 trains all scaffolded networks $S _ { i }$ in parallel, updating only the target block in each scaffold. Stage 2 discards the aligner, recomposes $\hat { F } = f _ { K } \circ \cdot \cdot \cdot \circ f _ { 1 }$ , and optionally applies a short end-to-end adaptation pass. We call the recomposed model before Stage 2 adaptation the cold-composed model; its quality directly measures how well scaffolded training aligned the independently trained blocks.

## 3. Experiments and Results

We evaluate MoT on the English portion of C4. The target model is a 12-layer, 1.3B-parameter decoder-only Transformer following the Gemma-1-2B width configuration: 256k token vocabulary, 2048-dimensional embeddings, RoPE, multi-query attention, and 16384-dimensional feed-forward layers. Optimization uses AdamW with the same batch size across the baseline and MoT runs, but Stage 1 uses a different post-warm-up learning-rate schedule; full details appear in Appendix E. The monolithic baseline is trained end-to-end for 128k updates, processing 33.6B tokens and reaching perplexity 15.0 at 268.4 EFLOPs. Unless otherwise noted, MoT uses K = 2 target blocks, a 4-layer aligner, disjoint data streams for the two submodels, and evaluation is always performed on the recomposed target model rather than on individual scaffolds.

![](images/82a5cec5eebae9c377d22111fbf53d1bfa578142547c927522d7a1dc55d53fa3.jpg)  
Figure 1. Mixture of Training (MoT). The illustrated preparation and slicing steps form Stage 0; scaffolded block training is Stage 1; and recomposition with optional end-to-end adaptation is Stage 2. Only the target block in each scaffold is updated.

We report perplexity, the exponential of mean token-level cross-entropy on held-out C4. Baseline and MoT runs share the data source, optimizer family, and batch setting, with stage-specific learning-rate schedules reported in Appendix E. Because Stage 1 reorganizes compute and exposure across parallel submodels, we report training and fully charged EFLOPs, aggregate tokens, and an idealized layer-equivalent critical-path estimate rather than presenting the latter as measured wall-clock or equal-hardware speedup.

Table 1 summarizes the main quality–compute trade-off across three MoT schedules. Cold composition trains the two 6-layer submodels for 50k updates each and evaluates the recomposed model before any end-to-end adaptation. MoT + 15k adaptation adds a short Stage-2 pass to measure how much mismatch remains after independent scaffolded training. MoT quality parity reinvests part of the saved compute by extending submodel training to 75k updates and using a 30k adaptation pass. Because the 4-layer aligner costs 29.7 EFLOPs once, we report both direct MoT training EFLOPs and fully charged EFLOPs that assign the entire aligner cost to one run. We describe amortized reuse separately below rather than choosing a particular reuse count in the table. The critical-path derivation appears in Appendix D.

Table 1. Main results. EFLOPs follow Appendix B; “Train EF” excludes aligner preparation, while “Full-charge $\mathrm { E F } ^ { \prime }$ includes its entire 29.7 EFLOP cost. Tokens are aggregated across the schedule. The idealized layer-equivalent critical path assumes concurrent Stage 1 jobs after aligner preparation and is not a measured equal-hardware speedup.
<table><tr><td>Model / schedule</td><td>PPL↓</td><td>Train EF</td><td>Full-charge EF</td><td>Tokens (B)</td><td>Critical-path estimate</td></tr><tr><td>Monolithic baseline</td><td>15.0</td><td>268.4</td><td>268.4</td><td>33.6</td><td>1.0×</td></tr><tr><td>MoT cold composition</td><td>19.3</td><td>128.2</td><td>157.9</td><td>26.2</td><td>4.2×</td></tr><tr><td>MoT + 15k adaptation</td><td>15.9</td><td>159.7</td><td>189.4</td><td>30.1</td><td>2.8×</td></tr><tr><td>MoT quality parity</td><td>15.0</td><td>255.3</td><td>285.0</td><td>47.1</td><td>1.7×</td></tr></table>

If the aligner is charged fully to a single run, the quality-parity schedule costs 255.3 + 29.7 = 285.0 EFLOPs, above the 268.4 EFLOP monolithic baseline. Quality parity is therefore not compute-saving in the fully charged single-run setting. More generally, if the same 29.7 EFLOP aligner is reused across R independent, shape-compatible target-model training runs, the effective quality-parity cost per run is

$$
2 5 5 . 3 + { \frac { 2 9 . 7 } { R } } .
$$

Here, R counts independent target-model runs that reuse the same aligner, not checkpoints or ablations from one training effort. The effective cost falls below the monolithic baseline for $R \geq 3 ,$ , so we interpret the quality-parity result as evidence for an amortized-reuse regime rather than an unconditional efficiency gain. The $5 0 \mathbf { k } + 1 5 \mathbf { k }$ schedule remains below the full 128k-step baseline budget even when the aligner is fully charged, reaching PPL 15.9 at $1 2 8 . 2 + 3 1 . 5 + 2 9 . 7 = 1 8 9 . 4$ EFLOPs. However, because we do not report a monolithic checkpoint trained or tuned at the same 189.4 EFLOP budget, this comparison does not establish an equal-compute advantage. Finally, the critical-path estimates assume that the two Stage 1 scaffold jobs run concurrently and exclude aligner preparation; they characterize time-to-result under additiona parallel hardware, not measured wall-clock time under equal resources.

The aligner is critical in our setting (Table 2; full matrix in Appendix F). Without it, cold-composition quality drops sharply. Disjoint streams improve PPL under the tested 4-layer aligner but worsen it without an aligner. Increasing K from 2 to 4 lowers compute but worsens cold-composition PPL, exposing a quality–efficiency trade-off.

Table 2. Compact cold-composition ablations. As in the “Full-charge EF” column of Table 1, this table charges the full 29.7 EFLOP aligner cost where present.
<table><tr><td>Setting</td><td>PPL↓</td><td>EFLOPs</td><td>Takeaway</td></tr><tr><td>No aligner, K = 2, shared data</td><td>38.9</td><td>104.8</td><td>incompatible</td></tr><tr><td>Aligner, K = 2, shared data</td><td>20.3</td><td>158</td><td>scaffold helps</td></tr><tr><td>Aligner, K = 2, disjoint data</td><td>19.3</td><td>158</td><td>best cold PPL</td></tr><tr><td>Aligner, K = 4, disjoint data</td><td>24.8</td><td>110</td><td>cheaper, worse PPL</td></tr></table>

These ablations provide behavioural evidence for the role of the aligner, but they do not by themselves fully diagnose the internal mechanism. The large degradation without an aligner is consistent with an interface-mismatch failure: independently trained depth slices do not necessarily produce hidden states that the next slice can use. The frozen aligner plausibly reduces this mismatch by making every slice train against the same surrounding representational context. Under the tested aligner settings, disjoint data streams improve the cold-composed model, suggesting that the recomposed target can benefit from broader token exposure when the scaffold preserves a shared interface. Increasing K gives more parallelism and lower Stage-1 compute, but each trainable block becomes smaller and the number of interfaces increases, so cold-composition quality drops. A fuller mechanistic account would require direct interface diagnostics, such as hidden-state similarity across boundaries, activation-norm drift, CKA/SVCCA comparisons, or layer-wise loss probes before and after recomposition.

## 4. Discussion and Limitations

The experiments support three conclusions. First, independently trained submodels can be recomposed into a coherent language model when trained inside a shared aligner scaffold; without the scaffold, cold composition degrades sharply. Second, the remaining mismatch is mostly recoverable rather than catastrophic: a short adaptation pass closes most of the gap, and a longer quality-parity schedule reaches the same reported perplexity as the monolithic baseline. Third, MoT changes the engineering shape of pre-training. Instead of one coupled run, it creates smaller jobs that can be scheduled, restarted, and ablated independently, and potentially reused across compatible training efforts, making it especially suitable for small-scale training research.

The current study is intentionally small-scale. It uses one model family, one dataset, and a limited set of schedules, and reports perplexity rather than downstream reasoning, factuality, calibration, or robustness benchmarks. We also do not yet include all compute-matched monolithic controls, such as baselines matched for fully charged MoT EFLOPs, aggregate token exposure, or estimated critical-path budget, nor do we report measured wall-clock time under equal hardware resources. Consequently, the current evidence demonstrates modular composability in this setting and identifies an amortized-reuse scheduling regime, rather than a uniform compute or equal-hardware speed advantage. As a proof of mechanism, MoT opens a concrete design space for reusable scaffolded sub-runs and their quality–compute–scheduling trade-offs.

## References

Bansal, Y., Nakkiran, P., and Barak, B. Revisiting model stitching to compare neural representations. CoRR, abs/2106.07682, 2021. URL https://arxiv.org/abs/2106.07682.

Chen, C., Yin, Y., Shang, L., Jiang, X., Qin, Y., Wang, F., Wang, Z., Chen, X., Liu, Z., and Liu, Q. bert2BERT: Towards

reusable pretrained language models. In Muresan, S., Nakov, P., and Villavicencio, A. (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2134–2148, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.151. URL https://aclanthology.org/2022.acl-long.151/.

Du, W., Luo, T., Qiu, Z., Huang, Z., Shen, Y., Cheng, R., Guo, Y., and Fu, J. Stacking your transformers: A closer look at model growth for efficient LLM pre-training. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-0336. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ 143ea4a156ef64f32d4d905206cf32e1-Abstract-Conference.html.

Gemma Team. Gemma: Open models based on gemini research and technology, 2024. URL https://arxiv.org/ abs/2403.08295.

Hernandez, A., Dangovski, R., Lu, P. Y., and Soljacic, M. Model stitching: Looking for functional similarity between representations, 2023. URL https://arxiv.org/abs/2303.11277.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J. W., Vinyals, O., and Sifre, L. Training compute-optimal large language models, 2022. URL https://arxiv.org/abs/2203.15556.

Jiang, Z., Huang, J., Chen, Z., Li, Y., Yu, G., Feng, C., Yang, Y., Yang, Z., and Lyu, M. R. L4: Diagnosing large-scale llm training failures via automated log analysis, 2025. URL https://arxiv.org/abs/2503.20263.

Lo, K. M., Liang, Y., Du, W., Fan, Y., Wang, Z., Huang, W., Ma, L., and Fu, J. M2mKD: Module-to-module knowledge distillation for modular transformers. arXiv preprint arXiv:2402.16918, 2024. URL https://arxiv.org/abs/ 2402.16918.

Mcdonald, R., Mohri, M., Silberman, N., Walker, D., and Mann, G. Efficient large-scale distributed training of conditional maximum entropy models. In Bengio, Y., Schuurmans, D., Lafferty, J., Williams, C., and Culotta, A. (eds.), Advances in Neural Information Processing Systems, volume 22. Curran Associates, Inc., 2009. URL https://proceedings.neurips.cc/paper\_files/paper/2009/file/ d81f9c1be2e08964bf9f24b15f0e4900-Paper.pdf.

Ni, Z., Wang, Y., Yu, J., Jiang, H., Cao, Y., and Huang, G. Deep incubation: Training large models by divide-andconquering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 17335–17345, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Ni\_Deep\_Incubation\_ Training\_Large\_Models\_by\_Divide-and-Conquering\_ICCV\_2023\_paper.html.

Shing, M., Koyama, M., and Akiba, T. Diffusionblocks: Block-wise neural network training via diffusion interpretation, 2026. URL https://arxiv.org/abs/2506.14202.

Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., and Schmidt, L. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time, 2022. URL https://arxiv.org/abs/2203.05482.

Yao, Y., Zhang, Z., Li, J., and Wang, Y. Masked structural growth for 2x faster language model pre-training. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=rL7xsg1aRn.

## A. Additional Related Work

Progressive / Growth Strategies: Progressive depth or capacity growth (Chen et al., 2022; Yao et al., 2024; Du et al., 2024) trains a smaller network first and incrementally adds layers or width, continuing end-to-end optimization of the enlarged model; this saves early compute but is inherently sequential and any early representational biases propagate forward. In contrast, MoT trains all planned layer groups in parallel from the outset under a shared aligner interface, turning depth segmentation into a parallel rather than temporal schedule and decoupling failures or slow convergence in one segment from others.

Model Merging and Weight Averaging: Model soups (Wortsman et al., 2022) (simple parameter averaging of multiple finetuned checkpoints) and earlier weight-averaging / merging techniques combine identically-shaped models to improve accuracy or robustness without inference cost increase, relying on the empirical flatness / basin connectivity of fine-tuned solutions. However, merging does not enlarge capacity and can lose complementary information if models are not well aligned; alignment is incidental, arising from starting from a common pretrained seed. In MoT each submodel occupies a distinct layer subset so assembling increases overall depth coverage (capacity) beyond any single trained component, and alignment can be actively enforced throughout submodel training by the frozen aligner interface, rather than assumed at the moment of averaging.

Parallel Learning and Loosely-Coupled Training: Weight averaging and model merging techniques have also found homes at training time in Local SGD (dating to at least (Mcdonald et al., 2009)) and its many related methods. These methods share the core technique of independently evolving models before leveraging some kind of parameter- or model delta-space merging. In the most extreme cases this takes the form of a one-shot average, in the style of model souping; often the technique is applied iteratively. MoT bears some resemblance to these local-training based methods, but crucially differs in the manner of composition, with the explicit goal of the composed model having higher capacity than any of its locally-trained slices.

Model Stitching and Representation Alignment: Model stitching research (Bansal et al., 2021; Hernandez et al., 2023) examines substituting intermediate layers between pretrained networks and inserting “stitch” projection layers to reconcile activation spaces, enabling analysis or hybrid reuse; success depends on latent space compatibility and often requires additional bridging parameters. MoT avoids explicit stitching layers at assembly time because its aligner enforces a stable intermediate representation contract during every submodel’s training, yielding direct composability without post-hoc adapters, and allowing frozen aligner parts to be discarded after composition.

Divide-and-Conquer and Block-wise Training: Deep Incubation (Ni et al., 2023; Lo et al., 2024) divides large vision transformers into submodules trained to replace segments of a small global “meta model,” improving ImageNet accuracy and reducing training cost; the meta model implicitly links modules. MoT applies the same broad divide-and-conquer pattern to autoregressive language-model pre-training. The present study emphasizes five axes in this setting: (i) a flexible aligner scale (not locked to one layer per module); (ii) pretrained aligners as scaffold interfaces, with instruction- or value-aligned variants left as future work; (iii) multiple split granularities and separately sampled data streams per submodel; (iv) explicit FLOP, memory, critical-path, and amortization accounting; and (v) end-to-end adaptation after cold composition.

Recent methods such as DiffusionBlocks (Shing et al., 2026) also train blocks independently, but cast residual blocks as diffusion denoising steps over assigned noise ranges; MoT instead keeps the autoregressive next-token objective and uses an aligner scaffold to make independently trained depth slices composable.

Hardware Fault Tolerance and Training Failures: Large monolithic training runs face hardware faults, instability, and silent performance regressions; recent analyses (Jiang et al., 2025) of large-scale logs underscore the operational complexity and failure modes in multi-week LLM training. By decomposing the workload into independently restartable submodel jobs, MoT could reduce the amount of work affected by a failure and simplify recovery because only the affected scaffold job would need to resume. We do not measure fault-recovery behavior here, so this remains a prospective systems benefit rather than an empirical result.

Summary: Existing approaches ensure compatibility of distributed components through continuous activation and gradient synchronization (data/model/pipeline parallelism), post-hoc adapters (stitching), parameter merging (model soups and local SGD), or a shared meta-model scaffold (Deep Incubation). Relative to this literature, MoT studies the scaffolded divide-and conquer pattern for autoregressive language-model pre-training, including flexible aligner depth, disjoint submodel data streams, end-to-end adaptation, and explicit FLOP, memory, critical-path, and amortization accounting. Behaviour-specific aligners and reuse across independent target-model runs remain directions for future evaluation.

## B. Compute Cost Analysis

## B.1. Baseline pre-training

For a transformer with N non-embedding parameters and a training corpus of D tokens, the approximate forward–backward cost is

$$
\begin{array} { r } { C _ { \mathrm { b a s e l i n e } } \approx 6 N D , } \end{array}
$$

where the factor six is the standard training-FLOP approximation: roughly 2ND FLOPs for the forward pass and 4ND FLOPs for the backward pass. This layer-dominated approximation excludes embedding lookup, output projection/softmax, input-pipeline, and other fixed costs; the 256k-vocabulary output head may therefore make the reported values optimistic as full-system compute estimates. When the token–to–parameter ratio follows the Chinchilla rule (Hoffmann et al., 2022) $M = D / N = 2 0$ , the expression reduces to

$$
C _ { \mathrm { b a s e l i n e } } \approx 1 2 0 N ^ { 2 } .
$$

## B.2. MoT pre-training

MoT partitions the target model into K trainable submodels of size $N _ { M }$ and introduces an aligner of size $N _ { A }$ , sliced into the same K segments and kept frozen. Let $D _ { M }$ be the token budget per submodel and $D _ { \mathrm { a d a p t } }$ the token budget for Stage 2 adaptation. Using the same layer-dominated approximation, a conservative per-submodel upper bound for Stage 1 is

$$
\begin{array} { r l } { C _ { \mathrm { m o t s u b m o d e l } } = \ 2 \big ( N _ { M } + ( K - 1 ) N _ { A } / K \big ) D _ { M } \ } & { ( \mathrm { f o r w a r d ~ p a s s } ) } \\ { + \ 4 N _ { M } D _ { M } \ } & { ( \mathrm { w e i g h ~ \& ~ c t i v a t i o n ~ g r a d i e n t s } ) } \\ { + \ 2 ( K - 1 ) N _ { A } / K D _ { M } \ } & { ( \mathrm { a c t i v a t i o n ~ g r a d i e n t s ~ f o r ~ f r o z e n ~ s l i c e s } ) . } \end{array}
$$

The final activation-gradient term charges every frozen slice as though it lies downstream of the trainable block. In an implementation that stops gradients through upstream frozen slices, this overcounts those upstream slices; we retain it as a transparent conservative bound. Because all K scaffolded models are trained in parallel, the aggregate Stage 1 cost under this accounting is $K \times C _ { \mathrm { m o t s u b m o d e l } } .$

Stage 2 adapts the recomposed model and costs $C _ { \mathrm { a d a p t } } \approx 6 N D _ { \mathrm { a d a p t } }$

Hence the overall MoT budget is

$$
C _ { \mathrm { M o T } } = C _ { \mathrm { a l i g n } } + K C _ { \mathrm { m o t s u b m o d e l } } + C _ { \mathrm { a d a p t } } ,
$$

where $C _ { \mathrm { a l i g n } }$ is the Stage 0 cost. In many scenarios the aligner can be reused from an off-the-shelf checkpoint, so we treat $C _ { \mathrm { a l i g n } }$ as amortizable. The main text reports direct and fully charged costs, then gives the amortized per-run cost explicitly as a function of the number of independent target-model runs that reuse the aligner.

Numerical example: For the 1.3B Gemma-style setting with K = 2 splits and a 4-layer aligner the reported layerequivalent cost is:

• Stage 0 (aligner): 29.7 EFLOPs

• Stage 1 (submodels): 128.2 EFLOPs

• Stage 2 (15 k steps): 31.5 EFLOPs

The total of 189.4 EFLOPs is 29% below the full 128k-step baseline budget of 268.4 EFLOPs while reaching PPL 15.9, closing most of the cold-composition gap after a 15k-step adaptation. This is not an equal-compute comparison: we do not report a monolithic checkpoint trained or tuned at the same 189.4 EFLOP budget.

## C. Memory Footprint Analysis

This appendix complements the FLOP and critical-path analysis by accounting for the memory terms introduced by scaffolded training. During Stage 1, each scaffold stores optimizer states and parameter gradients only for the trainable target block, while frozen aligner slices still contribute parameter storage and forward activations. As in Appendix B, N counts non-embedding Transformer parameters: shared token-embedding and output-head storage must be added for a full implementation-level estimate and may be material with a 256k-token vocabulary. The resulting peak memory therefore depends on the balance between reduced trainable-state storage and additional scaffold activations, as well as implementation choices such as activation checkpointing, recomputation, sharding, and attention kernels.

## C.1. Baseline memory usage

With FP32 parameters, AdamW optimization and no activation recomputation, a full end-to-end transformer pretrain requires

$$
\underbrace { 4 N } _ { \mathrm { m o d e l } } + \underbrace { 8 N } _ { \mathrm { A d a m W \ m o m e n t s ^ { \dagger } } } + \underbrace { 4 N } _ { \mathrm { g r a d i e n t s } } + \underbrace { 2 0 \ B T H L } _ { \mathrm { a c t i v a t i o n s } } \quad { \mathrm { b y t e s } } .
$$

Here N is the number of non-embedding parameters, B the local batch represented by the footprint, T the sequence length, H the hidden size and L the number of layers.

† In pure FP32 training, AdamW updates the model parameter directly and stores two additional FP32 moment tensors $( m , v )$ , for $2 \times 4 = 8$ bytes per parameter. A separate FP32 master-weight copy, sometimes used in mixed-precision training, is not assumed here.

Illustrative numerical example: The following calculation uses an 18-layer Gemma-2B-scale configuration to illustrate the memory terms; it is not the exact 12-layer, 1.3B headline run or a report of its hardware allocation. For an unsharded logical replica with N = 2B, B = 256, T = 1024, H = 2048, and L = 18, the baseline footprint is

$$
\mathrm { m o d e l = 8 G B , \quad A d a m W \ m o m e n t s = 1 6 G B , \quad g r a d s = 8 G B , \quad a c t s = 1 9 3 G B , }
$$

for a total of roughly 225 GB before parameter, optimizer-state, activation, or batch sharding.

## C.2. MoT memory usage

During Stage 1 each scaffolded model “frozen aligner slices + trainable submodel” holds

$$
\big ( 4 N _ { M } + 4 ( K - 1 ) N _ { A } / K \big ) \ + \ 8 N _ { M } + 4 N _ { M } \ + \ 2 0 B T H L _ { \mathrm { h y b } } \quad \mathrm { b y t e s } ,
$$

where $L _ { \mathrm { h y b } }$ denotes the number of layers executed by the scaffold, including the trainable target block and the retained frozen aligner slices. The terms correspond to:

• model-parameter storage for the trainable target block and retained frozen aligner slices,

• AdamW optimizer states and gradients only for the trainable submodel, since the aligner parameters are frozen, and

• forward activations for both the frozen aligner slices and the trainable submodel.

Table 3. Illustrative unsharded memory footprint for an 18-layer Gemma-2B-scale baseline versus one MoT scaffold (K = 2) in Stage 1. Each scaffold trains half of the target layers and retains half of the frozen aligner. This is an accounting example rather than the exact 12-layer, 1.3B headline configuration or a measured per-device footprint.
<table><tr><td></td><td>Model</td><td>Adam moments</td><td>Grads</td><td>Acts</td><td>Total</td></tr><tr><td>Baseline (18L)</td><td>8.0 GB</td><td>16.0 GB</td><td>8.0 GB</td><td>193 GB</td><td>225 GB</td></tr><tr><td>MoT, 4-L aligner</td><td>4.8 GB</td><td>8.0 GB</td><td>4.0 GB</td><td>118 GB</td><td>135 GB</td></tr><tr><td>MoT, 8-L aligner</td><td>5.6 GB</td><td>8.0 GB</td><td>4.0 GB</td><td>139 GB</td><td>157 GB</td></tr></table>

The table separates the main memory terms rather than reducing them to a single headline claim. MoT reduces optimizer and gradient storage for the trainable component of each scaffold, but the frozen aligner slices add parameter and activation terms. This makes the peak memory footprint implementation-dependent, especially under different activation-checkpointing or recomputation strategies.

With more than two splits, each individual aligner slice is smaller, but each scaffold contains K − 1 frozen aligner slices; the total Stage-1 footprint therefore depends on the trade-off between smaller trainable blocks and the retained frozen scaffold depth.

## D. Idealized Layer-Equivalent Critical-Path Analysis

The critical-path estimates reported in the main text are idealized schedule-level calculations, not measured end-to-end wall-clock timings. They assume that the K scaffolded Stage 1 jobs can be scheduled concurrently, so the elapsed time is determined by the slowest scaffold job rather than by the sum of all scaffold jobs. The calculation assigns equal cost to each Transformer layer and omits embedding lookup, output projection/softmax, data loading, communication, and other fixed costs. It is useful for describing the schedule’s layer-equivalent dependency path under available parallel hardware, but it should not be read as a same-accelerator-count throughput measurement. A fully measured systems comparison would need to account for hardware allocation, utilization, communication overheads, input-pipeline bottlenecks, activation checkpointing, and implementation-specific kernel efficiency.

A baseline step touches all $L _ { \mathrm { f u l l } }$ layers and costs $t _ { \mathrm { f u l l } }$ time unit per step. For consistency with the FLOP accounting in Appendix B, we assign one unit to a forward pass, two units to the full backward pass of a trainable layer, and one additional unit to propagating activation gradients through a frozen downstream layer. In the slower of the two $K = 2$ scaffolds, the frozen aligner slice follows the trainable target block and therefore lies on the backward path. The slower-scaffold-to-baseline layer-equivalent step-cost ratio is consequently

$$
c _ { \mathrm { s c a f f o l d } } = \frac { 3 L _ { \mathrm { t r a i n } } + 2 L _ { \mathrm { f r o z e n } } } { 3 L _ { \mathrm { f u l l } } } .
$$

In our setup each scaffold in Stage 1 contains $L _ { \mathrm { t r a i n } } = 6$ learnable layers (the split half of the 12-layer target) and $L _ { \mathrm { f r o z e n } } = 2$ frozen aligner layers (the 4-layer aligner is split evenly across the two scaffolds). Hence

$$
c _ { \mathrm { s c a f f o l d } } = { \frac { 3 L _ { \mathrm { t r a i n } } + 2 L _ { \mathrm { f r o z e n } } } { 3 L _ { \mathrm { f u l l } } } } = { \frac { 3 \times 6 + 2 \times 2 } { 3 \times 1 2 } } \approx 0 . 6 1 .
$$

Because the two scaffolds advance in lock-step, Stage 1 lasts $S _ { 1 }$ iterations and Stage 2 lasts $S _ { 2 }$ . With a 128 k-step monolithic baseline, the elapsed time is:

$$
T _ { \mathrm { M o T } } = S _ { 1 } c _ { \mathrm { s c a f f o l d } } t _ { \mathrm { f u l l } } + S _ { 2 } t _ { \mathrm { f u l l } } , \quad \mathrm { c r i t i c a l - p a t h ~ e s t i m a t e } = \frac { T _ { \mathrm { b a s e l i n e } } } { T _ { \mathrm { M o T } } } = \frac { 1 2 8 \mathrm { k } } { S _ { 1 } c _ { \mathrm { s c a f f o l d } } + S _ { 2 } } .
$$

Table 4 plugs in the schedules reported in Table 1. These are critical-path estimates after aligner preparation, not measurements that include Stage 0 aligner training time.

Table 4. Idealized layer-equivalent critical-path estimates for the main MoT settings after aligner preparation. The baseline trains for 128k full-model steps. For the 4-layer aligner with $K = 2 ,$ , the slower scaffold has $c _ { \mathrm { s c a f f o l d } } \approx 0 . 6 1$ . These values assume concurrent Stage 1 jobs and are not measured equal-hardware speedups.
<table><tr><td>Schedule</td><td> $S _ { 1 }$ </td><td> $S _ { 2 }$ </td><td> $c _ { \mathrm { s c a f f o l d } }$ </td><td>Time (k step-equiv.)</td><td>Critical-path estimate</td></tr><tr><td>Monolithic baseline</td><td></td><td>一</td><td></td><td>128.0</td><td>1.0×</td></tr><tr><td>MoT cold composition</td><td>50k</td><td>0k</td><td>0.61</td><td>30.6</td><td>4.2×</td></tr><tr><td>MoT + 15k adaptation</td><td>50k</td><td>15k</td><td>0.61</td><td>45.6</td><td>2.8×</td></tr><tr><td>MoT quality parity</td><td>75k</td><td>30k</td><td>0.61</td><td>75.8</td><td>1.7×</td></tr></table>

## E. Aligner Preparation

Experimental configuration. All runs use a batch size of 256 sequences of length 1024, AdamW with peak learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , and $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ ). The baseline, aligner-preparation runs, and Stage 2 adaptation use linear warm-up over the first 10% of their token budget followed by cosine decay. Stage 1 scaffold training instead uses the same warm-up fraction followed by a constant peak learning rate; a separate three-seed scheduler ablation found similar loss trajectories for warm-up–constant and warm-up–cosine schedules in a smaller 4-layer target setting. The models use post-normalization in the attention and feed-forward blocks and final-logit softening of 30. The main results are individual runs; only the scheduler ablation was replicated.

All aligners share the Gemma-1-2B (Gemma Team, 2024) width configuration but vary in depth and token budget (Table 5).   
Except for depth and token budget, aligner-preparation hyperparameters match the baseline.

Because MoT is meant to reuse off-the-shelf or behavior-specific guides, we adopt a weak form of data decoupling in this first study: aligners process independently seeded C4 dataloader streams containing 10–40B tokens. We did not enforce sample-level de-duplication or measure realized overlap, so these should be understood as separately sampled streams rather than disjoint corpora. This setup tests whether submodels can remain compositionally compatible under an aligner exposed to a partially different sample stream, a minimal proxy for the “different priors” scenario we aim to explore in future work.

Table 5. Aligner configurations and pre-training cost.
<table><tr><td>Aligner</td><td>#Params (B)</td><td>Token budget</td><td>PPL↓</td><td>FLOPs (EF)</td></tr><tr><td>4 layers, M=25</td><td>0.4</td><td>10B</td><td>20.7</td><td>29.7</td></tr><tr><td>4 layers, M=50</td><td>0.4</td><td>20 B</td><td>18.4</td><td>59.4</td></tr><tr><td>4 layers, M=100</td><td>0.4</td><td>40 B</td><td>17.6</td><td>118.8</td></tr><tr><td>6 layers, M=25</td><td>0.6</td><td>15B</td><td>17.8</td><td>66.5</td></tr><tr><td>8 layers, M=25</td><td>0.8</td><td>20 B</td><td>16.4</td><td>118.4</td></tr></table>

## F. Ablation Studies

## F.1. Effect of aligner size, split count, and batch sharing

Table 6 summarizes the main ablations. Models without an aligner or with four splits (K=4) save more compute but suffer higher perplexity. Under the 4-layer aligner, training each submodel on disjoint mini-batches outperforms shared-batch training; without an aligner, the same change worsens perplexity.

Table 6. Detailed ablations on aligner, batch sharing and split count for the 12-layer target model. Numbers to the left of “+” exclude the one-time 29.7 EF aligner cost; the arrow shows the total once that cost is added.
<table><tr><td>Aligner</td><td>Batch</td><td>Splits (K)</td><td>PPL↓</td><td>Compute (EF)</td></tr><tr><td>None</td><td>Same</td><td>2</td><td>38.9</td><td>104.8</td></tr><tr><td>None</td><td>Same</td><td>4</td><td>478.2</td><td>48.2</td></tr><tr><td>None</td><td>Different</td><td>2</td><td>50.4</td><td>104.8</td></tr><tr><td>None</td><td>Different</td><td>4</td><td>584.0</td><td>48.2</td></tr><tr><td>4-layer aligner (M=25)</td><td>Same</td><td>2</td><td>20.3</td><td> $1 2 8 . 2 + 2 9 . 7 \  \ 1 5 7 . 9$ </td></tr><tr><td>4-layer aligner (M=25)</td><td>Same</td><td>4</td><td>28.5</td><td> $8 0 . 4 + 2 9 . 7 \  \ 1 1 0 . 1$ </td></tr><tr><td>4-layer aligner (M=25)</td><td>Different</td><td>2</td><td>19.3</td><td> $1 2 8 . 2 + 2 9 . 7 \  \ 1 5 7 . 9$ </td></tr><tr><td>4-layer aligner (M=25)</td><td>Different</td><td>4</td><td>24.8</td><td> $8 0 . 4 + 2 9 . 7 \  \ 1 1 0 . 1$ </td></tr></table>

## Observations:

• Aligner vs. no aligner: Removing the aligner leads to a dramatic quality collapse (PPL rising from 19−20 to 39+), confirming that the aligner’s representational scaffold is essential.

• Shared vs. disjoint batches: Under the 4-layer aligner, different mini-batches improve PPL for both K = 2 (20.3→19.3) and $K = 4 ( 2 8 . 5  2 4 . 8 )$ . Without an aligner, they worsen PPL (38.9→50.4 and 478.2→584.0). Thus, broader token coverage helps in these runs only when the scaffold preserves interface compatibility.

• Split count: Increasing the splits from two to four further reduces Stage 1 compute by approximately 37% but degrades quality unless offset by a stronger adaptation. The trade-off illustrates MoT’s ability to dial between speed and accuracy.

## F.2. MoT Quality Trajectory

Cold-composition quality: Table 7 reports C4 validation perplexity and total FLOPs after completing only Stage 0 (aligner training) and Stage 1 (parallel submodel training), before any end-to-end adaptation. Although no layer sees gradients from the full network, the reassembled model remains within four to five perplexity points of the baseline. The four-layer M = 50 setting is one favorable point on the observed frontier: it reaches PPL 18.9 at 187.6 EFLOPs, approximately 30% below the fully trained baseline budget, while the M = 100 setting reaches the same reported PPL at higher compute.

Table 7. Cold Composition performance after Stage 0 + 1 (K=2 splits, no adaptation). Total FLOPs are shown as Stage 1 cost + Stage 0 cost → summed total.
<table><tr><td>Configuration</td><td>PPL↓</td><td>Total FLOPs (EF)</td></tr><tr><td>Baseline</td><td>15.0</td><td>268.4</td></tr><tr><td>MoT, 4-layer aligner (M=25)</td><td>19.3</td><td> $1 2 8 . 2 + 2 9 . 7 \  \ 1 5 7 . 9$ </td></tr><tr><td>MoT, 4-layer aligner (M=50)</td><td>18.9</td><td> $1 2 8 . 2 + 5 9 . 4 \  \ 1 8 7 . 6$ </td></tr><tr><td>MoT, 4-1ayer aligner (M=100)</td><td>18.9</td><td> $1 2 8 . 2 + 1 1 8 . 8 \  \ 2 4 7 . 0$ </td></tr><tr><td>MoT, 6-layer aligner (M=25)</td><td>19.3</td><td> $1 3 9 . 8 + 6 6 . 5 \ :  \ : 2 0 6 . 3$ </td></tr><tr><td>MoT, 8-layer aligner (M=25)</td><td>19.7</td><td> $1 5 1 . 5 + 1 1 8 . 4  2 6 9 . 9$ </td></tr></table>

Increasing aligner depth or token budget beyond this setting yields no consistent cold-composition improvement while increasing total compute. These runs do not identify the mechanism behind this pattern; direct interface diagnostics would be needed to test whether aligner capacity constrains the target blocks.

Adaptation benefit: Table 8 appends one end-to-end adaptation pass to every cold-composition run: 15 k steps (∼3.9 B tokens, 31.5 EF) for the 4-layer (M=25, 50) and 6-layer aligners, and a shorter 10 k pass (∼ 2.6 B tokens, 21.0 EF) for the largest 4-layer (M=100) and 8-layer aligners to keep their total compute in the same ballpark as the baseline. Even this short adaptation pass closes most of the gap: with the 4-layer aligner at M=25 the model reaches PPL = 15.9 while stil saving 29% of the baseline EFLOPs when the aligner cost is included and 40% when it is excluded. Additional aligner depth or token budget does not improve adapted PPL in these runs and reduces or reverses the compute savings. Because the largest 4-layer and 8-layer aligners receive only 10k adaptation steps rather than 15k, these rows are not controlled comparisons of aligner capacity alone.

Table 8. Perplexity after the Stage-2 adaptation pass. Stage-2 EFLOPs are 31.5 (15 k) or 21.0 (10 k). Total FLOPs are shown as Stage 2 cost + Stage 1 cost + Stage 0 cost → summed total.
<table><tr><td>Configuration</td><td>Stage-2 steps</td><td>PPL↓</td><td>Total EFLOPs</td></tr><tr><td>Baseline</td><td></td><td>15.0</td><td>268.4</td></tr><tr><td>MoT, 4-layer aligner (M=25)</td><td>15 k</td><td>15.9</td><td> $3 1 . 5 + 1 2 8 . 2 + 2 9 . 7  1 8 9 . 4$ </td></tr><tr><td>MoT, 4-layer aligner (M=50)</td><td>15 k</td><td>15.9</td><td> $3 1 . 5 + 1 2 8 . 2 + 5 9 . 4 \  \ 2 1 9 . 1$ </td></tr><tr><td>MoT, 4-layer aligner (M=100)</td><td>10 k</td><td>16.4</td><td> $2 1 . 0 + 1 2 8 . 2 + 1 1 8 . 8 \ \to \ 2 6 8 . 0$ </td></tr><tr><td>MoT, 6-layer aligner (M=25)</td><td>15 k</td><td>16.0</td><td> $3 1 . 5 + 1 3 9 . 8 + 6 6 . 5  2 3 7 . 8$ </td></tr><tr><td>MoT, 8-layer aligner (M=25)</td><td>10 k</td><td>16.7</td><td> $2 1 . 0 + 1 5 1 . 5 + 1 1 8 . 4  2 9 0 . 9$ </td></tr></table>