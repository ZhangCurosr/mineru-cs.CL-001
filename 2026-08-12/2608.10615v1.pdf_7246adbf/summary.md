---
title: "Simplex Relaxation for Discrete Diffusion"
source: https://arxiv.org/pdf/2608.10615v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:59:49"
field: "离散扩散生成模型"
keywords: ["discrete diffusion", "uniform diffusion", "simplex relaxation", "Dirichlet-categorical augmentation", "Rao-Blackwellized objective", "OpenWebText", "Sudoku generation"]
innovations: ["在保持均匀离散扩散分类边缘不变的前提下引入精确 Dirichlet-单纯形增广", "通过辅助分类解码的 Rao-Blackwellized 闭式反桥目标消除采样噪声", "基于同一增广层次导出保留分类输入的随机祖先采样器"]
benchmarks: ["OpenWebText unconditional generation", "Sudoku conditional and unconditional generation"]
---

# 论文速读：Simplex Relaxation for Discrete Diffusion

## 一句话总结
本文提出 Simplax，一种针对均匀离散扩散（uniform discrete diffusion）的 Dirichlet–categorical 精确增强方法，在保持原有分类污染过程不变的前提下，引入辅助单纯形变量构造 Rao–Blackwellized 反桥目标与随机祖先采样器；在 OpenWebText 无条件文本生成和约束性 Sudoku 生成上均取得最优或接近最优的生成质量与有效性。

## 研究问题与动机
- 均匀离散扩散通过将所有类别对称地替换到均匀分布来污染序列，相比掩码扩散无需引入特殊吸收状态；但现有工作的训练目标与反向更新仍完全基于采样得到的分类中间状态，表达形式受限。
- 若仅在原有分类扩散框架内“ richer"地构造目标与采样，而不改变向前污染过程本身，是否能获得更可追踪的训练与采样形式，并提升生成效果。
- 直接在增广的单纯形空间匹配反向桥（reverse bridge）面临困难： induced simplex reverse bridges 为移位 Dirichlet 分量混合，其 KL 散度通常不可追踪。
- 需要在不破坏原分类边缘分布的前提下，引入辅助概率结构以支撑 tractable objective 与对应的 stochastic reverse sampler。

## 核心贡献（创新点）
- 提出精确 Dirichlet–categorical 增强，将每个被污染的分类状态 $\mathbf{z}_t$ 与辅助单纯形变量 $\mathbf{w}_t$ 通过移位 Dirichlet 条件耦合，并以原均匀扩散过程为分类边缘；区别于多数将连续/松弛状态作为主要生成态的做法，本文的单纯形变量仅作为精确辅助桥变量。
- 推导可追踪的 categorical reverse-bridge 代理目标，通过对辅助分类解码 $\widetilde{\mathbf{z}}_t \sim \mathrm{Cat}(\mathbf{w}_t)$ 求期望并给出 Rao–Blackwellized 闭式（式 15），消除辅助解码带来的采样噪声；这与直接匹配不可追踪的 Dirichlet 混合反向桥形成本质区别。
- 基于同一增广层次导出 stochastic ancestral sampler，使用 denoiser 预测与辅助单纯形状态共同参数化反向更新；该采样器保持 $\mathbf{z}_t$ 作为网络输入，避免对完整词表做逐位置的稠密矩阵乘。
- 给出该目标的连续时间极限并证明非退化性，展示本文目标可视为 UDLM 连续时间目标的 simplex-relaxed 类比。

## 方法详解
- 构建包含 $\mathbf{x}, \mathbf{z}_s, \mathbf{z}_t, \mathbf{w}_s, \mathbf{w}_t$ 的联合分解：保留标准均匀扩散的 forward transition $q(\mathbf{z}_t|\mathbf{z}_s)$，并为每个时刻 $t$ 引入 $q(\mathbf{w}_t|\mathbf{z}_t, \mathbf{x}) = \mathrm{Dir}(\mathbf{w}_t; \eta_t \mathbf{p}_t + \mathbf{z}_t)$，其中 $\mathbf{p}_t(\mathbf{x}) = \alpha_t \mathbf{x} + (1-\alpha_t)\boldsymbol{\pi}$。
- 利用该层次可得精确边缘 $q(\mathbf{w}_t|\mathbf{x}) = \mathrm{Dir}(\mathbf{w}_t; \eta_t \mathbf{p}_t)$、精确解码 $q(\mathbf{z}_t|\mathbf{w}_t) = \mathrm{Cat}(\mathbf{z}_t; \mathbf{w}_t)$，以及以 $\mathbf{w}_t$ 为条件的提升后反桥 $q(\mathbf{z}_s|\mathbf{w}_t, \mathbf{x}) = \mathrm{Cat}(\mathbf{z}_s; \boldsymbol{\rho}_{s|t}(\mathbf{x}, \mathbf{w}_t))$。
- 训练时采样 $(\mathbf{w}_t, \mathbf{z}_t)$ 并使 denoiser 以 $\mathbf{z}_t$ 为输入预测 $\hat{\mathbf{x}}_\theta = f_\theta(\mathbf{z}_t, t)$；定义辅助解码 $\widetilde{\mathbf{z}}_t \sim \mathrm{Cat}(\mathbf{w}_t)$ 后优化 $\bar{\mathcal{L}}_{z_s|z_t,w_t} = \mathbb{E}_{q(\widetilde{\mathbf{z}}_t|\mathbf{w}_t)}[D_{\mathrm{KL}}(q(\mathbf{z}_s|\widetilde{\mathbf{z}}_t,\mathbf{x}) \| q(\mathbf{z}_s|\widetilde{\mathbf{z}}_t, \hat{\mathbf{x}}_\theta))]$。
- 通过边际化得到闭式损失 $\bar{\mathcal{L}}_{z_s|z_t,w_t} = \langle \mathbf{w}_t, \log \hat{\mathbf{p}}_t - \log \mathbf{p}_t \rangle + \langle \boldsymbol{\rho}_{s|t}(\mathbf{x}, \mathbf{w}_t), \log \mathbf{p}_s - \log \hat{\mathbf{p}}_s \rangle$，并进一步给出连续时间密度 $\ell_{\mathrm{ct}}$ 与积分目标 $\mathcal{L}_{\mathrm{ct}}$。
- 采样时从 $q(\mathbf{w}_{t_N}) = \mathrm{Dir}(\eta_{t_N}\boldsymbol{\pi})$ 出发，经 $q(\mathbf{z}_{t_N}|\mathbf{w}_{t_N})$ 得到初始分类状态，并在每步按 $\mathbf{z}_s \sim \mathrm{Cat}(\boldsymbol{\rho}_{s|t}(\hat{\mathbf{x}}_\theta, \mathbf{w}_t))$ 与 $\mathbf{w}_s \sim \mathrm{Dir}(\eta_s \hat{\mathbf{p}}_s + \mathbf{z}_s)$ 更新，保持 $\mathbf{z}_t$ 作为下一步的网络输入以避免额外稠密词表投影。

## 实验与结果
- OpenWebText 无条件生成使用 GPT-2 BPE tokenizer（词表 50,257）、序列长度 1,024 与 179M 参数 diffusion transformer；Simplax 主要实验使用 $\eta_t \equiv 0.01$ 并以 $\mathbf{z}_t$ 作 denoiser 输入。
- 在 OpenWebText 上对比 CANDI、UDLM、MDLM、Duo、FLM、LangFlow、S-FLM，按生成单字熵最接近数据熵 5.44 nats 的点选取操作点：Simplax 在 NFE=16 和 NFE=1,024 下于三种 LLM 评估器上获得最低 Gen. PPL；NFE=128 时在 GPT-2 Large/XL 上最优，LangFlow 在 Llama-2 7B 上最优。示例显示 NFE=16 时 GPT-2 L 为 90.5、GPT-2 XL 为 93.1、Llama-2 7B 为 49.3，NFE=1,024 时分别为 45.1、46.8、25.5。
- Sudoku 实验所有模型仅用 30-clue 数据训练，并在 40/35/30/25/20/17-clue 分布迁移以及 0-clue 无条件生成上评估。Simplax 在所有设置中取得最高性能：30-clue 条件准确率为 61.75%，17-clue 为 1.20%，无条件有效性达 95.85%，强于最强基线 Duo 的 80.95%。
- 消融显示使用 $\mathbf{z}_t$ 作为 denoiser 输入优于直接使用 $\mathbf{w}_t$；从 UDLM 预训练 800k 步再接续 200k 步 Simplax 目标可进一步改善 Gen. PPL–Gen. ENT 前沿。

## 相关工作脉络
- 均匀离散扩散与 UDLM：本文保留标准分类前向过程与 reverse posterior，但通过精确 Dirichlet–categorical 增广与辅助解码期望构造更灵活的连续时间目标，而非直接从分类噪声态推导 UDLM 型目标。
- 基于辅助变量的离散扩散变体：Di4C、VADD、CoDD 等通过混合或 latent 丰富反向分布；CANDI、Duo/FLM 等采用 hybrid/连续松弛；本文与它们的区别在于单纯形变量不作为主要生成态或 denoiser 输入，而是保持 $\mathbf{z}_t$ 输入、并在目标与采样层面进行精确辅助。
- 单纯形上的扩散与流建模：如 softmax 变换的 simplex diffusion、categorical SDE+CIROS、DDSM 与 Dirichlet flow matching 等方法把单纯形状态视为一等公民；本文则把这些几何工具用于标准离散扩散的精确增广，角色定位为辅助桥变量而非主生成过程。
- 连续时间离散扩散：Campbell 等人建立连续时间框架，Lou 等人通过数据分布比率建模；本文在连续时间极限下得到非退化目标，并可视为 UDLM 在 simplex 增广下的对应形式。
- 近期 scaling 与改进工作：Sahoo 等人的 scaling 工作关注 guided/self-correcting/few-step；本文的贡献不在采样步数压缩，而在提升训练目标的追踪性与抽样结构的严密性。

## 局限性与未来方向
- 当前 formulation 专门针对 uniform categorical corruption，未覆盖更一般的污染核。
- 引入的辅助单纯形状态的额外计算开销未进行全面量化评估。
- 浓度调度 $\eta_t$ 仍为设计选择而非由理论唯一确定。
- 未来方向包括推广到更广泛的分类污染核、开发更高效的反向求解器，并更系统地分析增广带来的计算成本与训练稳定性。

## 研究启发与可借鉴点
- 将“精确增广+边缘一致性”的思路迁移到其他离散扩散设定（如 absorbing/masked）中，可能同样带来可追踪的辅助目标。
- Rao–Blackwellized 化处理辅助解码期望的做法可复用于需要引入 latent categorical decode 的训练目标，从而消除蒙特卡洛噪声并保持闭式梯度。
- 实验上将 denoiser 输入与优化目标的支撑变量解耦（保留整数 token lookup 优势）是值得借鉴的工程实践，可在多种连续/松弛框架中复用。
- 在低提示密度（如 17-clue Sudoku）上的显著增益提示该方法在强约束下的全局一致性建模能力较强，适合探索约束序列生成与逻辑结构学习任务。
- 连续时间极限的推导流程可迁移到需要比较不同离散扩散目标在 infinitesimal 层面行为的研究中。

## 关键术语表
**Uniform discrete diffusion**：将每个 token 以对称方式替换到完整词表的均匀分布，不使用特殊吸收/mask 状态的离散扩散设定。
**Simplax**：本文提出的 Dirichlet–categorical 精确增广方法，保持原分类前向过程并为每个噪声时刻引入辅助单纯形变量。
**Rao–Blackwellized objective**：通过对辅助解码分布求期望并以解析方式边际化，消除采样噪声的可追踪训练目标形式。
**Reverse bridge**：在两个扩散时刻之间刻画从噪声状态回到更早状态的条件分布，本文同时给出分类与单纯形层面的反桥表达式。
**Stochastic ancestral sampler**：基于增广层次依次采样 $\mathbf{z}_s$ 与 $\mathbf{w}_s$ 的反向生成过程，保持分类输入友好性。
**Concentration parameter $\eta_t$**：控制 Dirichlet 辅助分布集中程度的超参，实验中采用常数 $\eta_t \equiv 0.01$。
**Generative perplexity–entropy tradeoff**：在文本生成中衡量输出分布贴近数据熵同时降低外部 LLM 评估困惑度的综合指标。

## 可复现要素
- OpenWebText：公开数据集，使用 GPT-2 BPE tokenizer，词表 50,257，序列长度 1,024； backbone 为 179M 参数 diffusion transformer，Adam(lr=$3\times10^{-4}$)，batch=512，1M 步；简化说明中报告使用 $\eta=0.01$，并从 UDLM 800k 步 checkpoint 接续 200k 步训练。
- Sudoku：基于 Deschenaux & Gulcehre 基准，48,000 训练谜题（30 clues）、每密度 2,000 验证集；Transformer(8 blocks, hidden 512, 8 heads, dropout 0.1)，20,000 步，Adam(lr=$3\times10^{-4}$)，batch=256。
- 代码/权重开源情况论文未明确声明；关键超参与架构细节在附录中有进一步说明。
