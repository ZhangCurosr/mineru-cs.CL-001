---
title: "Simplex Relaxation for Discrete Diffusion"
source: https://arxiv.org/pdf/2608.10615v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:00:21"
field: "离散扩散模型"
keywords: ["discrete diffusion", "Dirichlet-categorical augmentation", "Rao-Blackwellized objective", "simplex relaxation", "OpenWebText", "Sudoku generation"]
innovations: ["精确 Dirichlet-categorical 增强保持原始类别扩散为边际", "Rao-Blackwellized 反向桥损失解析闭合形式", "基于增强层次的随机祖先采样器"]
benchmarks: ["OpenWebText", "Sudoku"]
---

# 论文速读：Simplex Relaxation for Discrete Diffusion

## 一句话总结
本文提出 Simplax，一种针对均匀离散扩散的精确 Dirichlet-categorical 增强方法，通过引入辅助单形变量 $\mathbf{w}_t$ 构建可解的 Rao-Blackwellized 反向桥目标与随机祖先采样器，在 OpenWebText 文本生成与 Sudoku 约束生成任务上均显著优于基线。

## 研究问题与动机
- 现有离散扩散模型（如 UDLM、MDLM 等）的反向更新直接基于采样的类别状态，训练目标与采样过程均受限于离散中间状态，缺乏利用连续辅助结构优化训练目标的空间。
- 均匀扩散保留了原始类别腐蚀过程，但未充分利用扩散过程中的概率结构；能否在**不改变向前腐蚀过程**的前提下，通过引入辅助变量丰富训练目标与采样策略，是一个开放的方法论问题。
- 标准离散扩散中，直接匹配松弛空间中的单形反向桥 $D_{\mathrm{KL}}[q(\mathbf{w}_s|\mathbf{w}_t,\mathbf{x}) \| q(\mathbf{w}_s|\mathbf{w}_t,\hat{\mathbf{x}}_\theta)]$ 会导致 Dirichlet 混合分布的 KL 散度难以计算，阻碍了目标的可求解性。
- 现有辅助变量方法（如 VADD、Duo 等）通常将辅助变量作为去噪器的直接输入，而本文希望保留类别状态 $\mathbf{z}_t$ 作为网络输入，仅将其用于构建更丰富的训练目标与采样过程。

## 核心贡献（创新点）
1. **引入精确的 Dirichlet-categorical 增强层次结构**：在均匀离散扩散中，将每个腐蚀后的类别状态 $\mathbf{z}_t$ 与辅助单形变量 $\mathbf{w}_t$ 通过移位 Dirichlet 条件耦合，使原始均匀扩散过程成为其类别边际，同时提供从 $\mathbf{w}_t$ 精确解码回 $\mathbf{z}_t$ 的 Categorical 关系。
2. **推导可解的 Rao-Blackwellized 反向桥损失**：通过将标准离散反向 KL 散度对从 $\mathbf{w}_t$ 中额外采样的辅助类别 $\widetilde{\mathbf{z}}_t$ 求期望，并给出解析闭合形式（式 15），消除了辅助解码采样的蒙特卡洛噪声，同时保留 $\mathbf{z}_t$ 作为去噪器输入。
3. **设计基于增强层次的随机祖先采样器**：利用推导出的反向后验 $\rho_{s|t}(\hat{\mathbf{x}}_\theta, \mathbf{w}_t)$ 与 Dirichlet 转移构造多步反向采样流程，在保持计算效率（使用 token index lookup）的同时提升生成质量。
4. **证明连续时间极限存在性**：展示离散目标在非退化的一阶连续时间极限下收敛到式 (17) 的局部密度，并与 UDLM 的连续时间目标建立关联。

## 方法详解
**增强结构**：对于时间 $s < t$，引入联合因式分解：
$$q(\mathbf{x}, \mathbf{z}_s, \mathbf{z}_t, \mathbf{w}_s, \mathbf{w}_t) = q(\mathbf{x}) q(\mathbf{z}_s|\mathbf{x}) q(\mathbf{w}_s|\mathbf{z}_s, \mathbf{x}) q(\mathbf{z}_t|\mathbf{z}_s) q(\mathbf{w}_t|\mathbf{z}_t, \mathbf{x})$$
其中辅助单形变量定义为：
$$q(\mathbf{w}_t|\mathbf{z}_t, \mathbf{x}) = \mathrm{Dir}(\mathbf{w}_t; \eta_t \mathbf{p}_t + \mathbf{z}_t)$$
这里 $\mathbf{p}_t(\mathbf{x}) = \alpha_t \mathbf{x} + (1-\alpha_t)\boldsymbol{\pi}$ 是扩散时刻的类别边际分布，$\eta_t > 0$ 是浓度参数。

**关键性质**：
- 边际分布：$q(\mathbf{w}_t|\mathbf{x}) = \mathrm{Dir}(\mathbf{w}_t; \eta_t \mathbf{p}_t)$
- 精确解码：$q(\mathbf{z}_t|\mathbf{w}_t) = \mathrm{Cat}(\mathbf{z}_t; \mathbf{w}_t)$
- 提升后的反向后验（Categorical）：
$$\rho_{s|t}(\mathbf{x}, \mathbf{w}_t) = \mathbf{p}_s \odot \left[\alpha_{t|s}(\mathbf{w}_t \oslash \mathbf{p}_t) + (1-\alpha_{t|s})\langle \mathbf{w}_t, \boldsymbol{\pi} \oslash \mathbf{p}_t\rangle \mathbf{1}\right]$$

**训练目标（Rao-Blackwellized 形式）**：
$$\bar{\mathcal{L}}_{z_s|z_t, w_t} = \langle \mathbf{w}_t, \log \hat{\mathbf{p}}_t - \log \mathbf{p}_t \rangle + \langle \rho_{s|t}(\mathbf{x}, \mathbf{w}_t), \log \mathbf{p}_s - \log \hat{\mathbf{p}}_s \rangle$$
其中 $\hat{\mathbf{x}}_\theta = f_\theta(\mathbf{z}_t, t)$，$\mathbf{z}_t$ 独立于辅助解码 $\widetilde{\mathbf{z}}_t \sim \mathrm{Cat}(\mathbf{w}_t)$。

**采样过程**：
从 $\mathbf{w}_{t_N} \sim \mathrm{Dir}(\eta_{t_N}\boldsymbol{\pi})$ 开始，迭代执行：
$$\mathbf{z}_s \sim \mathrm{Cat}(\rho_{s|t}(\hat{\mathbf{x}}_\theta, \mathbf{w}_t)), \quad \mathbf{w}_s \sim \mathrm{Dir}(\eta_s \hat{\mathbf{p}}_s + \mathbf{z}_s)$$

## 实验与结果
**OpenWebText 无条件文本生成**：
- 使用 179M 参数的扩散 Transformer，GPT-2 BPE tokenizer（50,257 词表），序列长度 1,024，训练 1M 步。
- Simplax 在 NFE=16 和 NFE=1,024 时，在所有评估器（GPT-2 Large、GPT-2 XL、Llama-2 7B）下均获得最低生成困惑度；NFE=128 时在 GPT-2 Large/XL 下最优。
- 最佳结果（Llama-2 7B，NFE=1,024）：Simplax 达到 25.5，显著优于 UDLM（32.2）、MDLM（33.9）、LangFlow（28.2）。

**Sudoku 约束分类生成**：
- 所有模型仅在 30-clue 数据集上训练，评估覆盖 40→17 clues 及无条件（0 clue）场景。
- Simplax 在所有条件下均达到最高性能：
  - 40 clues: 98.55%（最高）
  - 30 clues（训练分布）: 61.75%
  - 17 clues（最稀疏）: 1.20%
  - 无条件生成有效性: **95.85%**，比最强基线 Duo（80.95%）提升约 15 个百分点。

## 相关工作脉络
- **UDLM**（Schiff et al., 2025）：直接基于类别状态推导连续时间反向 KL 目标；Simplax 引入辅助单形变量并在其基础上平均标准离散反向 KL，可视为 UDLM 目标的"单形松弛"连续时间类比。
- **Duo/Duo++**（Sahoo et al., 2025, Deschenaux et al., 2026）：引入高斯潜变量到掩码去噪中；Simplax 采用单形变量而非高斯变量，且保留类别状态作为去噪器输入。
- **FLM**（Lee et al., 2026）：在 one-hot 状态上进行欧几里得去噪；Simplax 不改变去噪器输入类型。
- **CANDI/CADD**（Pynadath et al., 2025, Zheng et al., 2026）：混合离散-连续扩散；Simplax 保持完整离散前向过程，辅助变量仅用于目标构建。
- **Di4C/VADD**：使用产品混合或高斯潜变量捕捉维度相关性；Simplax 的增强更结构化和精确。
- **Dirichlet Flow Matching / DDSM**：直接在单形上定义生成过程；Simplax 将单形变量定位为精确辅助桥而非主生成状态。

## 局限性与未来方向
- 当前形式专为均匀类别腐蚀设计，计算开销相对于标准离散扩散尚未充分量化。
- 浓度调度 $\eta_t$ 是额外设计选择，而非由理论决定。
- 未来方向包括：扩展到更广泛的类别腐蚀核、开发更高效的反向求解器。

## 研究启发与可借鉴点
1. **辅助变量的"精确边际保留"设计**：通过移位 Dirichlet 条件使离散扩散保持为增强层次结构的边际，这种"不改变前向过程"的增强思路可迁移到其他离散生成框架（如掩码扩散）。
2. **Rao-Blackwellized 损失构造**：通过解析消除辅助采样噪声，同时保持网络输入为离散状态——这一技巧可在需要丰富训练信号但不愿改变推理接口的设计中复用。
3. **单形松弛与连续时间极限的关联**：证明离散目标具有一阶非退化连续时间极限，为离散扩散的连续化分析提供了新视角，可启发其他离散模型的连续极限研究。
4. **计算效率权衡**：保留 $\mathbf{z}_t$ 作为去噪器输入避免额外稠密矩阵乘法，提示辅助变量方法的工程实现需仔细评估推理开销。

## 关键术语表
- **Simplex Relaxation（单形松弛）**：将离散状态耦合到连续单形变量上的精确概率增强，保持原扩散过程不变。
- **Dirichlet-categorical Hierarchy（Dirichlet-categorical 层次）**：通过移位 Dirichlet 条件建立的辅助变量与类别状态的精确联合分布结构。
- **Rao-Blackwellized Objective（Rao-Blackwellized 目标）**：对辅助采样期望进行解析边际化后得到的无偏、低方差训练损失闭合形式。
- **Reverse Bridge（反向桥）**：连接两个扩散时间点的条件后验分布，此处特指提升后经过 $\mathbf{w}_t$ 条件化的反向分布。
- **Stochastic Ancestral Sampler（随机祖先采样器）**：基于增强层次结构从终态逐步采样的反向生成过程。
- **Concentration Parameter（浓度参数）**：Dirichlet 分布的形状参数 $\eta_t$，控制单形变量的集中程度。

## 可复现要素
- **OpenWebText**：公开数据集，使用 GPT-2 BPE tokenizer（50,257 词表），序列长度 1,024；179M 参数扩散 Transformer，Adam 优化器，学习率 $3\times10^{-4}$，batch size 512，训练 1M 步；Simplax 使用常数浓度 $\eta_t \equiv 0.01$；模型从 UDLM checkpoint（800k 步）微调 200k 步。
- **Sudoku**：公开基准（48,000 训练样本），Transformer 8 层，hidden dim 512，8 个注意力头；训练 20,000 步，学习率 $3\times10^{-4}$，batch size 256。
- **代码开源情况**：论文未明确声明代码开源链接。
- **关键超参**：$\eta_t = 0.01$（OpenWebText），时间调度均匀，EMA decay 0.9999。
