---
title: "Simplex Relaxation for Discrete Diffusion"
source: https://arxiv.org/pdf/2608.10615v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:59:48"
field: "离散扩散生成模型"
keywords: ["discrete diffusion", "uniform diffusion", "simplex relaxation", "Dirichlet-categorical augmentation", "Rao-Blackwellization", "OpenWebText", "Sudoku"]
innovations: ["Exact Dirichlet-categorical augmentation preserving the original uniform categorical corruption as a marginal", "Rao-Blackwellized closed-form reverse-bridge training objective with non-degenerate continuous-time limit", "Stochastic ancestral sampler using z_t-input while leveraging w_t in objective and reverse update"]
benchmarks: ["OpenWebText unconditional generation", "Sudoku conditional/unconditional generation across 40/35/30/25/20/17/0 clues"]
---

# 论文速读：Simplex Relaxation for Discrete Diffusion

## 一句话总结
本文提出 **Simplax**，一种对均匀离散扩散的精确 Dirichlet–categorical 增强方法：在不改变原始分类损坏过程的前提下，引入辅助单纯形变量以构建可处理的 Rao–Blackwellized 反向桥训练目标与随机祖先采样器；在 OpenWebText 无条件文本生成和 数独约束分类生成上均取得最佳性能。

## 研究问题与动机
- 离散扩散模型（面向文本、生物序列等分类数据）的中间状态空间与反向预测问题均由损坏核（corruption kernel）决定；当前主流设计（掩码扩散、均匀扩散）分别引入了特殊吸收状态或对称地替换为均匀分布，但两者的训练目标与反向转换仍受限于分类中间状态本身。
- 均匀离散扩散（uniform discrete diffusion）的当前范式存在方法论空白：能否在**不改变底层分类损坏过程**的前提下，丰富其训练目标与反向采样器？
- 标准均匀扩散中反向更新直接通过采样分类状态表达，虽保持离散生成过程，却限制了对更 tractable 目标与采样器的构造空间。
- 作者希望引入一种围绕分类过渡的辅助概率结构，从而在不改变前向过程的基础上，获得理论上可处理的目标与采样器。

## 核心贡献（创新点）
1. **提出精确的 Dirichlet–categorical 增强**：将每个损坏的分类状态 $\mathbf{z}_t$ 与辅助单纯形变量 $\mathbf{w}_t$ 耦合（$q(\mathbf{w}_t|\mathbf{z}_t,\mathbf{x}) = \mathrm{Dir}(\cdot; \eta_t \mathbf{p}_t + \mathbf{z}_t)$），使原始均匀扩散过程成为其分类边缘分布。与已有工作（如 Duo、FLM 等将辅助变量作为主生成态）的本质区别在于：本文的单纯形变量仅为辅助桥接量，网络输入仍为分类状态 $\mathbf{z}_t$。
2. **推导可处理的 Rao–Blackwellized 反向桥目标**：通过将标准离散反向 KL 对辅助解码 $\widetilde{\mathbf{z}}_t \sim \mathrm{Cat}(\mathbf{w}_t)$ 取期望后精确边缘化，得到闭式损失（式 15），消除了蒙特卡洛采样噪声。与直接匹配 Dirichlet 混合桥（式 13，不可行）的本质区别在于：前者保持与标准离散扩散目标相近的形式，且拥有非退化连续时间极限。
3. **给出从同一增强层次导出的随机祖先采样器**：每一步同时采样 $\mathbf{z}_s$ 与 $\mathbf{w}_s$，$\mathbf{z}_s$ 作为下一轮去噪器输入，$\mathbf{w}_s$ 携带桥信息。与将单纯形状态直接送入去噪器（$w_t$-input）相比，$z_t$-input 既保留辅助变量的训练收益，又避免每位置的一次词汇表尺度密集矩阵乘法。
4. **系统实验验证**：在 OpenWebText 全 NFE 预算范围（16/128/1024）与数独多线索密度（40/35/30/25/20/17/0）下，Simplax 均达到最佳或接近最佳，尤其 17 线索（最小唯一解 regime）与无条件生成（有效性 95.85%）显著领先。

## 方法详解
- **辅助单纯形变量定义**：对 $t \in (0,1]$，令 $\mathbf{w}_t \in \Delta^{K-1}$ 满足
  $$q(\mathbf{w}_t | \mathbf{z}_t, \mathbf{x}) = \mathrm{Dir}(\mathbf{w}_t; \eta_t \mathbf{p}_t + \mathbf{z}_t),$$
  其中 $\mathbf{p}_t(\mathbf{x}) = \alpha_t \mathbf{x} + (1-\alpha_t)\pi$ 为扩散时间边缘分布，$\eta_t > 0$ 为浓度参数，加性 one-hot $\mathbf{z}_t$ 将松弛态锚定到采样到的离散 token。
- **精确层次结构**（Proposition 1）：
  1) $\mathbf{w}_t$ 的边缘分布为 $\mathrm{Dir}(\eta_t \mathbf{p}_t)$；2) 给定 $\mathbf{w}_t$，$\mathbf{x} \perp \mathbf{z}_t$，且 $q(\mathbf{z}_t|\mathbf{w}_t) = \mathrm{Cat}(\mathbf{w}_t)$（精确解码器）；3) 对 $s<t$，$q(\mathbf{z}_s|\mathbf{w}_t, \mathbf{x}) = \mathrm{Cat}(\rho_{s|t}(\mathbf{x}, \mathbf{w}_t))$，其中 $\rho_{s|t}$ 由式 (11) 给出；4) $q(\mathbf{w}_s|\mathbf{w}_t, \mathbf{x})$ 为 shifted Dirichlet 混合。
- **Rao–Blackwellized 训练目标**：在训练时联合采样 $(\mathbf{w}_t, \mathbf{z}_t)$，并以 $\widehat{\mathbf{x}}_\theta = f_\theta(\mathbf{z}_t, t)$ 预测干净分布；定义辅助解码 $\widetilde{\mathbf{z}}_t \sim \mathrm{Cat}(\mathbf{w}_t)$，优化
  $$\bar{\mathcal{L}}_{z_s|z_t,w_t}(\mathbf{w}_t, \mathbf{z}_t, \mathbf{x}; s,t) = \mathbb{E}_{q(\widetilde{\mathbf{z}}_t|\mathbf{w}_t)}\!\left[ D_{\mathrm{KL}}\!\big(q(\mathbf{z}_s|\widetilde{\mathbf{z}}_t,\mathbf{x}) \,\|\, q(\mathbf{z}_s|\widetilde{\mathbf{z}}_t, \widehat{\mathbf{x}}_\theta)\big) \right].$$
  该期望可精确边缘化为闭式（Proposition 2，式 15）：
  $$\bar{\mathcal{L}} = \langle \mathbf{w}_t, \log \hat{\mathbf{p}}_t - \log \mathbf{p}_t \rangle + \langle \rho_{s|t}(\mathbf{x},\mathbf{w}_t), \log \mathbf{p}_s - \log \hat{\mathbf{p}}_s \rangle.$$
- **连续时间极限**：当 $s = t - \Delta, \Delta \downarrow 0$ 时，式 (15) 具非退化一阶极限（Proposition 3，式 17–18），其被识别为 UDLM 目标的单纯形松弛连续时间类比。
- **随机祖先采样器**（式 19–21）：从 $\mathbf{w}_{t_N} \sim \mathrm{Dir}(\eta_{t_N}\pi), \mathbf{z}_{t_N} \sim \mathrm{Cat}(\mathbf{w}_{t_N})$ 出发，每步去噪器预测 $\widehat{\mathbf{x}}_\theta$，再采样
  $$\mathbf{z}_s \sim \mathrm{Cat}(\rho_{s|t}(\widehat{\mathbf{x}}_\theta, \mathbf{w}_t)), \quad \mathbf{w}_s \sim \mathrm{Dir}(\eta_s \hat{\mathbf{p}}_s + \mathbf{z}_s).$$
- **设计诊断结论**：相比 $w_t$-input（每位置需 $\mathbf{w}_t^\top E$ 密集乘积），$z_t$-input 在相同目标下获得更优 Gen. PPL–Gen. ENT 前沿且更省计算；UDLM 预热初始化（800k + 200k 迭代）可进一步提升前景。

## 实验与结果
- **OpenWebText**（无条件文本生成）：179M 参数 diffusion transformer，GPT-2 BPE（$|\mathcal{V}|=50257$），序列长 1024，Adam（lr=$3\times10^{-4}$），batch=512，1M 迭代。基准对比：CANDI、UDLM、MDLM、Duo、FLM、LangFlow、S-FLM。评估指标为生成 unigram 熵（数据基准 5.44 nats）与 GPT-2 Large/XL、Llama-2 7B 评估的 perplexity。
  - NFE=16：Simplax 在三项评测器下均取得最低 Gen. PPL（GPT-2 L: 90.5 / XL: 93.1 / Llama-2: 49.3）。
  - NFE=128：GPT-2 L/XL 下最优（56.9 / 58.9）；Llama-2 7B 次优（LangFlow 30.0 vs Simplax 31.4）。
  - NFE=1024：三项评测均最优（45.1 / 46.8 / 25.5）。
  - UDLM 预训练 + Simplax 微调可进一步提升 Gen. PPL–Gen. ENT 前沿。
- **Sudoku**（约束分类生成）：8 层 Transformer（hidden=512），25–29M 参数。训练集 48k 个 30-clue 数独；测试在 40/35/30/25/20/17/0 clues 上进行。
  - 条件求解准确率：Simplax 在所有线索密度下均为最高，尤其 17-clue regime 达 **1.20%**（次优 Duo 仅 0.40%）。
  - 无条件生成有效性：**95.85%**，显著优于最强基线 Duo 的 80.95%（提升约 15 pp）。

## 相关工作脉络
- **D3PM / 均匀离散扩散主线**（Hoogeboom 2021; Austin 2021; Campbell 2022; Lou 2024; Zhang 2025）：建立统一框架与标准变分训练配方；本文沿用其分类前向过程与反向后验形式，但不改变前向核本身。
- **UDLM**（Schiff 2025）：直接从分类损坏状态推导连续时间反向 KL 目标；本文的式 (18) 可视为 UDLM 目标的 simplex-relaxed 版本，关键区别在于引入 $\mathbf{w}_t$ 辅助平均使目标保持 tractable 并具备相同连续时间极限。
- **Di4C / VADD / CoDD**（Hayakawa 2025; Xie 2026; Li 2026）：通过混合乘积模型、高斯隐变量或概率电路丰富反向分布；本文同样引入辅助变量，但角色定位为**精确桥接量**而非主生成态，且保留 $z_t$-input。
- **Duo / Duo++ / FLM**（Sahoo 2025; Lee 2026; Deschenaux 2026）：分别在连续松弛与欧氏空间构造去噪器；本文与它们的核心差异是：单纯形变量只用于构建目标与采样器，主网络仍在分类空间操作。
- **CANDI / CADD**（Pynadath 2025; Zheng 2026）：离散–连续混合扩散；本文不引入连续主状态，仅在离散扩散旁构建精确 Dirichlet 层次。
- **单纯形扩散 / Dirichlet 方法**（Floto 2023; Richemond 2023; Avdeyev 2023; Stark 2024; Chandra 2026）：将扩散过程本身定义在单纯形上；本文与它们几何相近，但 $\mathbf{w}_t$ 不充当第一类生成状态，而是附加于标准离散扩散的精确辅助桥。

## 局限性与未来方向
- 当前构造**仅适用于均匀分类损坏**，未覆盖更一般核（如吸收型/结构化核）。
- 引入辅助单纯形状态的**额外计算开销**未做系统刻画；$z_t$-input 虽规避了输入端的密集投影，但目标与采样仍涉及 $\mathbf{w}_t$ 相关运算。
- 浓度调度 $\eta_t$ 仍是**经验设计选择**，理论未给出最优 schedule。
- 未来方向：扩展到更广泛的分类损坏核、开发更高效反向求解器、分析开销–收益权衡。

## 研究启发与可借鉴点
- **辅助变量的"桥接而非替代"设计**：将 $\mathbf{w}_t$ 仅用于训练目标与采样，而网络输入保持为分类状态 $z_t$，既获得松弛结构的训练信号，又避免额外密集投影；该思想可迁移至其他离散扩散变体。
- **Rao–Blackwellized 边缘化技巧**：通过解析平均掉辅助解码 $\widetilde{\mathbf{z}}_t$ 消除采样噪声并获得闭式损失，为其他含混合/辅助结构的离散扩散目标设计提供范式。
- **UDLM 预训练 + 新目标微调**：实验显示 800k UDLM 预热 + 200k Simplax 目标微调可进一步提升前景；这种"共享前向 + 不同反向目标"的迁移策略在扩散语言模型中值得系统化研究。
- **在极稀疏条件 regime 的有效泛化**：数独 17-clue（理论最小唯一解）上的相对提升显著，提示该框架对强约束、低信息场景有潜力；可与 constraint satisfaction、combinatorial generation 方向结合。

## 关键术语表
**Simplax**：本文提出的方法名，指代对均匀离散扩散施加的 Dirichlet–categorical 精确增强。
**单纯形松弛（Simplex relaxation）**：引入辅助单纯形变量 $\mathbf{w}_t$ 以松弛/扩展离散扩散的反向桥结构，而保持分类前向不变。
**Rao–Blackwellized 反向桥目标**：通过对辅助解码 $\widetilde{\mathbf{z}}_t$ 解析边缘化得到的闭式训练损失，消除了蒙特卡洛方差。
**反向桥（Reverse bridge）**：给定 $t$ 时刻状态与干净数据，刻画 $s<s't$ 时刻后验分布的过渡条件；本文分别在 $\mathbf{z}$ 与 $\mathbf{w}$ 空间定义。
**均匀扩散（Uniform diffusion）**：将 token 按对称方式替换为均匀分布（无特殊 mask 吸收态）的损坏核。
**浓度参数 $\eta_t$**：控制 $\mathrm{Dir}(\eta_t \mathbf{p}_t)$ 围绕均值 $\mathbf{p}_t$ 集中程度的超参；本文固定为 0.01。
**NFE（Number of Function Evaluations）**：反向采样步数；越小代表推理越快，NFE=16/128/1024 覆盖低到高预算。
**Gen. PPL / Gen. ENT**：生成困惑度与生成 unigram 熵；前者越低越好，后者用于对齐数据分布（OpenWebText 基准 5.44 nats）。

## 可复现要素
- **数据集**：OpenWebText（公开）、Sudoku（基于 Deschenaux & Gulcehre 2026 与 Alp 2024 生成器；种子 42；训练 48k、评估各密度 2k）——**公开**。
- **代码/权重**：论文未明确声明代码与权重开源。
- **关键超参**：OpenWebText：179M 参数 diffusion transformer、1024 序列长、lr=$3\times10^{-4}$、batch=512、1M 迭代；Simplax 固定 $\eta=0.01$，$z_t$-input，不用自条件。Sudoku：8 层 Transformer、hidden=512、8 heads、dropout=0.1、lr=$3\times10^{-4}$、batch=256、20k 步、EMA decay=0.9999、antithetic time sampling 开启、seed=1。
