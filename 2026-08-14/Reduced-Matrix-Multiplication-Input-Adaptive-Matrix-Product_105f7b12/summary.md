---
title: "Reduced-Matrix-Multiplication-Input-Adaptive-Matrix-Product"
source: https://arxiv.org/pdf/2608.13426v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:49:03"
field: "大语言模型高效推理"
keywords: ["LLM推理加速", "矩阵乘积缩减", "输入自适应计算", "注意力剪枝", "免训练优化", "Transformer冗余分析"]
innovations: ["提出训练免的输入自适应矩阵乘积缩减方法RMM，按激活范数动态选取收缩轴Top-K索引", "证明激活列范数TopK选择在信息不对称下具有极小极大最优性", "揭示Transformer中attention-side冗余度显著高于MLP的结构非对称性"]
benchmarks: ["MMLU", "GSM8K", "HUMANEVAL", "RULER", "CNN/DAILYMAIL", "POPE", "BLINK", "ARC-Challenge", "COPA", "PIQA"]
---

# 论文速读：Reduced-Matrix-Multiplication-Input-Adaptive-Matrix-Product

## 一句话总结
提出 Reduced Matrix Multiplication（RMM），一种无需训练的输入自适应推理方法，通过在 Transformer 矩阵乘法的共享收缩维度上按激活范数选取 Top-K 关键索引来减少计算量，实现可调控的精度–效率权衡，并在 1B–70B 语言模型及多模态模型上验证有效性与实际加速收益。

## 研究问题与动机
- Transformer 推理中注意力与 MLP 层包含大量高维矩阵乘法，逐 token 执行全量乘积是否存在冗余？是否可为当前输入自适应地裁剪共享收缩轴上的计算？
- 现有静态剪枝/低秩近似方法在推理前固定结构，无法根据输入动态调整；KV-cache 压缩与 token 剪枝作用于输入/缓存层面，而非矩阵乘积本身的收缩维度。
- 激活稀疏方法（如 TEAL、CATS）针对激活张量元素做稀疏化，关注投影层输入，未直接作用于 $QK^\top$、$PV$ 等注意力内部矩阵乘积的收缩轴。
- 随机近似矩阵乘法需多次采样以降低方差，不适合单次前向的 Transformer 推理场景；需要一种确定性、输入依赖的索引选择策略。

## 核心贡献（创新点）
1. **RMM 框架**：提出训练免、输入自适应的矩阵乘积缩减方法，以用户可控的保留率 $\rho$ 在每次推理中动态选取收缩轴 Top-K 索引，不修改权重。与固定剪枝的本质区别在于保留模式随输入、层、头和解码步动态变化。
2. **激活感知维度选择的最小最大最优性**：证明仅基于激活矩阵列范数的 TopK 选择在信息不对称（B 不可见）设定下是极小极大最优的，最小化最坏情况近似误差。
3. **统一覆盖注意力与 MLP 的矩阵乘积视角**：将 $QK^\top$（特征维收缩）、$PV$（token 维收缩）以及线性/MLP 投影纳入同一框架，扩展了激活稀疏方法的作用范围。
4. **结构非对称性的系统发现**：通过组件级消融揭示 attention-side 计算冗余度显著高于 MLP，MLP 中 Up/Gate/Down 投影的敏感程度差异明显，为部署提供组件级保留策略依据。
5. **端到端加速验证**：基于自定义 Triton kernel 在 NVIDIA A100 上验证理论计算节省可转化为实际运行时收益，长序列下加速更显著；并展示与 INT8 权重量化的兼容性。

## 方法详解
- **统一形式**：将 Transformer 中各类矩阵乘积统一为 $Y = AB$，其中 $A \in \mathbb{R}^{n \times d}$ 为当前输入决定的激活矩阵，$B \in \mathbb{R}^{d \times m}$ 为权重或中间表示，收缩维度为 $d$。
- **RMM 定义**：给定保留率 $\rho \in (0,1]$，选取索引集 $\mathcal{I} = \mathrm{TopK}(\{\|A_{:,j}\|_2\}_{j=1}^d, \lceil \rho d \rceil)$，计算 $\mathrm{RMM}_\rho(A,B) = A_{:,\mathcal{I}} B_{\mathcal{I},:}$。
- **注意力应用**：对单头注意力，计算特征得分 $s_j = \|Q_{:,j}\|_2$，选取 $\mathcal{I}$，得到缩减注意力分数 $\tilde{S} = \frac{1}{\sqrt{d_h}} Q_{:,\mathcal{I}} K_{:,\mathcal{I}}^\top$；可选地再对 $PV$ 做 token 维选择，计算 $a_t = \|P_{:,t}\|_2$，选取 $\mathcal{J}$，得到 $\tilde{O} = P_{:,\mathcal{J}} V_{\mathcal{J},:}$。
- **MLP/线性投影应用**：对激活 $X$ 和权重 $W$，计算 $s_j = \|X_{:,j}\|_2$，选取 $\mathcal{I}$，计算 $\tilde{Y} = X_{:,\mathcal{I}} W_{\mathcal{I},:}$，对 Up/Gate/Down 各自独立选择。
- **复杂度**：密度计算 $O(ndm)$，RMM 降至 $O(n \rho_d d m)$；$QK^\top$ 从 $O(L_q L_k d_h)$ 降至 $O(L_q L_k \rho_d d_h)$；$PV$ 从 $O(L_q L_k d_h)$ 降至 $O(L_q \rho_t L_k d_h)$。选择开销 $O(nd)$ 相对可忽略。
- **误差界**：$\|AB - A_{:,\mathcal{I}} B_{\mathcal{I},:}\|_F \leq \sum_{j \notin \mathcal{I}} \|A_{:,j}\|_2 \|B_{j,:}\|_2$；相对误差上界为 $\sqrt{\epsilon_A(\rho) \epsilon_B(\rho)}$，其中 $\epsilon_A$ 为被丢弃激活能量的比例。

## 实验与结果
- **模型与任务**：LLaMA 3.1（1.5B/8B/70B）、LLaMA 3.2（1.5B/3B）、Qwen3（7B/32B）、Qwen2.5-VL-7B-Instruct；覆盖零样本 QA（COPA、PIQA、COMMONSENSEQA、ARC-E/C）、语言建模（WIKITEXT、BOOKCORPUS）、数学/代码（GSM8K、HUMANEVAL）、长上下文（RULER-CWE/HOTPOT）、摘要（CNN/DAILYMAIL）、多模态（POPE、BLINK 系列）。
- **基线**：SparseGPT、Wanda、SliceGPT、magnitude pruning、H2O、静态 RMM、随机剪枝、TEAL。
- **QA 对比（LLaMA 3.1 8B, RR=0.5）**：RMM 平均准确率 59.8，优于 SparseGPT（56.1）、Wanda（52.7）、SliceGPT（37.0）、Magnitude（39.3）；全模型 69.8。
- **摘要（CNN/DailyMail, RR=0.5）**：RMM ROUGE-1=34.2，远优于静态（28.0）、随机（5.7）、H2O（24.4）；RR=0.8 时 RMM（37.5）几乎持平全模型（37.4）。
- **缩放趋势**：RR=0.8 时 LLaMA 3.1 70B 多数任务接近全模型；更小模型在 GSM8K/HUMANEVAL 上下降更明显；RR≤0.6 时大模型仍保持更高绝对精度。
- **长上下文（RULER, LLaMA 3.1 8B）**：RR=0.8 和 0.5 在 5K/15K/30K 长度下与基线几乎一致，未见系统性退化。
- **多模态（Qwen2.5-VL-7B）**：RR=0.8 时 RMM 在所有 BLINK/POPE 任务上与全模型几乎相同；RR=0.5 仍显著优于静态和随机剪枝。
- **额外 VLM 泛化**：LLaVA-1.5-7B、Gemma 3 12B、InternVL3-8B 在 RR=0.8 下 POPE 准确率与基线相当。
- **INT8 兼容**：LLaMA 3.1 8B INT8 加载下，attention-side RMM 在 RR=0.8 时 COPA 准确率 77.4（INT8 基线 81.4）。
- **延迟（LLaMA 3.1 8B, A100, RR=0.8）**：kernel 级 $QK^\top$ 加速 1.36×–1.56×，$AV$ 加速 1.67×–1.89×；端到端延迟在序列 4096 时加速 1.40×。
- **LLaMA 3.1 70B 延迟**：序列 4096 时密集版 OOM，RMM 以 1001.45ms 完成（2048 长度下 1.41× 加速）。

## 相关工作脉络
1. **静态剪枝（SparseGPT、Wanda、SliceGPT、Magnitude）**：在推理前固定剪枝模式作用于模型结构，本文与它们的差异在于 RMM 不修改权重，且保留模式随输入动态变化。
2. **输入/缓存级优化（H2O、token 压缩、KV-cache 管理）**：作用于 token 或 cache 层面缩短上下文，本文直接削减矩阵乘积内部的收缩维度计算，两者作用层次不同。
3. **激活稀疏方法（TEAL、CATS）**：在投影层输入上做幅度阈值稀疏，本文与其在线性/MLP 投影上有机制重叠，但 RMM 进一步扩展到 $QK^\top$ 和 $PV$ 等注意力内部矩阵乘积，且选择策略为 TopK 而非阈值。
4. **随机近似矩阵乘法（Drineas et al., 2006）**：基于重要性采样的蒙特卡洛估计，需多次采样降方差；本文采用确定性激活感知选择，适合单次前向且误差可控。
5. **结构化剪枝/低秩近似（Sun et al., Ma et al., Gao et al.）**：关注模型结构的预压缩，本文侧重于推理时按需缩减活跃计算，不减少参数量与存储。

## 局限性与未来方向
- 聚焦于免训练设定，未探索结合微调/预训练目标的可学习自适应缩减。
- 多模态模型的组件冗余模式尚未系统研究，VLM 各子模块（视觉编码器、投影、语言解码）的差异化缩减策略有待探索。
- 当前实现未与所有 Transformer 组件、推理框架及量化后端完全融合，端到端收益依赖序列长度、硬件和 kernel 集成开销。
- RMM 不减少模型参数量与权重存储内存，仅在活跃计算层面节省；与低比特量化的联合优化 kernel 仍是未来方向。
- 保留率需手动设定或依赖无标签一致性扫描，尚未形成通用的自动校准方案。

## 研究启发与可借鉴点
1. **最小最大最优的选择理论**：TopK 按激活列范数选择在信息不对称设定下的最优性证明，可作为类似"输入自适应计算缩减"方法的设计范式。
2. **组件级冗余非对称的发现方法**：通过逐项消融 attention-side vs MLP-side 及各投影（Up/Gate/Down）的敏感度，揭示了结构化部署策略的价值——可迁移至其他模型架构的冗余分析。
3. **矩阵乘积视角的统一框架**：将注意力内部乘积与线性投影纳入同一缩减语言，避免了针对各操作分别设计优化器的碎片化思路。
4. **无标签一致性校准思路**：用小规模无标签数据做密集/缩减模型的输出一致性扫描以自动选取 RR，为免训练方法的超参调优提供了实用启发。
5. **与 INT8 量化的兼容性验证**：证明 RMM 可与现有量化流程叠加作用于不同方面（权重精度 vs 活跃计算），为多层级推理优化组合提供了实证参考。

## 关键术语表
**Reduced Matrix Multiplication (RMM)**：一种训练免、输入自适应的 Transformer 推理加速方法，在矩阵乘法的共享收缩维度上按激活范数选取 Top-K 索引进行计算缩减。
**Retention Ratio (RR)**：用户可控的保留率参数 $\rho$，决定每个矩阵乘法中保留的收缩维度比例，控制精度–效率权衡。
**Minimax Optimality**：在仅可见激活矩阵 A、不可见权重矩阵 B 的信息不对称设定下，TopK 按列范数选择是最小化最坏情况近似误差的最优策略。
**Retained Energy**：被选中维度所捕获的激活二范数平方占总能量（$\|A\|_F^2$）的比例，用于诊断缩减对各组件的影响程度。
**Structural Asymmetry**：Transformer 内部 attention-side 计算冗余度高、更易缩减，而 MLP 组件（尤其 Up 投影）对缩减更敏感的结构非对称现象。
**Activation-Aware Selection**：根据当前输入的激活大小动态决定保留哪些维度，使缩减模式随输入、层、头和解码步变化。

## 可复现要素
- **数据集/基准**：COPA、PIQA、COMMONSENSEQA、ARC-Easy、ARC-Challenge、MMLU、WIKITEXT、BOOKCORPUS、GSM8K、HUMANEVAL、RULER-CWE、RULER-HOTPOT、CNN/DAILYMAIL、POPE、BLINK Art Style/Forensic Detection/Counting；均为标准公开基准。
- **代码/权重**：代码和项目资源将在 https://github.com/Zesearch/rmm-llm 开源；使用官方预训练权重，无需微调。
- **关键超参**：保留率 RR ∈ {0.9, 0.8, 0.7, 0.6, 0.5}；注意力 feature 保留率 $\rho_d$ 和 token 保留率 $\rho_t$；MLP 各投影可独立设置；INT8 实验使用 bitsandbytes 量化加载。
- **硬件**：NVIDIA A100、RTX A6000、RTX 6000 Ada、L40S；bfloat16 精度；推理使用 PyTorch + HuggingFace Transformers。
- **Batch size**：7B/8B 模型通常为 8–16，70B 为 1–2；长上下文评估最长至 30K tokens。
