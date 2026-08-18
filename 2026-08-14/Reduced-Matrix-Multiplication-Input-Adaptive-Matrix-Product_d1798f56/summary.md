---
title: "Reduced-Matrix-Multiplication-Input-Adaptive-Matrix-Product"
source: https://arxiv.org/pdf/2608.13426v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:49:00"
field: "大语言模型高效推理"
keywords: ["inference acceleration", "matrix multiplication", "input-adaptive pruning", "Transformer", "training-free optimization"]
innovations: ["提出RMM，沿矩阵乘法收缩轴动态选择TopK激活维度以训练自由方式削减计算", "证明TopK by column norm在信息不对称下的minimax最优性", "揭示Attention侧与MLP侧冗余度的结构性非对称，提出组件感知异质保留策略"]
benchmarks: ["ARC-Challenge", "ARC-Easy", "COPA", "PIQA", "CommonsenseQA", "GSM8K", "HumanEval", "MMLU", "WikiText", "BookCorpus", "CNN/DailyMail", "RULER", "POPE", "Blink Art Style", "Blink Forensic Detection", "Blink Counting"]
---

# 论文速读：Reduced-Matrix-Multiplication-Input-Adaptive-Matrix-Product

## 一句话总结
本文提出 **Reduced Matrix Multiplication (RMM)**，一种训练自由、输入自适应的 Transformer 推理加速方法，通过在矩阵乘法的共享收缩维度上动态选择关键索引来减少计算量；实验表明该方法可在大幅降低 FLOPs 的同时保持较高精度，且对 LLaMA、Qwen 等 1B–70B 模型及多模态模型均有效。

## 研究问题与动机
- Transformer 推理成本随模型规模快速上升，注意力（attention）和 MLP 层中大量高维矩阵乘法是主要瓶颈。
- 现有方法（结构化剪枝、低秩近似、KV-cache 管理、激活稀疏化等）多作用于固定模型结构或输入序列，**未直接回答**：每个 Transformer 矩阵乘法的收缩轴是否需要根据当前输入自适应削减？
- 激活稀疏化（如 TEAL、CATS）作用于投影层的输入激活张量，而 RMM 面向的是矩阵乘法的**共享收缩维度**，覆盖范围更广（包括 $QK^\top$ 和 $PV$）。
- 静态剪枝在给定数据集上选取固定维度子集后无法泛化到其它任务；动态激活感知选择是否更具鲁棒性尚未有系统验证。

## 核心贡献（创新点）
- **提出 RMM（Reduced Matrix Multiplication）**：一种无需训练、基于当前激活幅值沿矩阵乘法收缩轴动态选择 TopK 索引的方法，直接替换 Transformer 中 attention 与 MLP 的核心矩阵乘法。
- **证明 TopK by column norm 的 minimax 最优性**：在仅观测激活矩阵 $A$、未知权重 $B$ 的信息不对称条件下，按列范数选取最大的 $\rho d$ 个维度可最小化最坏情况近似误差。
- **揭示 Transformer 内部结构性非对称**：注意力侧计算（attention-side）的冗余度远高于 MLP 侧；同等 RR 下，MLP 剪枝导致性能断崖式下降，而注意力剪枝表现稳健。
- **提供可控制的精度–效率权衡**：通过单一超参 retention ratio $\rho$，在判别、自回归生成、长上下文、多模态等多场景下均实现平滑、可预测的降算性能折中。
- **验证端到端延迟收益**：在 A100 上使用自定义 Triton kernel 实现，序列长度 4096 时 attention GEMM 最大提速 1.89×，端到端最大 1.40×；在 70B 模型 4096 长度下避免 OOM。

## 方法详解
- **统一矩阵乘形式**：将 Transformer 中所有核心运算抽象为 $Y = AB$，其中 $A \in \mathbb{R}^{n \times d}$ 为当前输入激活，$B \in \mathbb{R}^{d \times m}$ 为权重或中间表示，$d$ 为共享收缩维度。
- **RMM 定义**：给定保留比 $\rho \in (0,1]$，选择索引集 $\mathcal{I} = \text{TopK}(\{\|A_{:,j}\|_2\}_{j=1}^d,\lceil\rho d\rceil)$，计算 $\text{RMM}_\rho(A,B) = A_{:,\mathcal{I}} B_{\mathcal{I},:}$。
- **激活感知维度选择**：以激活列范数 $\|A_{:,j}\|_2$ 作为特征重要性得分，完全确定性地选取当前输入下最重要的特征维度，每次前向均动态重算。
- **Attention 应用**：对单头注意力的 $Q \in \mathbb{R}^{L_q \times d_h}$，沿 head 特征维度选 $\mathcal{I}$ 得到缩减注意力分数 $\widetilde{S} = \frac{1}{\sqrt{d_h}} Q_{:,\mathcal{I}} K_{:,\mathcal{I}}^\top$；同时可选地沿 token 维度对 $PV$ 做同样选择。
- **MLP/线性投影应用**：对激活 $X$ 和权重 $W$，按同样规则沿隐藏维度选 $\mathcal{I}$，计算 $X_{:,\mathcal{I}} W_{\mathcal{I},:}$。
- **复杂度**：将 $O(ndm)$ 降至 $O(n\rho_d dm)$；$QK^\top$ 从 $O(L_q L_k d_h)$ 降至 $O(L_q L_k \rho_d d_h)$；token 级 $PV$ 从 $O(L_q L_k d_h)$ 降至 $O(L_q \rho_t L_k d_h)$。选择开销 $O(nd)$ 相对可忽略。
- **误差界**：$\|AB - A_{:,\mathcal{I}}B_{\mathcal{I},:}\|_F \leq \sum_{j \notin \mathcal{I}} \|A_{:,j}\|_2 \|B_{j,:}\|_2$；能量集中时误差小，且 TopK 在该上界下是最优的。

## 实验与结果
- **模型与任务**：LLaMA 3.1 70B/8B、LLaMA 3.2 3B/1.5B、Qwen3 32B、Qwen3.1 7B、Qwen2.5-VL-7B；涵盖 ARC-C/E、COPA、PIQA、CommonsenseQA、GSM8K、HumanEval、MMLU、WikiText、BookCorpus、CNN/DailyMail 摘要、RULER 长上下文、POPE/Blink 多模态任务；全零样本评估。
- **静态剪枝对比（LLaMA 3.1 8B, RR=0.5）**：RMM 平均准确率 **59.8%**，显著优于 SparseGPT（56.1%）、Wanda（52.7%）、SliceGPT（37.0%）、Magnitude（39.3%）和 Full（69.8%）。
- **摘要生成（LLaMA 3.1 8B, RR=0.5）**：RMM ROUGE-1 = **34.2**，优于 Static（28.0）、Random（5.7）、H2O（24.4）；RR=0.8 时 RMM（37.5）接近 Full（37.4）。
- **跨模型缩放（Table 3）**：相同 RR 下更大模型更鲁棒；LLaMA 3.1 70B 在 RR=0.8 时多数基准几乎无损，1B/3B 较小模型在 RR≤0.7 出现更早的性能拐点。
- **长上下文（RULER, LLaMA 3.1 8B）**：RR=0.8/0.5 在 5K/15K/30K 各长度上与 Full 几乎一致（CWE 98.0–98.2、Hotpot 50.5–56.4），无系统性劣化。
- **多模态（Qwen2.5-VL-7B, Table 6）**：RR=0.8 时 POPE=82.0（Full 83.7）、Blink 各任务接近满模型；RR=0.5 仍显著优于 Static/Random。
- **组件级分析（LLaMA 3.1 8B, RR=0.7, Table 12）**：Attention-side 准确率下降仅 3.52 分（保留能量 89.69%）；MLP-Up 下降 16.32 分、Gate 7.20、Down 3.51；Whole-MLP 下降 18.78 分，证实 MLP 侧高度敏感且不可整体粗剪。
- **延迟（A100, LLaMA 3.1 8B, RR=0.8）**：Kernel 级 $QK^\top$ 最高 1.56×、$AV$ 最高 1.89×；端到端 seq=4096 时 1.40×；LLaMA 3.1 70B 在 seq=4096 时 Dense 侧 OOM、RMM 完成推理。
- **INT8 兼容（Table 14）**：与 bitsandbytes INT8 并行使用时，RR=0.8 在 COPA 上 77.4 vs Baseline 81.4，行为可控。

## 相关工作脉络
- **静态结构化剪枝（SparseGPT、Wanda、SliceGPT、Magnitude Pruning）**：在推理前确定固定剔除模式，作用于权重本身；RMM 不改变权重，在推理时按当前激活动态选择，保证更好的泛化与稳定性。
- **输入/KV 级优化（H2O、token 压缩、sparse attention、调度）**：聚焦上下文序列长度或缓存管理；RMM 直接作用于 Transformer 内部矩阵乘法的收缩轴，二者作用层面互补而非替代。
- **训练自由激活稀疏化（TEAL、CATS）**：对投影层输入做幅度阈值；RMM 在投影层与之部分重叠，但统一 formulation 可扩展到 $QK^\top$、$PV$ 等内部注意力乘法，覆盖范围更广。
- **随机近似矩阵乘法（Drineas et al., 2006）**：基于重要分布随机采样估计矩阵积；RMM 采用确定性、激活感知的 TopK 选择，单次前向无需多次采样平均，误差可控、可预测。
- **Transformer 冗余分析（Peng et al., 2023; Sajjad et al., 2023）**：证明神经元/头/层存在冗余；RMM 在此基础上提出“收缩轴粒度”的系统性动态削减框架并给出理论保证。

## 局限性与未来方向
- 当前为训练自由方法，**未探索**结合预训练目标引入可学习压缩/自适应计算的设计。
- 多模态模型内部不同组件（视觉 encoder、投影、LM decoder）在动态剪枝下的冗余模式尚未系统研究。
- 单一 retention ratio 并非普适，实际部署需针对模型、任务、组件分别调节；论文建议通过无标签一致性扫描选择，但缺乏自动化工具链。
- 当前实现仅替换部分 attention kernel（Triton），未与完整 Transformer 框架、所有组件及量化后端完全融合，端到端收益受序列长度、硬件、kernel 开销影响较大。
- RMM 不减少模型参数与存储权重内存，仅减少活跃计算；对内存带宽受限场景的收益需进一步评估。

## 研究启发与可借鉴点
- **Minimax 最优性证明思路**可推广至其他输入自适应近似场景：在仅观测一operand 的信息不对称下，选择 TopK by norm 作为通用近似策略具有理论保障。
- **组件感知的异质 retention 策略**（attention 激进、MLP 保守、MLP 内 Up/Gate/Down 分别设 RR）是可直接复用的工程经验，能有效提升整体精度–效率折中。
- **无标签一致性扫描**（用少量 unlabeled prompts 比较 full vs reduced 输出一致率）可作为部署时自动确定 RR 的轻量实践，值得借鉴于本团队的推理优化管线。
- **能量保留率（Retained Energy）作为诊断指标**：可将 $\epsilon_A(\rho)$ 与性能退化关联分析，辅助设计“按激活能量集中程度动态调参”的自适应系统。
- RMM 与 INT8 量化的兼容性证明了“计算轴削减 + 权重精度压缩”可在同一推理管线上叠加使用，可作为后续多技术融合加速的起点。

## 关键术语表
- **Reduced Matrix Multiplication (RMM)**：在矩阵乘法 $Y=AB$ 中，依据当前激活 $A$ 的列范数沿共享收缩维度动态选取 TopK 索引进行计算的方法，不修改模型权重。
- **Retention Ratio (RR, $\rho$)**：用户可控的保留比例超参，决定每个矩阵乘法保留收缩轴维度的分数，控制精度–效率权衡。
- **Activation-aware dimension selection**：以激活矩阵列范数 $\|A_{:,j}\|_2$ 作为特征重要性度量，动态选取当前输入下最重要的特征维度。
- **Minimax optimality**：在仅能观测 $A$ 而 $B$ 未知的信息不对称约束下，TopK by column norm 选择可最小化最坏情况近似误差上界。
- **Attention-side vs MLP-side redundancy asymmetry**：注意力计算（$QK^\top$、$PV$）可被大幅削减而性能稳健，MLP 侧（尤其 Up 投影）对削减极度敏感。
- **Retained Energy**：被选中的特征维度所保留的激活平方范数占比，用于衡量所选子空间的信息覆盖度。
- **TEAL / CATS**：训练自由激活稀疏化基线方法，对投影层输入激活做幅度阈值化；RMM 在投影层与其部分重叠但覆盖更广。
- **RULER**：用于评估长上下文推理能力的基准（含 CWE、Hotpot 等子任务），测试模型在 5K/15K/30K 长度下的表现。

## 可复现要素
- **代码**：论文声明代码与项目资源将在 https://github.com/Zesearch/rmm-llm 开源（提交时未发布）。
- **模型权重**：使用官方预训练权重，无需微调；LLaMA 3.1/3.2、Qwen3/Qwen3.1、Qwen2.5-VL 等均可从官方渠道获取。
- **数据集/基准**：ARC-E/C、COPA、PIQA、CommonsenseQA、GSM8K、HumanEval、MMLU、WikiText、BookCorpus、CNN/DailyMail、RULER、POPE、Blink 等，均为公开基准。
- **关键超参**：retention ratio $\rho \in \{0.9, 0.8, 0.7, 0.6, 0.5\}$，主实验通常对 target 矩阵乘统一使用同一 RR；component-aware 策略中不同组件可设不同 RR。
- **硬件与实现**：主实验在 NVIDIA A100/RTX A6000/6000 Ada/L40S；精度 bfloat16；使用 Hugging Face Transformers + 自定义 Triton kernel；batch size 按模型规模取 1–16。
- **INT8 实验**：使用 bitsandbytes INT8 权重加载。
