---
title: "AWARe-Mitigating-Catastrophic-Forgetting-via-Activation-Weig"
source: https://arxiv.org/pdf/2608.11758v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:33:25"
field: "多模态持续学习"
keywords: ["灾难性遗忘", "多模态大语言模型", "参数高效微调", "激活重要性", "持续学习"]
innovations: ["基于激活权重的自适应参数保留框架 AWARe，首次将激活 saliency 从压缩领域迁移至 MLLM 遗忘缓解", "行级梯度掩码冻结机制，无需修改模型架构且训练后可恢复原始配置", "通用校准集（MMMU）替代方案，在上游数据不可得时仍保持高性能"]
benchmarks: ["IconQA", "COCO-Caption", "MLLM-DCL", "Qwen2.5-VL-7B on MLLM-DCL"]
---

# 论文速读：AWARe-Mitigating-Catastrophic-Forgetting-via-Activation-Weig

## 一句话总结
AWARe 提出了一种基于激活权重的自适应参数保留方法，在下游微调前通过分析上游校准数据的神经元激活重要性，选择性冻结关键参数，从而在保持 MLLM 通用能力的同时有效适应新任务，显著缓解灾难性遗忘问题。

## 研究问题与动机
1. **灾难性遗忘是 MLLM 部署的核心瓶颈**：MLLM 在下游任务微调时，新任务梯度更新会覆盖对上游能力至关重要的参数，导致已有能力严重退化。
2. **现有方法难以兼顾效率与稳定性**：经验回放需要存储大量数据，计算开销高；正则化方法难以扩展到十亿级参数规模；LoRA 等 PEFT 方法虽减少训练参数，但仍无法阻止预训练知识的侵蚀。
3. **梯度/静态先验驱动的选择不充分**：SPIDER、Model Tailor 等方法依赖梯度或静态特征分析，忽略了前向传播过程中神经元激活的动态特性，难以准确识别真正重要的参数子集。
4. **激活 saliency 在压缩领域已有验证但未被用于遗忘缓解**：AWQ、Wanda 等工作证明激活感知的重要性估计能有效指导量化与剪枝，本文将其创新性地迁移至持续学习场景。

## 核心贡献（创新点）
1. **提出 AWARe 框架**：将激活 saliency 估计与梯度掩码相结合，在微调阶段冻结高重要性神经元，本质区别在于利用动态激活分布而非静态权重幅度来判断参数重要性。
2. **零架构修改的即插即用设计**：通过 `AwareLinear` 训练期 wrapper 实现行级冻结，训练后可移除，不改变推理引擎兼容性，区别于 DO RA、LoRASculpt 等引入额外模块或复杂稀疏结构的方法。
3. **通用校准集替代方案**：当上游数据不可得时，可使用 MMMU 等通用多模态基准作为校准集，仅增加极低开销仍保持高性能，区别于依赖上游数据复现的传统 continual learning 方法。
4. **系统的消融与敏感性分析**：揭示了 self-attention 投影层（q/k/v_proj）与 mm_projector 的选择对稳定性-可塑性平衡的关键作用，以及 30% 冻结比例的全局最高激活策略的最优性。

## 方法详解
**两阶段流程**：

1. **知识画像（Knowledge Profiling）**：
   - 从上游任务训练集（或通用数据集如 MMMU）中采样校准集 $S_{cal}$（每任务约 200 条）。
   - 对目标线性层（q_proj, k_proj, v_proj, mm_projector），执行前向传播获取激活张量 $\mathbf{A} \in \mathbb{R}^{B \times L \times d_{out}}$。
   - 三阶段 saliency 计算：
     - 步骤一：沿序列长度维度计算 L2-norm → $a'_{i,k} = \sqrt{\sum_{j=1}^{L} a^2_{i,j,k}}$
     - 步骤二：逐样本沿隐藏维度做 L2-normalization → $a''_{i,k} = a'_{i,k} / \|\mathbf{a}'_i\|_2$，确保 saliency 反映神经元相对重要性而非绝对尺度。
     - 步骤三：沿批次维度取平均 → $s_k = \frac{1}{B}\sum_{i=1}^{B} a''_{i,k}$，得到输出神经元重要性向量 $\mathbf{s} \in \mathbb{R}^{d_{out}}$。

2. **受限微调（Constrained Fine-tuning）**：
   - 设定保留比例 $\rho$（默认 30%），按 saliency 全局排序，选取 Top-$\rho$ 神经元索引集 $\mathcal{I}$。
   - 构建行级梯度掩码 $\mathbf{M}_{grad} \in \{0,1\}^{d_{out} \times d_{in}}$：若行索引 $k \in \mathcal{I}$，则该行置 0（冻结）；否则置 1。
   - 参数更新公式：$\mathbf{W}^{(t+1)} = \mathbf{W}^{(t)} - \eta \cdot (\nabla_\mathbf{W} \mathcal{L}_{down} \odot \mathbf{M}_{grad})$。
   - 实现上采用 `AwareLinear` wrapper，训练结束后移除，保持原始模型架构不变。
   - 视觉编码器全程冻结。

## 实验与结果
**基座模型**：LLaVA-v1.5-7b（另含 Qwen2.5-VL-7B 扩展实验）

**评估数据集**：
- 上游任务：OKVQA、OCRVQA、GQA、TextVQA
- 下游单任务：IconQA（VQA）、COCO-Caption（图像描述，CIDEr 指标）
- 持续学习：MLLM-DCL 五任务序列（RS、Med、AD、Sci、Fin）

**核心指标**：Retention (R)、Efficiency (E)、Harmonic Mean (H)

**单任务下游适配结果（Table 1）**：
- **AWARe（上游校准）**：IconQA H=**103.2**（R=98.4, E=108.4），COCO-Caption H=**108.0**（↑3.8 vs 次优），均优于所有基线。
- **AWARe (MMMU)**：无上游数据时仅用通用校准集，IconQA H=101.7（↑3.1），COCO-Caption H=107.9（↑3.7），性能损失极小。
- Full-FT 上游任务几乎完全遗忘（R≈0.2~0.4），凸显遗忘问题的严重性。

**持续学习结果（Table 2, MLLM-DCL）**：
- AWARe Avg=**65.58**，Last=**62.10**，均超越最强基线 DISCO（Avg=62.40, Last=60.33），提升幅度 +3.18 / +1.77。
- Backward Transfer (BWT) = -6.74，优于 DISCO 的 -5.57 但整体轨迹最优。

**消融关键结论**：
- 激活 saliency 选择显著优于随机选择、权重范数、混合策略（Table 3）。
- 全局最高激活策略 30% 冻结比例达到最优 H 值（Table 4）。
- 联合 target q/k/v_proj + mm_projector 效果最佳；冻结 MLP 会导致上游性能暴跌至 30 分以下（Figure 5）。
- 校准集大小 200 条已足够稳定，标准差 <0.5（Table 6）。

## 相关工作脉络
1. **LoRA / DoRA**：低秩适配器方法，减少可训练参数但缺乏主动遗忘保护机制（Biderman et al., 2024 指出 LoRA 仍会破坏上游特征）。AWARe 不依赖低秩分解，直接约束原始参数。
2. **Model Tailor / SPIDER / LoRASculpt**：基于梯度或 saliency 的选择性微调方法，依赖静态或一次性统计量。AWARe 利用跨样本激活分布的相对重要性，更具动态适应性。
3. **Orth-Reg / CorDA**：正则化导向方法，前者鼓励正交性，后者通过上下文分解构建 adapter。AWARe 无需额外正则项或 adapter，纯参数冻结实现。
4. **AWQ / Wanda**：模型压缩领域激活感知量化与剪枝工作，证明激活统计能有效估计权重重要性。本文首次将此思想系统迁移至 MLLM 灾难性遗忘缓解。
5. **ModalPrompt / HiDe / SEFE / DISCO / CL-MoE**：持续学习领域近期 SOTA，大多引入 prompt、MoE 或联邦学习机制。AWARe 以极简参数冻结策略在 MLLM-DCL 上取得最优平均表现。

## 局限性与未来方向
1. **校准集质量敏感性**：当上游数据不可得时使用通用数据集（如 MMMU）替代，其覆盖范围可能限制对特定领域知识的保护效果。
2. **模型规模泛化待验证**：实验仅在 7B 量级 LLaVA-v1.5 和 Qwen2.5-VL 上进行，更大规模模型（如 70B+）的行为未知。
3. **任务序列长度受限**：MLLM-DCL 仅包含 5 个任务，更长的持续学习序列下冻结策略的累积效应未充分评估。
4. **极端分布偏移仍需调参**：当下游任务与上游差异极大时，固定 $\rho$ 可能需要在稳定性与可塑性间重新权衡。

## 研究启发与可借鉴点
1. **激活 saliency 作为参数重要性代理指标**：可从压缩/剪枝领域直接迁移至遗忘缓解场景，计算成本极低（单次前向传播），具有广泛应用潜力。
2. **行级冻结机制（AwareLinear wrapper）**：无需修改训练框架即可实现选择性参数冻结，是一种轻量且通用的工程技巧。
3. **per-sample L2 normalization 的 saliency 计算设计**：消除了样本间绝对尺度差异的影响，使 saliency 反映的是神经元相对贡献，这一设计值得在其他 saliency-based 方法中借鉴。
4. **mm_projector 纳入目标层的选择**：消融实验揭示 mm_projector 对下游概念学习至关重要，将其与 self-attention 投影联合保护/微调的策略为多模态适配提供了新视角。
5. **通用校准集可行性验证**：证明了在数据受限场景下使用广泛可用的基准（如 MMMU）进行 profile 是可行的，降低了方法部署门槛。

## 关键术语表
**Catastrophic Forgetting（灾难性遗忘）**：深度学习模型在学习新任务时，原有已学知识被严重覆盖或遗忘的现象。
**Activation-based Saliency（基于激活的重要性评分）**：通过校准样本前向传播统计各神经元的激活强度，以此评估参数对保留预训练知识的重要性。
**Retention Ratio（保留比例）ρ**：控制需冻结的高重要性神经元占总参数的比例，本文最优值为 30%。
**MLLM-DCL（Multimodal LLM Continuous Learning）**：面向多模态大语言模型的持续指令微调基准，包含遥感、医学 VQA、自动驾驶等五任务序列。
**Global-Highest Selection（全局最高激活选择）**：跨所有目标层统一排序 saliency 并选取 Top-k%，而非逐层独立选择。
**AwareLinear**：训练期 wrapper，将线性层分解为活跃与冻结两部分以实现掩码梯度更新，训练结束后移除不影响推理。

## 可复现要素
- **数据集**：OKVQA、OCRVQA、GQA、TextVQA、IconQA、COCO-Caption、MLLM-DCL（均公开）；MMMU（公开）。
- **代码开源**：https://github.com/kaln27/AWARe
- **基座模型**：LLaVA-v1.5-7b、Qwen2.5-VL-7B（公开权重）。
- **关键超参**：ρ=30%，校准集每任务 200 条（或 MMMU 600 条），Epoch=3，LLM 学习率=2e-5，mm_projector 学习率=2e-4，Batch Size=16，Cosine Annealing + 0.03 warmup，AdamW。
- **目标层**：q_proj, k_proj, v_proj, mm_projector。
