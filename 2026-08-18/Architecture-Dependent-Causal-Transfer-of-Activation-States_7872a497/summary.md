---
title: "Architecture-Dependent-Causal-Transfer-of-Activation-States"
source: https://arxiv.org/pdf/2608.16347v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:25:05"
---

# 论文速读：Architecture-Dependent-Causal-Transfer-of-Activation-States

## 一句话总结
论文系统检验了不同架构的 LLM 能否绕过自然语言中介、直接通过 learned projection 因果性转移内部激活状态。实验表明跨模型表征对齐确实存在且优于随机基线，投影网络可在因果解码器架构对之间实现远高于随机的检索准确率，但端到端生成级因果注入仅在 Qwen2-0.5B→Phi-3-mini 一对中显著成功，证明激活级转移是架构依赖的而非普适的，且成功转移的是“表征载体”而非“语义内容”。

## 研究问题与动机
1. **通信开销与隐式假设**：当前 AI 系统间通信几乎全部依赖自然语言作为中间层，带来编码/解码开销、token 成本、延迟与潜在信息损失；本文追问：不同架构的 LLM 内部表征空间是否足够相似，从而支持直接交换激活状态？
2. **相似性 ≠ 可利用性**：现有工作（如 CKA、neuron-level correlation）已证明独立训练模型的表征存在相似性，但未验证这种相似性能否被显式映射并利用，更未证明其能在控制实验中因果性地改变目标模型行为。
3. **对齐指标的真实性检验**：主流对齐指标（CKA、Procrustes）易受少数高幅值激活维度与 superposition 干扰，需引入随机初始化 null baseline 与更稳健的 rank-based 指标验证信号是否真正依赖训练权重。
4. **柏拉图表征假说的实证边界**：Platonic Representation Hypothesis 主张表征会跨架构趋同；本文希望通过 RELATED vs. SYNONYM 控制与架构族系对比，检验这种趋同究竟源于架构共性还是数据共现统计。

## 核心贡献（创新点）
1. **三阶段递进实验范式**：提出“被动相似度分析→主动跨模型投影检索→端到端生成级因果注入”的检验链条，并在各阶段预先注册成功标准、负对照与统计校正方案，避免事后解释偏差。
2. **指标稳健性修正**：实证发现 CKA 与 Orthogonal Procrustes 在中间层会出现 trained < untrained 的反常排序，提出以 rank-based 的 mutual k-NN alignment 作为主指标，有效规避 superposition 驱动的高幅值维度失真。
3. **因果边界的哲学-工程双重划定**：明确将成功 claim 限定为“representational vehicle（表征载体）”的因果转移，而非“content（语义/理解）”的共享，借助哲学中的 vehicle/content 区分与 symbol grounding 问题严格界定贡献范围。
4. **架构族系的预测力证实**：发现 causal decoder-only 架构家族内部的表征对齐显著强于其与 bidirectional encoder（FLAN-T5）的对齐，架构类型比训练机构或数据源更能预测跨模型投影与转移的可行性。

## 方法详解
- **模型与数据构造**：选用 Qwen2-0.5B、Phi-3-mini-128k-instruct（4-bit）、Mistral-7B-v0.1（4-bit）与 FLAN-T5-base，覆盖三种开发方与两种注意力机制。构建 12 对 RELATED（话题/关联）与 12 对 SYNONYM（语义近义但词汇相异）概念对；对 causal 模型排除 position 0 进行 mean-pooling，以规避 attention sinks 与首 token 主导激活造成的伪接近。
- **对齐与 null baseline**：使用 CKA、Orthogonal Procrustes、mutual k-NN 三种指标；仅对 Qwen2/FLAN-T5 对计算相同架构随机初始化模型的 null baseline；SIGNIFICANCE 通过置换检验与 Benjamini–Hochberg FDR 校正（共 60 次比较）。
- **跨模型投影网络**：训练 MLP 将源模型中层（fraction=0.5）隐藏状态映射至目标模型同层空间；训练/评估数据与相似度分析正交。评估 metric 为 top-1 retrieval accuracy（20 候选池，chance=5%），并设 untrained MLP 为架构基线。
- **端到端因果状态转移**
