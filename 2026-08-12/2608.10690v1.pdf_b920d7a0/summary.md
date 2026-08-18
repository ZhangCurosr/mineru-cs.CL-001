---
title: "Can Released LLM Vocabularies Support Token-Level Estimation of Hidden Corpora?"
source: https://arxiv.org/pdf/2608.10690v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:36:44"
---

# 论文速读：Can Released LLM Vocabularies Support Token-Level Estimation of Hidden Corpora?

## 一句话总结
本文验证了不同语料训练出的 BPE 词表中“Token ID–词频占比”分布具有跨语言/跨领域的稳定可迁移性，并据此提出 QGDE 方法：通过多分位数趋势拟合与局部高斯密度加权，仅凭发布的 tokenizer 词表即可实现对隐藏预训练语料中任意 Token 占比的细粒度估计，Token 级平均相对误差最低降至 3.00%，聚合至类别级后为 3.08%。

## 研究问题与动机
- LLM 权重虽常公开，但预训练语料构成通常保密，现有基于发布词表的研究（如 DMI、PoCTrace）仅能输出语类/领域级的粗粒度混合比例，或仅针对特定污染 Token 组做 prevalence 估计，缺乏对**任意目标 Token** 的细粒度反演能力。
- 核心科学假设尚未系统验证：不同语料上训练的 BPE 词表是否共享稳定的 Token ID–ratio 全局分布形态，从而支持从已知语料到隐藏目标语料的分布迁移。
- 现有中位数/单曲线估计方法会丢弃同一 ID 附近的垂直分布宽度信息，难以支撑点级别的高精度预测。
- 需要一种既能保留分布形态、又能输出单点估计，且对源语料混合比例具有一定鲁棒性的通用框架。

## 核心贡献（创新点）
- **揭示 ID–ratio 分布的跨语料可迁移性**：通过 KL 散度与熵归一化构造 directional transfer similarity score，证明英/法/日/中文本及 Web/Wiki/Code/Math 等不同语言与领域训练的 BPE 词表在 log-log 空间中共享稳定的全局分布形态。
- **提出 QGDE（分位数引导密度估计）框架**：利用多分位数回归拟合 ID–ratio 趋势族，结合局部高斯核密度加权，将粗粒度的分布信号转化为任意目标 Token 的精确占比点估计。
- **设计 Quantile Anchor Coverage (QAC) 目标**：以垂直带宽覆盖度最大化自动筛选最优分位数锚点集合，避免冗余趋势，实验表明 K≈11–14 时覆盖度与估计误差均趋于饱和。
- **建立 Token 级到 Category 级的聚合链路**：提出基于目标 token 在源类别中出现频次的分配权重 $\pi_{c,i}$，实现细粒度估计向语类/领域混合比例的无损汇总，在控制实验与真实 SmolLM 场景下均显著优于 DMI 等基线。

## 方法详解
- **问题形式化**：给定目标 tokenizer 词表 $\mathcal{V}^*=\{(v_i, t_i)\}$，学习映射 $\widehat{f}(t_i) \approx r_i^*$，并支持对 token 子集 $S$ 求和得到类别混合比例 $\widehat{R}(S)=\sum_{i\in S}\widehat{r}_i$。
- **多分位数趋势拟合**：基于 Zipf 定律，将已知语料的 token ID 与占比转换至 log-log 空间 $(x_j=\log t_j, y_j=\log r_j)$，对每个分位数水平 $\tau$ 使用非对称损失 $\rho_\tau$ 拟合线性趋势 $q_\tau(x)=a_\tau+b_\tau x$，保留垂直方向的分布宽度。
- **分位数锚点选择（QAC）**：定义覆盖度 $C(\mathcal{T}_K)=\sum_{(x_j,y_j)\in\mathcal{P}}\mathbf{1}[\min_{\tau\in\mathcal{T}_K}|y_j-q_\tau(x_j)|<h_y]$，通过网格搜索选取使覆盖度最大的 $K$ 个锚点 $\mathcal{T}_K^\star$，确保趋势覆盖密集区而非稀疏/冗余区。
- **局部密度加权点估计**：对目标 token ID $t_i$，各锚点给出候选对数占比 $z_{i,\tau}=q_\tau(\log t_i)$；在局部 ID 窗口 $\mathcal{N}_i$ 内以候选值为中心计算高斯核权重 $W_{i,\tau}=\sum_{(x_j,y_j)\in\mathcal{N}_i}\exp(-(y_j-z_{i,\tau})^2/(2h_y^2))$，最终估计 $\widehat{y}_i=\sum_\tau \frac{W_{i,\tau}}{\sum_{\tau'}W_{i,\tau'}}z_{i,\tau}$。
- **类别级聚合**：按 token 在各源类别语料中的出现频次归一化分配权重 $\pi_{c,i}=n_{c,i}/\sum_{c'}n_{c',i}$，计算 $\widehat{\alpha}_c=\sum_{i\in\mathbb{Z}}\frac{\widehat{r}_i}{\sum_j\widehat{r}_j}\pi_{c,i}$，实现从 token 级到语言/领域比例的转换。

## 实验与结果
- **数据集与设置**：语言域源为 mC4 四语切片，目标为 OSCAR 同语种；领域域源为 FineWeb/Wiki/CodeParrot/OpenWeb-Math，目标为 RedPajama-C4/BookCorpus/RedPajama-GitHub/FineWebMath；真实场景采用 SmolLM tokenizer 及其训练语料（FineWebedu 87.30%、Cosmopedia-v2 11.11%、Pythonedu 1.59%）。
- **评估基线**：Direct ID-ratio Transfer（按位置复制占比）、PoCTrace（单中位趋势）、DMI（类别级 source-independent 基线）。
- **主要结果**：
  - **控制实验**：QGDE 在绝大多数源混合设置下显著优于基线；语言设置最佳 MRE 为 **3.00%**（K=11, target-like），领域设置最佳 MRE 为 **3.08%**（K=12, target-like）；混合源（70%-mixed）普遍优于单一源，且主导组件比例变化对误差影响较小。
  - **
