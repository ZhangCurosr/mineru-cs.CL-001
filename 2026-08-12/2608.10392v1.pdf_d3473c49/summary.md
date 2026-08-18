---
title: "Share First, Route What Remains:"
source: https://arxiv.org/pdf/2608.10392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:08:04"
---

# 论文速读：Share First, Route What Remains:

## 一句话总结
本文提出 UniF-MoE，一个将共享专家建模、细粒度通道选择与动态路由统一为“先共享、后路由余量”顺序过程的 Token-Adaptive MoE 框架，在 DomainBed 与 GLUE 基准上以更低激活计算量实现更优精度，并显著降低推理延迟与显存占用。

## 研究问题与动机
- 现有 MoE 研究通常将共享专家（shared experts）、细粒度计算（fine-grained computation）与动态路由（dynamic routing）作为独立机制并行开发，忽略了三者之间存在的基础依赖关系。
- 传统 top-k 路由将完整专家视为最小计算单元，且对所有 token 分配固定专家数，导致简单 token 出现重复计算、困难 token 容量不足。
- 在稀疏上采样（sparsely upcycled）FFN 专家中，共激活专家在部分 value 位置高度对齐；移除这些可复用位置后，专家偏好发生显著改变，且恢复原输出所需的残差专家数随共享覆盖率提升而下降。
- 缺乏统一的计算预算分配机制：复用计算提取、内部内容选择与动态专家数量决策应是同一分配问题的有序阶段，而非三个平行可调旋钮。

## 核心贡献（创新点）
- **揭示共享与路由的内生依赖**：通过 key-value 通道分解证明共激活专家的复用响应集中在特定 value 位置，移除后路由偏好重构，确立“share first, route what remains”的有序原则，区别于将共享与路由并行的既有工作。
- **提出 UniF-MoE 统一框架**：以单一 token-依赖预算协调共享宽度、共享内容选择与残差专家数量，实现 intra-expert 与 inter-expert 稀疏性的联合控制，避免独立决策导致的重复计算或容量错配。
- **设计 Gram 正则化路由几何**：对 router embedding 施加 Gram 矩阵正则项并结合正交初始化，以低开销方式引导路由方向分离与归一化，促进专家角色多样化与重叠稀疏化。
- **跨模态高效验证**：在 DeiT-S/16 与 BERT-large 骨干上同步提升视觉域泛化与语言理解精度，并较 top-2 GMoE 减少 9.1% 激活参数、16.1% FLOPs，推理时间与内存分别降低 45.2% 与 52.7%。

## 方法详解
- **Blockwise Shared-Residual Partitioning**：每层包含 1 个共享专家 $E_{\mathrm{shr}}$ 与 $K$ 个残差专家 $\{E_1,\dots,E_K\}$，每个专家中间宽度 $H$ 划分为 $B$ 个对齐块，块宽 $M=H/B$。所有专家初始化为同一稠密 FFN 拷贝，保证同索引通道可比。
- **Token-Adaptive Shared Modeling**：
  - **共享需求分数**：在 router embedding 矩阵新增一列 $\mathbf{W}_{\mathrm{shr}}$，计算 $\alpha(\mathbf{x}) = \tau + (1-2\tau)\sigma(\mathbf{x}\mathbf{W}_{\mathrm{shr}})$，其中 $\tau=(B-1)/B^2$，$\alpha(\mathbf{x})\in(\tau, 1-\tau)$ 控制共享块数量与路径权重。
  - **共享块选择**：用每块 up-projection key 均值 $\mu_b$ 作为原型，计算优先级 $u_b(\mathbf{x})=\mathbf{x}\mu_b$，取 Top-$b(\mathbf{x})$ 组成共享索引集 $\mathcal{T}_{\mathrm{shr}}(\mathbf{x})$，其余为残差集 $\mathcal{T}_{\mathrm{res}}(\mathbf{x})$。
- **Cumulative Residual-Expert Routing**：残差需求 $\beta(\mathbf{x})=1-\alpha(\mathbf{x})$。将 router 输出的 softmax 亲和力 $s(\mathbf{x})$ 降序排列为 $p_i(\mathbf{x})$，激活满足 $\sum_{i=1}^n p_i(\mathbf{x})\geq\beta(\mathbf{x})$ 的最小前缀专家数 $k(\mathbf{x})$，实现动态专家数量。
- **Shared-Residual Output Merging**：输出 $\mathbf{y}=\alpha(\mathbf{x})E_{\mathrm{shr}}^{\mathcal{S}}(\mathbf{x})+\sum_{i=1}^{k(\mathbf{x})}p_i(\mathbf{x})E_{q_i(\mathbf{x})}^{\mathcal{R}}(\mathbf{x})$，共享路径执行一次复用计算，重复专家开销仅局限于残差块。
- **Training Objective**：$\mathcal{L}=\mathcal{L}_{\mathrm{task}}+\lambda_{\mathrm{div}}\mathcal{L}_{\mathrm{div}}$，其中 $\mathcal{L}_{\mathrm{div}}=\|(\mathbf{W}_g^\star)^\top\mathbf{W}_g^\star-\mathbf{I}_{K+1}\|_F$，配合正交初始化保持路由嵌入单位范数与方向分离。
- **计算复杂度**：Token 激活块数 $C_B(\mathbf{x})=b(\mathbf{x})+k(\mathbf{x})[B-b(\mathbf{x})]$，传统 top-k 为 $kB$，额外专家仅追加剩余块计算。

## 实验与结果
- **数据集与基线**：视觉使用 DomainBed（PACS, VLCS, OfficeHome, TerraIncognita, DomainNet），骨干 DeiT-S/16（$K=6,B=8$）；语言使用 GLUE（CoLA, MRPC, QNLI, MNLI, RTE），骨干 BERT-large（$K=16,B=16$）。基线涵盖 Dense、静态 MoE（GMoE, EMoE, EMoE-L）、域泛化方法（LFME, DMDA, PC-MoE）与动态 MoE（DynMoE, MASS）。
- **主要结果**：
  - **DomainBed 平均准确率 69.5%**，PACS 89.6%、VLCS 81.7%、DomainNet 49.4% 最优，OfficeHome 74.2% 并列第一，TerraIncognita 52.6% 略低于 LFME 的 53.4%。
  - **GLUE 五项任务全面领先**，平均 82.76%，超越所有固定 top-k 变体及其逐任务最优包络线。
- **效率对比（VLCS）**：相对 top-2 GMoE，激活参数减少 9.1%（21.01 vs 23.11 M），FLOPs 减少 16.1%（8.70 vs 1
