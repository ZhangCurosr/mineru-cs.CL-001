---
title: "Share First, Route What Remains:"
source: https://arxiv.org/pdf/2608.10392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:07:10"
field: "Mixture of Experts 高效计算与动态路由"
keywords: ["mixture-of-experts", "token-adaptive computation", "shared experts", "dynamic routing", "domain generalization", "efficient inference"]
innovations: ["揭示 FFN 内部 channel 对齐与路由条件的系统性依赖，提出'先共享再路由剩余'的统一三阶段原则", "UniF-MoE 通过单 token 级共享-剩余预算联合协调共享宽度、共享内容与剩余专家数", "Gram 正则化路由嵌入促进正交归一路由方向，降低专家 co-activation 达 62.9%"]
benchmarks: ["DomainBed", "GLUE"]
---

# 论文速读：Share First, Route What Remains: A Unified Framework for Token-Adaptive MoE Computation

## 一句话总结
本文提出 UniF-MoE，一个将共享专家建模、细粒度计算与动态路由统一为有序三阶段的 token 自适应 MoE 框架，核心原则是"先共享，再路由剩余"——通过 FFN 内部 key-value 通道的分解发现可复用内容与剩余专家需求之间存在系统性依赖关系，进而以单一 token 级预算协调三阶段决策，在保持性能的同时显著降低激活计算量与推理延迟。

## 研究问题与动机
- **决策独立性的盲点**：现有 MoE 研究沿三个互补方向推进（共享专家、细粒度专家、动态路由），但这些决策通常独立设计，忽略了关键依赖：提取可复用计算会改变剩余内容、最优专家选择也会变化，独立的 router 可能重复执行已计算的共享工作。
- **FFN 内部的 channel 分解**：MoE 中的专家本质是 FFN 的 key-value 通道求和，稀疏扩容（sparse upcycling）初始化后专家间对齐位置可比较，使分析 channel 级复用成为可能。
- **三阶段排序而非并行调参**：共享建模、细粒度计算、动态路由是同一分配问题的三个阶段，而非三个平行旋钮；反转或割裂顺序会导致已修改的响应仍被重新路由。
- **诊断动机**：传统 top-k 路由回答"选哪些专家"却隐藏了"共享多少"和"专家数可变"两个决策，导致简单 token 冗余计算、困难 token 容量不足。

## 核心贡献（创新点）
- **揭示路由条件依赖性**：首次在稀疏扩容 FFN 中证明共激活专家的 value 位置对齐度与路由条件强相关（Pearson 0.697），提取对齐位置后 router rank 仅 26.9% 保持不变，且共享覆盖率与剩余专家需求负相关（−0.673），将共享建模、细粒度计算、动态路由统一为一个有序原则。
- **UniF-MoE 统一框架**：通过共享带宽（shared width）、共享内容（shared content）、剩余专家数（residual expert count）的单 token 级联合预算实现三阶段协调，而非独立决策，耦合 intra-expert 与 inter-expert 稀疏性。
- **累积路由质量机制（cumulative routing mass）**：用 β(x) = 1 − α(x) 定义剩余需求，激活最小前缀使累积亲和力 ≥ β(x)，将专家数量直接绑定于前序共享分配。
- **Gram 正则化路由嵌入**：初始化并对 Router 嵌入施加 Gram 约束 L_div = ||(W_g*)^T W_g* − I||_F，使路由方向正交归一，促进专家角色多样性和稀疏重叠。
- **跨模态高效验证**：在 DomainBed（5 视觉域）和 GLUE（5 语言任务）上全面优于静态和动态 MoE 基线，同时降低推理延迟与显存占用，并开源代码。

## 方法详解
**块级共享-剩余分区（Blockwise Shared-Residual Partitioning）**：每层含 1 个共享专家 E_shr 和 K 个剩余专家，每个 FFN 中间层宽 H 被划分为 B 个对齐块（每块宽 M = H/B），共享执行选定块子集，剩余专家执行补集块。

**Token-Adaptive 共享建模**：
- 共享需求分数：α(x) = τ + (1 − 2τ)σ(xW_shr)，其中 τ = (B−1)/B²，控制共享块数 b(x) = round(B·α(x))，保证每 token 至少 1 个共享块和 1 个剩余块。
- 共享块选择：μ_b = mean(K_shr block b)，u_b(x) = xμ_b，按优先级选 Top-B 块为共享集合 T_shr(x)，其余为 T_res(x)。

**累积剩余专家路由**：
- 剩余需求 β(x) = 1 − α(x)，与共享需求共享单预算。
- 按亲和力降序排列专家 p₁ ≥ p₂ ≥ … ≥ p_K，激活满足 Σ_{i=1}^{n} p_i(x) ≥ β(x) 的最小前缀长度 k(x)。
- 最终输出：y = α(x)·E_shr^S(x) + Σ_{i=1}^{k(x)} p_i(x)·E_{q_i(x)}^R(x)。

**训练目标**：L = L_task + λ_div·|| (W_g*)^T W_g* − I ||_F，W_g* = [W_shr, W_g] 正交初始化，λ_div 控制路由方向分离程度。

**计算复杂度**：每 token 激活 C_B(x) = b(x) + k(x)·[B − b(x)] 个块，额外专家只需执行剩余 B − b(x) 个块而非完整 FFN。

## 实验与结果
- **视觉**：DomainBed 五数据集（PACS/VLCS/OfficeHome/TerraIncognita/DomainNet），DeiT-S/16  backbone。UniF-MoE 平均准确率 **69.5%**，领先第二（LFME 68.5% / PC-MoE 68.4%）；PACS 89.6%、VLCS 81.7%、DomainNet 49.4% 均为最佳。
- **语言**：GLUE 五任务（BERT-large backbone，K=16, B=16）。UniF-MoE 平均 **82.76%**，超越所有固定 top-k（top-4 最佳 81.69%，oracle 6-task 最佳 81.95%）、DynMoE（81.64%）和 MASS（82.19%）；CoLA 66.83%、MRPC 91.57%、QNLI 93.10%、RTE 75.47% 均为最佳。
- **效率**（VLCS 上）：相对 top-2 GMoE，参数激活量低 **9.1%**，FLOPs 低 **16.1%**，推理时间低 **45.2%**，推理内存低 **52.7%**；I-TPS 仅 0.17s vs GMoE 0.31s、DynMoE 1.06s。
- **消融**：逐阶段替换为非自适应版本均显著降低精度并增加计算；固定 α 最破坏性；Prefix 选择损失 token 特异性共享内容；Top-2 剩余路由忽略剩余需求变化。
- **Gram 正则**：λ_div = 0.01 为最优，使专家对 co-activation 降低 62.9%，路由嵌入接近正交。
- **块数 B**：B=8/16 表现最佳；B=4 过粗，B=32 原型过于局部化。

## 相关工作脉络
- **DeepSeekMoE / Union-of-Experts**：构建显式共享专家，但不通过 token 级共享分配来定义 specialization 的处理对象；UniF-MoE 使共享-剩余依赖显式化，先共享再路由。
- **Emergent MoE / EMoE / MoSE**：关注 FFN 内部的细粒度或嵌套结构，但独立于动态路由决策；UniF-MoE 通过共享预算将两者串联。
- **DynMoE / MASS / Alloc-MoE**：动态调节激活专家数量，但独立于共享建模；UniF-MoE 用累积路由质量将专家数绑定到剩余需求。
- **orthogonality/variance 专家 specialization 方法**：通过正交/方差正则降低重叠，不驱动 token 级共享分配；UniF-MoE 的 Gram 约束服务于路由几何而非强制输出正交。
- **Sparse Upcycling (Komatsuzaki et al.)**：从预训练 FFN 复制初始化专家，是本文方法的前提条件（块对齐），UniF-MoE 在此基础上引入分层共享-剩余分解。

## 局限性与未来方向
- **块粒度敏感性**：B 过小（4）过于粗糙，过大（32）原型过于局部，需权衡选择；在高维大模型中 Block 数可能需更大规模搜索。
- **依赖稀疏扩容对齐假设**：方法建立在所有专家从同一预训练 FFN 复制的基础上，对随机初始化或非对齐专家的泛化能力未验证。
- **仅验证至 DeiT-S / BERT-large 规模**：未在大语言模型（如 DeepSeek、LLaMA 系列）上验证，Scaling Law 行为未知。
- **Gram 正则超参需调优**：λ_div 过小则方向相关，过大则压制 task loss，需要更自适应的正则策略。
- **计算开销**：块均值聚合和累积路由带来额外 overhead，在极低资源场景下可能抵消部分收益。

## 研究启发与可借鉴点
- **有序分解原则可迁移**："先共享再路由剩余"的 ordering 思想可推广至 Attention 模块的 KV 缓存复用、多层之间的特征复用等场景，作为统一框架的设计范式。
- **累积路由质量机制**：用 Σp_i ≥ β 的最小前缀确定专家数量，比 Top-k / threshold 更直接地绑定计算预算，可借鉴到任何需要 token 自适应专家数的场景。
- **Gram 正则路由嵌入**：简单有效的正交归一化约束，以极低代价提升路由多样性和稀疏重叠，可作为通用路由模块的即插即用组件。
- **channel-level 诊断分析**：FFN 内部的 key-value channel 对齐度分析（cosine similarity + co-activation correlation）是揭示 MoE 内部行为的有效工具，可复用于其他 MoE 变体的机理分析。
- **团队结合机会**：与本团队在自适应计算、动态路由方向高度契合，可探索将 UniF-MoE 的共享-剩余预算分配到 Attention heads 或跨 Transformer 层复用，以及将累积路由扩展至多模态 MoE 场景。

## 关键术语表
- **Token-Adaptive Computation**：根据输入 token 的语义需求动态调整计算量的设计范式，区别于对所有 token 施加相同计算预算。
- **Sparse Upcycling**：将预训练 dense FFN 复制到多个 expert 中作为初始化，使不同 expert 在 channel 级别保持对齐关系。
- **Cumulative Routing Mass**：按亲和力降序累积 expert 权重，直至累积和覆盖剩余需求 β(x)，以此决定激活专家数量。
- **Gram Regularizer**：对 Router 嵌入矩阵施加 Gram 矩阵逼近单位阵的约束，促进路由方向的正交性和稀疏重叠。
- **Shared-Residual Budget**：将总计算预算 α(x) + β(x) = 1 分配到共享和剩余两个 pathway，实现联合优化而非独立分配。
- **Value Position Alignment**：专家 FFN 中 down-projection value 行向量之间的余弦相似度，用于估计可复用响应的 channel 级重叠。
- **Co-activation Frequency**：两个 expert 被 router 同时选中的 token 比例，与 value alignment 呈正相关（Pearson 0.697）。
- **Residual Expert Demand**：共享响应被移除后，为以给定精度恢复原始输出所需的最少剩余 expert 数量。

## 可复现要素
- **数据集**：DomainBed（PACS、VLCS、OfficeHome、TerraIncognita、DomainNet）+ GLUE（CoLA、MRPC、QNLI、MNLI、RTE），均已公开
- **代码**：开源，https://github.com/existence0420/UniF-MoE
- **权重**：论文未提及模型权重开源
- **关键超参**：
  - Vision：K=6（残差 expert 数），B=8（块数），d=384，H=1536，dropout=0.1，stochastic depth=0.1
  - Language：K=16，B=16，max_seq_len=128，batch_size=32，FP16 混合精度，AdamW，learning rate ∈ {2e-5, 3e-5, 5e-5}
  - λ_div = 0.01，τ = (B−1)/B²，ε = 0.05（诊断实验），γ_m = 10⁻⁶ · tr(R_m R_m^T)/m
