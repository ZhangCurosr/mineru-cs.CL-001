---
title: "Share First, Route What Remains:"
source: https://arxiv.org/pdf/2608.10392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:07:39"
field: "高效稀疏神经网络架构"
keywords: ["mixture of experts", "token-adaptive computation", "shared experts", "dynamic routing", "domain generalization", "FFN decomposition"]
innovations: ["揭示共享建模-细粒度计算-动态路由之间的序贯依赖关系，提出'先共享、后路由'统一原则", "UniF-MoE：通过单一 token 依赖预算耦合共享宽度、共享内容和残差专家数的三阶段统一框架", "Gram 正则化塑造正交归一化路由几何，促进专家多样性并降低共激活率 62.9%"]
benchmarks: ["DomainBed", "GLUE"]
---

# 论文速读：Share First, Route What Remains:

## 一句话总结
本文提出 **UniF-MoE**，一种统一的 Token 自适应 MoE 计算框架，核心思想是先将 FFN 专家按 key-value 通道分解、识别并提取可复用共享计算，再对剩余残差内容执行动态专家路由，从而将共享建模、细粒度计算与动态路由统一为"先共享、后路由"的序贯决策流程。

## 研究问题与动机
- **原子化专家假设的浪费**：传统 top-k MoE 将完整专家视为计算原子单元，当多个专家对同一 token 产生冗余响应时仍重复计算，简单 token 亦被分配固定专家数，造成浪费。
- **独立机制之间的潜在冲突**：共享专家（提取共用知识）、模块/嵌套专家（内部粒度调节）、动态路由器（按 token 调整专家数）三条独立改进线各自优化，但未考虑三者间的依赖关系——一旦提取共享部分，剩余内容改变，最优专家选择和所需容量均会随之变化。
- **缺乏序贯协调的统一框架**：现有方法将共享决策、内容选择、专家数量决策作为平行旋钮独立控制，缺少一个以单一 token 级预算协调 intra-expert 与 inter-expert 稀疏性的统一视角。
- **经验动机：共激活专家存在可分离的共享子空间**：通过对 TerraIncognita 训练的 top-2 MoE 进行诊断分析，发现高频共激活专家对在上界高达 80% 的 value 位置上方向一致；提取这些位置后重新训练路由器，仅 5.7% 的 token 保留原有 top-2 集合，说明共享提取后专家偏好发生显著变化，且所需残差专家数随共享覆盖率上升而下降（相关系数 −0.673）。

## 核心贡献（创新点）
- **揭示路由条件依赖性，提炼统一序贯原则**：在稀疏上采样（sparsely upcycled）专家中发现共享响应与 token 特有响应之间存在路由条件依赖性，首次将共享建模、细粒度计算与动态路由整合为"先共享、后路由"的单一有序原则，而非三者并行调节。
- **提出 UniF-MoE 统一框架**：通过单一 token 依赖预算耦合三个自适应决策——共享宽度（α(x) 决定共享块数）、共享内容（key 原型选择具体块）、残差专家数（累积路由质量覆盖 β(x) = 1−α(x)），协调 intra-expert 与 inter-expert 稀疏性，而非独立决策。
- **Gram 正则化塑造简洁路由几何**：对路由器嵌入施加 Gram 约束（正交归一化），促进专家角色多样化和路由方向分离，实验显示平均成对共激活率下降 62.9%，同时保留有用专家协作。
- **跨视觉与语言基准的精度-效率统一提升**：在 DomainBed（5 数据集）和 GLUE（5 任务）上均取得最优结果，相对 top-2 GMoE 激活参数减少 9.1%、FLOPs 减少 16.1%、推理时间减少 45.2%、显存减少 52.7%，在所有对比 MoE 中实现最强的实测推理性能。

## 方法详解
- **Blockwise Shared-Residual Partitioning（分块共享-残差划分）**：每层包含 1 个共享专家 $E_{\mathrm{shr}}$ 和 K 个残差专家 $\{E_1, \ldots, E_K\}$，每个专家（中间宽度 H）划分为 B 个对齐块，每块宽度 $M=H/B$。所有专家从同一稠密 FFN 初始化，块边界在训练开始时对齐；每个 token 的共享专家执行一块子集，残差专家执行互补子集，同一隐藏位置不重复分配。
- **Token-Adaptive Shared Modeling（Token 自适应共享建模）**：
  - **共享需求分数**：在路由器嵌入矩阵中加入可学习列 $\mathbf{W}_{\mathrm{shr}}$，得到 $\alpha(\mathbf{x}) = \tau + (1-2\tau)\sigma(\mathbf{x}\mathbf{W}_{\mathrm{shr}})$，其中 $\tau=(B-1)/B^2$，$\alpha(\mathbf{x})$ 同时控制共享块数和共享路径混合权重，保证 $b(\mathbf{x}) \in \{1, \ldots, B-1\}$，即每个 token 至少使用 1 个共享块并留出 1 个残差块。
  - **共享块选择**：用共享专家每块的 up-projection key 均值 $\mu_b$ 作为原型，计算优先级 $u_b(\mathbf{x}) = \mathbf{x}\mu_b$，取 TopK 确定 $\mathcal{T}_{\mathrm{shr}}(\mathbf{x})$，其余为残差块集合 $\mathcal{T}_{\mathrm{res}}(\mathbf{x})$。
- **Cumulative Residual-Expert Routing（累积残差专家路由）**：残差需求 $\beta(\mathbf{x}) = 1 - \alpha(\mathbf{x})$；将原始路由分数排序为 $p_1 \geq p_2 \geq \cdots \geq p_K$，激活满足 $\sum_{i=1}^{k} p_i \geq \beta(\mathbf{x})$ 的最小前缀，即 $k(\mathbf{x}) = \min\{n : \sum_{i=1}^{n} p_i \geq \beta(\mathbf{x})\}$。残差专家数由共享分配后剩余需求决定，实现了顺序协调。
- **Shared-Residual Output Merging（共享-残差输出合并）**：最终输出 $\mathbf{y} = \alpha(\mathbf{x}) E_{\mathrm{shr}}^{S}(\mathbf{x}) + \sum_{i=1}^{k(\mathbf{x})} p_i(\mathbf{x}) E_{q_i(\mathbf{x})}^{\mathcal{R}}(\mathbf{x})$，共享路径加权 $\alpha(\mathbf{x})$，残差专家保留原始相似度权重；总系数质量满足 $1 \leq \alpha(\mathbf{x}) + P_{k(\mathbf{x})}(\mathbf{x}) < 1 + p_{k(\mathbf{x})}(\mathbf{x})$， overshoot 受控。
- **Training Objective（训练目标）**：损失 $\mathcal{L} = \mathcal{L}_{\mathrm{task}} + \lambda_{\mathrm{div}} \mathcal{L}_{\mathrm{div}}$，其中 $\mathcal{L}_{\mathrm{div}} = \|\left(\mathbf{W}_g^\star\right)^\top \mathbf{W}_g^\star - \mathbf{I}_{K+1}\|_F$，通过 Gram 正则化保持嵌入正交归一化；初始化为正交矩阵；实测每 token 激活块数 $C_B(\mathbf{x}) = b(\mathbf{x}) + k(\mathbf{x})[B - b(\mathbf{x})]$，当 $k=1$ 时恰好等于一个完整 FFN。

## 实验与结果
- **数据集与模型配置**：视觉骨干为 ImageNet 预训练 DeiT-S/16（$d=384$, $H=1536$），转换层 8、10，$K=6$, $B=8$；语言骨干为 BERT-large，转换层 20、22，$K=16$, $B=16$。
- **视觉基准（DomainBed）**：PACS、VLCS、OfficeHome、TerraIncognita、DomainNet。UniF-MoE 平均准确率 **69.5%**，领先 GMoE（67.9%）和 LFME（68.5%）；在 PACS（**89.6%**）、VLCS（**81.7%**）、DomainNet（**49.4%**）上刷新 SOTA，OfficeHome 与 GMoE 并列 74.2%，TerraIncognita 以 52.6% 仅次于 LFME（53.4%）。
- **语言基准（GLUE）**：CoLA、MRPC、QNLI、MNLI、RTE。UniF-MoE 平均 **82.76%**，超越所有固定 top-k 变体及动态 MoE（DynMoE 81.64%，MASS 82.19%）；CoLA（**66.83%**）、MRPC（**91.57%**）、QNLI（**93.10%**）、RTE（**75.47%**）均刷新，MNLI 为 86.84%（接近 fixed top-2 最佳 86.73%）。
- **效率对比（VLCS 实测）**：相对 top-2 GMoE，激活参数减少 **9.1%**、FLOPs 减少 **16.1%**、推理时间减少 **45.2%**、推理显存减少 **52.7%**；在所有 MoE 基线中取得最低推理时间和显存。
- **消融结论**：移除任一自适应决策（固定 α=0.4 / 固定 top-2 残差激活 / 前缀块选择）均导致精度下降和计算上升，表明三大阶段的协调增益来自序贯配合而非稀疏性叠加；λ_div=0.01 为最佳正则强度。

## 相关工作脉络
- **DeepSeekMoE / Union-of-Experts**：通过显式添加共享专家或虚拟共享层提取共用知识；差异在于 UniF-MoE 将共享提取与残差路由序贯协调于单一预算内，而非独立机制叠加。
- **Emergent MoE / MoSE / MoNE**：通过 key 质心分解或嵌套/可瘦身专家调节内部粒度；差异在于 UniF-MoE 的粒度选择（哪些块共享）由 token 自适应决定，且与后续专家数量联动，而非静态分割。
- **DynMoE / MASS / Alloc-MoE**：动态调节激活专家数量；差异在于 UniF-MoE 的专家数量由"共享后剩余需求"决定，而非独立估计 token 难度。
- **Orthogonality/Variance 正则（Guo et al., 2025）/ MP-MoE**：鼓励专家 specialization 以减少重叠；差异在于 UniF-MoE 的 Gram 正则作用于路由器嵌入而非专家输出本身，通过路由几何间接控制重叠，保留有益协作。
- **PC-MoE**：扰动 cosine 路由器统计特性以提升域泛化；差异在于 UniF-MoE 从 FFN 内部结构出发，先分离共享响应再动态路由，定位更底层。
- **Sparse Upcycling（Komatsuzaki et al., 2023）**：从预训练稠密 FFN 初始化稀疏 MoE；UniF-MoE 在此基础上引入块级共享-残差分解，是 upcycling 框架的自然扩展而非替代。

## 局限性与未来方向
- **诊断基于单模型层**：分析仅在 TerraIncognita 训练的 top-2 MoE 的第 10 层进行，观察到的相关性（如共享比与残差需求的相关系数 −0.673）是否在不同层深度、不同模型规模下保持稳健，有待验证。
- **TerraIncognita 上略逊于 LFME**：LFME 在该数据集上以 53.4% 超过 UniF-MoE 的 52.6%，提示在强域特定背景（位置/相机视角）场景下，显式域专业化仍具优势，共享-残差框架对此类场景的处理可能不足。
- **块粒度 B 的选择存在权衡**：消融显示 B=8 或 16 表现最佳，过粗（B=4）或过细（B=32）均损害性能，但对不同任务/模型规模的最优 B 仍需经验调参。
- **未讨论极端共享情况**：当 α(x)→1 时几乎全部计算走共享路径，残差路由退化为最小规格，此时是否会损失 token 特有信息、以及如何防止共享专家成为"万能瓶颈"，论文未深入分析。

## 研究启发与可借鉴点
- **序贯统一而非并行叠加的设计哲学**：将三个独立研究方向（共享、粒度、动态路由）整合为单一预算驱动的序贯决策链，这一思路可迁移至其他稀疏计算场景（如稀疏注意力、条件计算），避免多机制叠加时的资源浪费与决策冲突。
- **基于通道分解的诊断方法**：将 FFN 表达为 key-value 通道之和、用余弦相似度度量共激活对齐、用 ridge-stabilized least squares 量化残差需求——这套分析工具可复用于诊断其他 MoE 变体的内部结构。
- **Gram 正则化的简洁路由几何**：正交归一化约束使路由嵌入指向不同方向，兼具分离专家角色与保持归一化尺度的双重效果，实现成本极低（$\lambda_{\mathrm{div}}=0.01$ 即有效），值得在其他稀疏路由模型中尝试。
- **Token 自适应联合预算的设计**：$\alpha(\mathbf{x}) + \beta(\mathbf{x}) = 1$ 的单一预算约束天然协调共享与残差，避免了分别估计两者重要性时的不一致，可推广至其他需要多阶段资源分配的任务。
- **结合团队方向的机会**：团队若关注低资源场景或边缘推理部署，UniF-MoE 的实测推理时间和显存大幅降低（45%/53%）特性可直接适配；同时其 token 自适应特性可用于动态推理（early-exit）或跨模态统一架构的构建。

## 关键术语表
- **Token-Adaptive MoE**：根据每个 token 的语义需求自适应调整共享计算量与激活专家数的 MoE 变体，而非对所有 token 使用固定专家数。
- **Sparsely Upcycled Expert**：从预训练稠密 FFN 初始化、通过稀疏化得到的高维专家，保持通道对齐使跨专家比较可行。
- **Key-Value Channel Decomposition**：将 FFN 输出按 up-projection key 和 down-projection value 配对的隐藏通道展开，每个通道独立贡献一个向量响应。
- **Shared-Demand Score α(x)**：由共享专家路由嵌入计算得出的标量，控制共享块数量和共享路径的混合权重，取值介于 τ 与 1−τ 之间。
- **Cumulative Routing Mass**：按路由分数降序累加专家亲和度，直到累计质量覆盖残差需求 β(x)，由此决定激活的残差专家数 k(x)。
- **Gram Regularizer**：对路由器嵌入矩阵施加 Frobenius 范数正则，使其偏离单位矩阵的程度最小化，从而促进嵌入正交归一化和路由方向分离。
- **DomainBed**：由 PACS、VLCS、OfficeHome、TerraIncognita、DomainNet 五个域泛化数据集组成的综合评测基准。
- **Intra-Expert / Inter-Expert Sparsity**：前者指专家内部部分通道（块）不参与计算，后者指仅激活部分专家，UniF-MoE 通过统一预算联合协调两者。

## 可复现要素
- **数据集**：DomainBed（5 个公开数据集：PACS、VLCS、OfficeHome、TerraIncognita、DomainNet）；GLUE（5 个公开 NLU 任务）。均公开可用。
- **代码**：开源，地址 https://github.com/existence0420/UniF-MoE。
- **权重**：论文未提及独立权重发布，基于 ImageNet 预训练 DeiT-S/16 和 BERT-large-cased 微调。
- **关键超参**：视觉 $K=6$, $B=8$；语言 $K=16$, $B=16$；$\lambda_{\mathrm{div}}=0.01$；hidden dropout 0.1；stochastic-depth 0.1；训练最多 10 epoch，FP16 混合精度，AdamW 优化器，learning rate 取自 $\{2\times10^{-5}, 3\times10^{-5}, 5\times10^{-5}\}$；GPU 为 NVIDIA RTX 3090。
- **实验环境**：Python 3.8.20, PyTorch 2.4.1, CUDA 12.1, cuDNN 9.1, Transformers 4.46.3。
