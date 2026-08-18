---
title: "Unifying-Depth-and-Width-Pruning-for-LLMs-via-Binary-Knapsac"
source: https://arxiv.org/pdf/2608.12953v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:23:42"
field: "大语言模型压缩与高效推理"
keywords: ["LLM compression", "structured pruning", "knapsack optimization", "depth pruning", "width pruning", "model efficiency"]
innovations: ["将结构化剪枝建模为带互斥约束的0/1背包问题，提供条件最优的粗粒度组件选择", "提出CRAFT指标量化压缩预算遵从度，实现接近精确的目标压缩比", "双轴剪枝：背包驱动的粗粒度深度剪枝+GLU感知的细粒度宽度填充"]
benchmarks: ["WikiText-2", "LAMBADA", "PIQA", "PROST", "CommonsenseQA", "ARC-Easy", "ARC-Challenge", "MathQA", "OpenBookQA", "MedQA", "BLiMP", "BoolQ", "Winogrande", "CoQA", "TruthfulQA", "Winogender", "Moral Stories"]
---

# 论文速读：Unifying-Depth-and-Width-Pruning-for-LLMs-via-Binary-Knapsac

## 一句话总结
论文提出 SNIPER（Structured Knapsack-optimization-based Pruner），一种针对 LLM 的双轴结构化剪枝框架：先通过 0/1 背包动态规划在粗粒度组件上求解条件最优的深度剪枝配置，再在 MLP 上做细粒度宽度剪枝以精确填满压缩预算，显著改善现有方法对目标压缩比的遵从性并提升跨架构任务稳定性。

## 研究问题与动机
- 现有结构化剪枝依赖贪心/局部启发式决策，忽略层间组件依赖，导致短视选择与不可逆的次优配置。
- 深度剪枝操作粗粒度原子单元，难以精确贴合目标压缩比（实测偏差可达 33%）；宽度剪枝能细粒度压缩，但引发不规则张量形状，推理加速有限。
- 现有方法在压缩预算遵从性、跨任务稳定性与跨架构泛化方面存在显著“容量松弛（capacity slack）”。
- 校准数据分布与样本量对剪枝结果影响较大，现有方法在该维度上鲁棒性不足。

## 核心贡献（创新点）
- 提出 SNIPER 双轴剪枝框架：深度轴通过 0/1 背包求解粗粒度组件的条件最优选择，宽度轴对剩余 MLP 维度进行细粒度填充。与贪心剪枝的本质区别在于提供关于固定重要性估计下的条件最优保障。
- 引入 CRAFT（Compression Ratio Adherence Factor）量化预算遵从度；SNIPER 在多种预算下达到 CRAFT ≈ 0.98，而现有剪枝器最高偏差达 33%。
- 设计基于边际退化（logits L2 差）的迭代重要性估计，使得评分能反映多组件联合剪枝后的全局依赖，优于单层/单组件 leave-one-out 度量。
- 在 4 种现代架构（含稠密、推理专用、融合 MLP、MoE）与 18 项跨域任务上系统评测，SNIPER 在所有配置下的平均排名 mean rank = 1.25，显著优于 6 个 SOTA 剪枝基线。
- 证明 SNIPER 的重要性分数具有跨压缩比的迁移稳定性，允许一次估计后复用，显著降低重复计算成本。

## 方法详解
- **组件分解与互斥约束**：将每层 Transformer 分解为注意力块 A_l、FFN 块 M_l 和整层保留项 T_l；保留 T_l 等价于同时保留 A_l 与 M_l，因此三者构成互斥选择。
- **问题形式化（0/1 背包）**：目标是在参数预算 C 内最大化所选组件的重要性之和，将粗粒度剪枝映射到 0/1 Knapsack Problem。
- **动态规划求解与离散化**：状态 f(i, j) 表示前 i 个组件在容量 j 下的最大重要性；通过离散化因子 α 将时间复杂度从 Θ(N·C) 降至 Θ(N·⌊C/α⌋)，在保持高 CRAFT 的同时控制求解时长。
- **迭代重要性估计**：v(x_i) = ‖Z − Z_drop^(i)‖² − ‖Z − Z_retain^(i)‖²，其中 Z 为校准集 logits；该度量刻画在当前部分剪枝模型上移除 x_i 的边际性能退化，相比静态/单组件度量更能反映组件间非线性交互。
- **第二阶段（细粒度残差剪枝）**：利用第 l 层的重要性 u_l 通过 softmax 反向分配 MLP 剪枝比例 ρ_l，并在 GLU 风格 MLP（MLP(X) = (σ(XG^T) ⊙ (XU^T))D^T）中选择敏感度 Ω_j = |D_{j,:} · (G_{:,j} ⊙ U_{:,j})| 最低的第 j 列联合删除（U/G 列与 D 行同步）。
- **后剪恢复微调（RFT）**：使用 SlimOrca 数据集与 LoRA（rank=64，scaling=16，dropout=0.05，lr=2e-4，1 epoch）进行恢复，以缓解结构化移除带来的性能损失。

## 实验与结果
- **模型与压缩比**：LLaMA-3.1-8B-Instruct、Qwen3-8B、Phi-4-14B、GPT-OSS-20B；重点报告 CR=25% 与 CR=35%（含/不含 RFT）。
- **任务体系（18 项 / 5 域）**：生成（WikiText-2、LAMBADA）、世界理解（PIQA、PROST、CommonsenseQA）、领域知识（ARC-Easy/Challenge、MathQA、OpenBookQA、MedQA）、NLU/NLI（BLiMP、BoolQ、Winogrande、CoQA）、安全与对齐（TruthfulQA、Winogender、Moral Stories）。
- **主要结果**：SNIPER 在所有配置下 mean rank = 1.25；Phi-4-14B（35% CR + RFT）Avg RP 达 81.58%，较 ShortGPT 提升 4.4%、较 2SSP 提升 11.42%；GPT-OSS-20B（35% CR + RFT）Avg RP 达 86.99%，较 2SSP 高 2.31%、较 ShortGPT 高 16.48%。
- **预算遵从性**：SNIPER CRAFT ≈ 0.98；最弱基线 CRAFT 低至 0.58，实际压缩比偏离目标最高可达 33%。
- **跨任务稳定性**：SNIPER 的 Std RP 普遍低于基线，例如 Qwen3-8B 在 35% CR + RFT 下降至 15.90%，显著优于基线的 17–30%。

## 相关工作脉络
- **ShortGPT / SLEB / ReplaceMe（深度剪枝）**：以层为单位移除或替换，速度快但难以精确控制压缩比；SNIPER 以背包优化替代贪心选层，并用宽度阶段精确补足预算。
- **SliceGPT / LLM-Pruner（宽度剪枝）**：沿通道/嵌入降维，压缩精度高但引发不规则张量、硬件友好性差且无法处理 MoE 路由层；SNIPER 在粗粒度阶段保留张量规整性，仅在剩余容量内做有结构的宽度裁剪。
- **2SSP（双轴剪枝基线）**：同样采用两阶段策略，但第二阶段在 GPT-OSS 等架构上退化为无效操作；SNIPER 的两阶段协同更完整，跨架构通用性更强。
- **SparseGPT / Wanda / LLM Surgeon（非结构化/半结构化剪枝）**：依赖特殊硬件或 N:M 稀疏才能兑现理论加速；SNIPER 始终产出结构化权重，便于现成加速部署。
- **量化与蒸馏类方法**：不改变原始架构，难以带来等比例的推理加速；SNIPER 通过结构性移除直接降低 FLOPs 与内存占用。

## 局限性与未来方向
- 当前对 MoE 架构使用同质化剪枝信号，未引入专家特异性的非均匀重要性评估。
- 重要性估计（校准驱动）与细粒度预算分配尚缺理论最优性保证，仅有粗粒度阶段具备条件最优性。
- 动态规划阶段仍依赖校准数据的代表性；极端分布偏移下重要性估计可能退化。
- 仅验证了 dense / fused-MLP / MoE 四类架构，对更广泛的新型结构（如多令牌预测、超长上下文结构）泛化性待进一步检验。

## 研究启发与可借鉴点
- 将结构化剪枝建模为带互斥约束的 0/1 背包问题，可为其他模型压缩任务（如 Transformer 解码头、路由门控）提供可复用的全局优化视角。
- 迭代式边际重要性估计（对比 retain vs. drop 的 logits 差）值得推广至多头注意力、门控/路由组件的联合评估。
- 引入 CRAFT 作为预算遵从指标，可直接用于横向对比不同压缩框架的工程可用性。
- “粗粒度全局搜索 + 细粒度局部填充”的双轴范式可与低秩适配、块级量化等技术组合，形成更高层级的统一压缩管线。
- 重要性分数的跨压缩比迁移实验表明，一次性离线估计后可服务于多目标部署，这一摊销策略值得工程化沉淀。

## 关键术语表
- **结构化剪枝**：按完整架构单元（层、头、通道块）进行剔除，以产生硬件友好的稀疏模式。
- **深度剪枝**：沿层轴移除整个 Transformer 层或注意力块，粒度粗、加速明显但灵活度低。
- **宽度剪枝**：沿通道/嵌入轴削减神经元或特征维度，粒度细、形状可控，但易造成不规则张量。
- **CRAFT**：Compression Ratio Adherence Factor，衡量实际压缩比相对目标压缩比的贴合程度，越高越优。
- **0/1 背包优化**：在容量约束下选择若干物品以最大化总价值的经典 NP-hard 问题；本文将其用于粗粒度组件选择。
- **互斥组件选择**：T_l 与 {A_l, M_l} 不能同时被选中的约束，体现层级结构的语义依赖。
- **GLU 风格 MLP**：MLP(X) = (σ(XG^T) ⊙ (XU^T))D^T 的形式，要求 U/G 列与 D 行同步删除以维持结构一致性。
- **RFT（恢复微调）**：后剪枝阶段的轻量化微调（如 LoRA），用于补偿结构化移除引入的性能损失。

## 可复现要素
- **模型与数据集**：LLaMA-3.1-8B-Instruct、Qwen3-8B、Phi-4、GPT-OSS-20B；校准集 SlimOrca（50 样本），RFT 数据集 SlimOrca（2,000 样本）。
- **代码**：论文开源仓库为 github.com/parmanu-lcs2 / parmanu.lcs2.in。
- **关键超参**：离散化因子 α=32；LoRA rank=64，scaling=16，dropout=0.05，lr=2e-4，1 epoch，batch size=2；温度 τ=1（softmax 预算分配）；上下文长度校准 512 tokens，RFT 1024 tokens。
- **硬件与环境**：单卡 NVIDIA A100 80GB；基于 PyTorch 与 HuggingFace Transformers；评测使用 LM-Evaluation-Harness。
- **压缩配置**：主要报告 CR=25% 与 CR=35%，含/不含 RFT 的对比。
