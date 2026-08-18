---
title: "Unifying-Depth-and-Width-Pruning-for-LLMs-via-Binary-Knapsac"
source: https://arxiv.org/pdf/2608.12953v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:23:47"
field: "大语言模型压缩与高效推理"
keywords: ["structured pruning", "LLM compression", "knapsack optimization", "dual-axis pruning", "depth pruning", "width pruning", "model efficiency"]
innovations: ["提出SNIPER双轴剪枝框架，用0/1背包DP替代贪心实现条件最优深度剪枝", "引入CRAFT度量量化预算遵从度，解决现有方法33%压缩比偏差问题", "设计迭代边际重要性估计与GLU-MLP敏感列剪枝启发式，兼顾精度与推理加速"]
benchmarks: ["WikiText-2", "LAMBADA", "PIQA", "PROST", "CommonsenseQA", "ARC-Easy", "ARC-Challenge", "MathQA", "OpenBookQA", "MedQA", "BLiMP", "BoolQ", "Winogrande", "CoQA", "TruthfulQA", "Winogender", "Moral Stories"]
---

# 论文速读：Unifying-Depth-and-Width-Pruning-for-LLMs-via-Binary-Knapsac

## 一句话总结
本文提出 SNIPER，一个基于 0/1 背包动态规划的两阶段结构化剪枝框架，首先在深度轴上对粗粒度组件进行条件最优选择，再通过宽度轴微调精确命中目标压缩比，在 4 种架构、18 个任务上均优于 6 个 SOTA 基线，平均排名 1.25，预算遵从度 CRAFT=0.98。

## 研究问题与动机
- **剪枝粒度与贪心优化之间的矛盾**：现有结构化剪枝方法（深度剪枝或宽度剪枝）依赖局部贪心启发式，忽视层间/组件间的依赖关系，导致不可逆的次优决策。
- **目标压缩比严重偏离**：现有方法存在"容量松弛（capacity slack）"，实际压缩比偏离目标值高达 33%，难以在内存受限的部署环境中可靠使用。
- **任务级性能不稳定**：贪心剪枝往往过度优化孤立指标（如困惑度），导致分布外任务出现高达 30% 的性能偏离和较高的跨任务方差。
- **深度/宽度剪枝难以兼顾**：纯深度剪枝精度不足，纯宽度剪枝产生不规则张量形状导致实际推理加速有限，缺乏统一双轴框架。

## 核心贡献（创新点）
1. **SNIPER 双轴结构化剪枝框架**：先通过 0/1 背包 DP 在深度轴上做条件最优粗剪枝，再做宽度轴细剪枝填补剩余预算；与现有工作本质区别在于用动态规划替代贪心，保证相对于固定重要性估计的条件最优性。
2. **迭代边际重要性估计**：基于部分剪枝模型上移除某组件导致的 logit 发散度（Eq. 3）迭代计算组件重要性，而非孤立评估单个参数；与 SLEB 等 leave-one-out 方法相比，能捕获全局组件间非线性依赖。
3. **CRAFT（Compression Ratio Adherence Factor）度量**：首次系统量化剪枝方法对目标压缩比的遵从度；现有方法 CRAFT 低至 0.58，SNIPER 达 0.98，解决了部署可信性问题。
4. **GLU-MLP 列剪枝启发式**：针对现代 LLM 的 Gated Linear Unit MLP 结构，定义列敏感度 Ω_j = |D_{j,:} · (G_{:,j} ⊙ U_{:,j})| 选择待剪列，比 magnitude/activation-aware 启发式平均高 2–4%、任务方差低 13%。
5. **可迁移的重要性分数**：证明 SNIPER 的重要性分数具有跨压缩比的可迁移性（35%→25% 仅降 0.17% Avg RP），可 amortize 最耗时的估计步骤。

## 方法详解
**Stage 1：粗粒度深度剪枝（背包 DP）**
- 将模型分解为组件集合 X = ∪_l {A_l, M_l} ∪ {T_l}，其中 T_l 为完整保留第 l 层，A_l/M_l 为单独保留 Attention/MLP，互斥约束要求 T_l 与 {A_l, M_l} 不可同时选。
- 目标：max Σ v(x), s.t. Σ w(x) ≤ C（Eq. 1），映射为 0/1 背包问题。
- DP 递推：f(i,j) = max(f(i−1,j), f(i−1, j−w(x_i)) + v(x_i))，对互斥组作为单步多分支处理。
- 引入离散化因子 α 将复杂度从 Θ(N·C) 降至 Θ(N·⌊C/α⌋)，论文取 α=32 兼顾效率与精度。
- 回溯得到最优组件集 S*。

**Stage 2：细粒度宽度剪枝（MLP 列剪枝）**
- 剩余预算 ΔC = C − Σ_{x∈S*} w(x) 需进一步剪 MLP 列。
- 按层重要性 u_l = v(T_l) 用 softmax 负指数分配剪枝比例：ρ_l = exp(−u_l/τ) / Σ_k exp(−u_k/τ)。
- 对每层剪 N_l = ⌊ρ_l · N_total⌋ 列，列敏感度 Ω_j = |D_{j,:} · (G_{:,j} ⊙ U_{:,j})|，选取最小 Ω_j 的列删除（需同步删除 U/G 的第 j 列和 D 的第 j 行以维持 GLU 结构一致性）。

**迭代重要性估计（Section 3.3）**
- v(x_i) = ||Z − Z_drop^{(i)}||² − ||Z − Z_retain^{(i)}||²，基于 calibration batch 的 logit 二范数差衡量移除/保留 x_i 对预测保真度的边际影响。
- 与静态重要性不同，该估计在"由 DP 决定的当前部分剪枝模型 M^{(i−1)}"上进行，捕获组件间联合效应。

## 实验与结果
**评测设置**
- 模型：LLaMA-3.1-8B-Instruct、Qwen3-8B、Phi-4-14B（fused-MLP）、GPT-OSS-20B（MoE）。
- 基线：ReplaceMe、SLEB、ShortGPT、SliceGPT、LLM-Pruner、2SSP（共 6 个）。
- 任务：18 个任务覆盖 5 个领域（生成、世界理解、领域知识、NLU/NLI、安全/伦理）。
- 压缩比：25%、35%（部分实验含 50%），均含/不含 RFT（Recovery Fine-Tuning，LoRA rank=64）。

**主要结果**
- **平均排名**：SNIPER mean rank = **1.25**（最优），所有基线均有至少一处配置性能严重退化，SNIPER 最差排名仅 3。
- **LLaMA-3.1-8B @ 25% CR+RFT**：SNIPER Avg RP = **87.34%**，显著优于 2SSP（83.19%）、SLEB（78.45%）等；Std RP = **10.00%** 最低。
- **Qwen3-8B @ 25% CR+RFT**：SNIPER Avg RP = **82.64%**，Std RP = **12.20%**；相对 ShortGPT 任务方差低约 11.45%。
- **Phi-4-14B @ 35% CR+RFT**：SNIPER Avg RP = **81.58%**，超 ShortGPT（77.18%）4.4%、超 2SSP（70.16%）11.42%。
- **GPT-OSS-20B @ 35% CR+RFT**：SNIPER Avg RP = **86.99%**，超 2SSP（84.68%）2.31%、超 ShortGPT（70.51%）16.48%。
- **CRAFT**：SNIPER = **0.98**（各目标 CR 下均稳定），基线最低仅 0.58（LLM-Pruner @ 50%）。
- **裁剪效率**：SNIPER 总剪枝时间仅比 SLEB 多约 4 分钟，其中 96% 耗时在重要性估计，实际剪枝步骤仅 56 秒；重要性分数可跨压缩比复用（ amortization ）。

## 相关工作脉络
1. **深度剪枝（ReplaceMe/ShortGPT/SLEB）**：这些方法逐层贪心评估冗余度后删除整层，无法保证全局最优且存在目标 CR 偏离；SNIPER 用背包 DP 保证条件最优，并引入互斥约束处理 T_l 与 {A_l, M_l} 的选择。
2. **宽度剪枝（SliceGPT/LLM-Pruner）**：基于 PCA 投影或依赖图剪神经元，产生不规则张量形状，难以获得实际推理加速；SNIPER 仅在 Stage 2 对 MLP 列做结构化删除，保持硬件友好形状。
3. **双轴剪枝（2SSP）**：同样采用两阶段策略，但以 attention block 为唯一粗粒度单元；SNIPER 同时考虑 Attention 和 MLP 块，并在 Stage 1 使用 DP 而非贪心，且 CRAFT 明显更优（0.98 vs ~0.85）。
4. **未结构化剪枝（SparseGPT/Wanda）**：达到高稀疏率但需特殊硬件支持，实际延迟收益不确定；SNIPER 输出规则结构化稀疏，直接适配现有加速器。
5. **重要性估计（magnitude/activation-aware）**：SNIPER-MAG 和 SNIPER-ACT ablation 表明二者分别比 SNIPER 低 2–4% Avg RP 且 Std RP 高 13%，验证了基于 DP 配置迭代边际贡献评估的优越性。
6. **MoE 剪枝**：论文承认当前 SNIPER 对 MoE 架构采用同质信号，未来需引入 expert-specific 非均匀信号——这提示了与本团队 MoE 压缩方向的结合点。

## 局限性与未来方向
- **MoE 架构的同质剪枝信号**：当前对 GPT-OSS-20B（MoE）采用与 dense 模型相同的importance估计，未针对 expert 路由层设计专用信号，可能限制剪枝效果。
- **理论保证的边界**：目前仅有 Stage 1 的条件最优性保证（针对固定重要性估计），Stage 2 的列剪枝和重要性估计本身缺乏类似的理论最优性保证。
- **Calibration 数据依赖**：虽证明对 calibration 分布变化鲁棒，但仍需少量校准样本（论文使用 50 条 SlimOrca）计算重要性。
- **DP 复杂度与超大模型**：当模型参数规模极大时，即使离散化后 DP 的可行解空间仍可能受限；论文未讨论在百 B 级以上模型上的可扩展性。
- **RFT 依赖**：部分配置（如 35% CR）性能优势需配合 RFT 才显著，无 RFT 时的增益相对减弱。

## 研究启发与可借鉴点
1. **背包 DP 迁移至其他模型压缩任务**：任何涉及"从离散组件集中选择子集以优化某目标+容量约束"的场景（如 LoRA adapter 选择、MoE expert 裁剪、网络架构搜索）均可借鉴此框架。
2. **CRAFT 可作为剪枝方法的标准评测指标**：建议后续工作中将 CRAFT 纳入 benchmark 报告体系，推动方法向"预算精确满足"方向演进，避免只报告 Avg RP 忽略预算偏差。
3. **迭代边际重要性估计范式**：在组件级评估重要性时，基于"当前部分剪枝状态"而非原始完整模型计算边际贡献，能更好捕获组件间交互效应，该方法可推广至 Transformer 子模块（如 layer norm、residual 分支）的剪枝。
4. **重要性分数跨任务/跨预算可迁移性**：本研究证明重要性分数可跨压缩比复用且性能损失微小，这一发现启示可构建"一次估计、多次剪枝"的实用流水线，大幅降低部署成本；可进一步探索跨架构迁移（如从 LLaMA 到 Qwen 的分数迁移）。
5. **GLU-MLP 敏感列剪枝启发式**：Ω_j = |D_{j,:} · (G_{:,j} ⊙ U_{:,j})| 同时利用三个投影矩阵的乘积效应，比单一 magnitude 或 activation 更能反映真实贡献；该设计可直接应用于所有使用 GLU 的现代 LLM（Qwen、Phi、LLaMA 系列）。

## 关键术语表
**SNIPER**：Structured Knapsack-optimization-based Pruner，本文提出的双轴结构化剪枝框架，Stage 1 用 0/1 背包 DP 做深度粗剪，Stage 2 用 MLP 列剪枝做宽度精剪。
**CRAFT（Compression Ratio Adherence Factor）**：压缩比遵从因子，定义为实际压缩比/目标压缩比，用于量化剪枝方法对目标预算的精确满足程度，SNIPER 达 0.98。
**条件最优（Conditional Optimality）**：SNIPER 在固定组件重要性估计下通过 DP 求得的最优解，虽不如全局最优强，但优于任何贪心方法（后者无任何最优性保证）。
**离散化因子 α**：将组件参数量 discretize 的系数，用于降低背包 DP 的时间复杂度（从 Θ(N·C) 降至 Θ(N·⌊C/α⌋)），论文取 α=32。
**迭代边际重要性估计**：在部分剪枝模型 M^{(i−1)} 上评估组件 x_i 被移除时的 logit 散度变化，以此作为 v(x_i)，能捕获组件间的非线性交互。
**GLU-MLP**：Gated Linear Unit MLP，现代 LLM 的主流 MLP 变体，计算为 (σ(XG^T) ⊙ (XU^T))D^T；SNIPER 的 Stage 2 针对此结构进行列剪枝。
**Avg RP / Std RP**：平均性能保留率（pruned/base）和跨任务性能保留率标准差，前者衡量总体效果，后者衡量任务级稳定性。
**RFT（Recovery Fine-Tuning）**：剪枝后的恢复微调，本文使用 LoRA（rank=64）在 SlimOrca 上微调 1 epoch。

## 可复现要素
- **数据集**：SlimOrca（校准+RFT，公开）、Alpaca（校准分布消融，公开）、C4（校准分布消融，公开）；评测任务 WikiText-2、LAMBADA、PIQA、PROST、CommonsenseQA、ARC-Easy/Challenge、MathQA、OpenBookQA、MedQA、BLiMP、BoolQ、Winogrande、CoQA、TruthfulQA、Winogender、Moral Stories（均为公开数据集）。
- **代码/权重**：论文 GitHub 为 github.com/parmanu-lcs2/parmanu.lcs2.in（具体仓库名需核实）；基线模型 LLaMA-3.1-8B-Instruct、Qwen3-8B、Phi-4、GPT-OSS-20B 为公开权重。
- **关键超参**：离散化因子 α=32；temperature τ=1；RFT 用 LoRA rank=64、scaling=16、dropout=0.05、lr=2e−4、batch_size=2、1 epoch；calibration 样本数 50、context_length=512；RFT context_length=1024、2000 samples。
- **硬件**：单卡 NVIDIA A100 80GB VRAM。
- **框架**：PyTorch + HuggingFace Transformers + lm-evaluation-harness。
