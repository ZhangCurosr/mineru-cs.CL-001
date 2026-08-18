---
title: "From-Sequence-to-Structure-Relational-Uncertainty-Propagatio"
source: https://arxiv.org/pdf/2608.16002v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:38:49"
field: "可信大语言模型"
keywords: ["Uncertainty Quantification", "LLM Agents", "Trajectory Analysis", "Graph-based Reasoning", "Risk Propagation"]
innovations: ["提出RUPA框架，将Agent执行轨迹建模为有向关系依赖图进行不确定性传播", "设计关系感知边权重学习机制，自动校准七类依赖关系的传播重要性", "通过outcome-blind校准实现轨迹级早期失败检测与下游Agent性能提升"]
benchmarks: ["τ-2", "Terminal-Bench-2", "GAIA"]
---

# 论文速读：From-Sequence-to-Structure-Relational-Uncertainty-Propagatio

## 一句话总结
本文提出了 RUPA（Relational Uncertainty Propagation for Agents），一种面向 LLM Agent 的轨迹级不确定性量化框架，通过将执行轨迹建模为有向关系依赖图并进行关系感知的不确定性传播，有效捕捉长程推理过程中风险的累积与传递，显著优于现有仅依赖局部信号或线性序列假设的不确定性估计方法。

## 研究问题与动机
- **核心问题**：如何在多步推理、工具调用与环境交互的长时域 Agent 执行中，准确量化执行风险、提前识别潜在失败。
- **现有方法的不足**：
  1. 传统 UQ 方法（预测熵、序列概率、口语化置信度）仅依赖当前输出的局部信号，无法捕捉跨步骤的错误累积，在 τ-2 等基准上 AUROC 接近随机水平（如 Airline 域仅 0.205）。
  2. 近期 Agent 专用方法（SAUP、Tracer、UProp）将轨迹视为线性序列，按时间距离或语义相似度聚合不确定性，忽视了步骤间固有的**关系型依赖结构**（如重复行为、反馈冲突、目标对齐等）。
  3. 实证分析表明，失败信号广泛分布于整个轨迹中，而非集中于最终步骤；高失败风险步骤伴随高重复率（0.981）和停滞率（0.883），说明错误通过依赖关系逐步传播。
- **关键洞察**：Agent 不确定性应被视为**轨迹级别属性**，其演化依赖于推理状态、工具调用与环境反馈之间的关系结构，而非孤立步级的置信度序列。

## 核心贡献（创新点）
1. **揭示了关系依赖是 Agent 不确定性演化的关键来源**：指出传统方法将轨迹建模为独立预测或线性序列的根本缺陷，提供了首个从结构依赖角度分析 Agent 失败信号的实证研究。
2. **提出 RUPA 轨迹级不确定性量化框架**：将 Agent 执行历史自动建模为有向依赖图，节点覆盖推理状态/工具调用/环境反馈，边编码七类关系类型，实现关系感知的不确定性传播。
3. **关系感知的边权重学习机制**：根据各依赖类型在轨迹间的结构变异度自动分配传播权重，使重要历史依赖 exert 更大影响，同时抑制无关分支的噪声。
4. **联合本地不确定性与传播历史不确定性的轨迹级估计**：通过公式(5) $R_t = \lambda_u U_t + \lambda_h H_t$ 融合当前步本地置信度与图传播累积风险，输出 trajectory-level 不确定性评分。
5. **全面的基准验证与下游应用**：在 τ-2、Terminal-Bench-2、GAIA 三个代表性 Agent 基准上，使用 6 个开源 LLM（26B–230B）验证了 RUPA 在不确定性估计质量、早期失败检测和不确定性引导的 Agent 执行中的优势。

## 方法详解
**RUPA 框架包含三个核心模块：**

1. **关系轨迹图构建（Relational Trajectory Graph Construction）**
   - 将执行前缀表示为有向图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，节点 $i \in \mathcal{V}$ 对应执行事件（用户指令、助手推理/动作、工具调用、环境观测）。
   - 在有限历史上下文内仅构建依赖边，定义**七类关系类型**：
     - Sequential（序列）：紧邻前驱步骤
     - Latest（最新）：最近的环境/用户指令
     - Repetition（重复）：高度相似的推理模式或工具使用
     - Progression（递进）：扩展现有解决步骤的推理
     - Parallel（并行）：同一任务下的替代推理分支
     - Feedback（反馈）：指示执行失败或不稳定输出的环境反馈
     - Goal Alignment（目标对齐）：当前步骤与原始任务目标的语义依赖
   - 边类型通过文本嵌入距离、词法线索匹配（如"next/therefore"检测 Progression，"alternative/instead"检测 Parallel，"error/timeout"检测 Feedback）确定。

2. **关系感知不确定性传播（Relation-aware Uncertainty Propagation）**
   - **本地不确定性 $U_t$**：助手节点采用预测熵；环境节点基于可观测交互信号（执行失败、空响应、冲突反馈）。
   - **边权重计算**（公式 1）：
     $$w_{it} = \rho_{\tau_{it}} \tilde{r}_{it} \delta^{\text{age}(i,t)-1}$$
     其中 $\rho_\tau$ 为关系可靠性系数（基于训练轨迹的归一化强度变异度），$\tilde{r}_{it}$ 为关系强度，$\delta$ 为时间衰减因子。
   - **目标对齐分数**（公式 2）：$Q_{it} = 1 - S(y_t, x)$，直接计算当前步骤与目标文本的相似度补。
   - **传播不确定性聚合**（公式 3）：
     $$G_t = \frac{\sum_{i \in \mathcal{N}(t)} w_{it}(P_i + Q_{it})}{\sum_{i \in \mathcal{N}(t)} w_{it} + \epsilon}$$
     其中 $P_i$ 为历史节点存储的传播不确定性，$\mathcal{N}(t)$ 为依赖邻居。
   - **指数衰减动量**（公式 4）：
     $$m_t = \frac{\sum_{k < t} \gamma^{t-k} P_k}{\sum_{k < t} \gamma^{t-k} + \epsilon}, \quad H_t = \eta_g G_t + \eta_m m_t$$
     保留长期执行趋势，使早期关键失误持续影响后续推理。
   - **最终不确定性估计**（公式 5）：
     $$R_t = \lambda_u U_t + \lambda_h H_t$$
     轨迹级不确定性通过对步级分数聚合得到。

3. **Outcome-blind 校准**：所有图相关超参数（关系权重、时间衰减、历史窗口）仅在未标记训练轨迹上确定，不利用测试标签。

## 实验与结果
- **数据集**：τ-2（对话决策）、Terminal-Bench-2（命令行软件工程）、GAIA（开放域复杂问题解决）。
- **模型**：6 个开源 LLM，覆盖 Qwen3.5-27B、Qwen3.6-35B-A3B、Gemma-4-26B-it、Gemma-4-31B-it、GPT-OSS-120B、MiniMax-M2.7。
- **基线**：Entropy、Seq-prob、SAUP、Tracer、UProp。
- **评估指标**：AUROC、AUPRC、F1（最优阈值）。
- **主要结果**：
  - **不确定性估计质量**：RUPA 在所有模型和基准上均取得最佳性能。例如 MiniMax-M2.7 平均 AUROC 从最强基线 0.694 提升至 0.718；Qwen3.5-27B 从 0.608 提升至 0.656；Gemma4-31B 从 0.842 提升至 0.861。
  - **早期失败检测**：Prefix-based 评估显示，RUPA 在仅观察少量步骤时即可显著区分成功/失败轨迹，优势在轨迹前 20%–50% 阶段最为明显。
  - **下游 Agent 性能提升**：不确定性引导的采样策略（选择最低不确定性候选动作）在 Terminal-Bench-2 上将 Qwen3.5-27B 准确率从 0.105（随机）提升至 0.213，在 GAIA 上也持续优于基线。
  - **图建模的补充信息价值**：Entropy 匹配分析表明，即使在局部熵值相近的轨迹子集中，RUPA 仍能将 AUROC 维持在约 0.85，显著高于传统方法的 ~0.5。
- **消融实验**：移除图建模导致 AUROC 从 0.718 降至 0.678，移除传播机制降至 0.689，随机图拓扑导致 0.681，验证了关系图结构和传播机制的核心贡献。

## 相关工作脉络
1. **传统 LLM 不确定性量化**（Entropy、Seq-prob、Verbalized、Sampling-based）：设计用于单轮生成，无法处理长时域 Agent 轨迹，本文实证表明其在 τ-2 上 AUROC 仅 0.205–0.485，接近随机。
2. **Agent 专用 UQ 方法（SAUP、Tracer、UProp）**：首次将不确定性估计扩展到完整 Agent 轨迹，但仍采用线性序列假设聚合历史信号，未能建模步骤间的关系依赖结构。
3. **关系/图结构在 LLM 中的应用**：Jelodar et al. (2026)、Sun et al. (2026) 探索了 Graph Learning 与 Agent 的结合，但聚焦于推理增强而非不确定性量化；本文首次将图结构显式用于风险传播建模。
4. **轨迹风险分析**（Kirchhof et al., 2025; Zhang et al., 2026b）：指出现有 Agent UQ 需要重新评估，提出 Selaur 等方法利用不确定性奖励，但未系统建模依赖传播机制。
5. **工具调用与错误恢复**：Toolformer (Schick et al., 2023)、WebArena (Zhou et al., 2024) 等工作关注 Agent 的工具使用能力，本文从不确定性视角补充了可靠性保障层面。

## 局限性与未来方向
- **关系类型的枚举依赖人工设计**：当前七类关系虽覆盖主要模式，但可能遗漏特定领域特有的依赖结构；可探索通过自监督学习自动发现关系类型。
- **嵌入模型依赖性**：边构建依赖 bge-m3 嵌入，在资源受限场景可能成为瓶颈；论文提及可用 token overlap 替代，但未系统评估。
- **仅评估开源模型**：所有实验基于 6 个开源 LLM，对闭源商业 Agent 系统（如 GPT-4o Agent、Claude Code）的泛化性有待验证。
- **线性传播假设**：公式(3)的加权聚合隐含了无环依赖假设，对存在循环引用的复杂轨迹（如多轮调试迭代）的处理未深入讨论。
- **评估基准有限**：仅覆盖对话、命令行、开放域三类任务，对代码生成、多模态 Agent 等场景的适用性未验证。

## 研究启发与可借鉴点
1. **轨迹图建模的通用性**：将 Agent 执行历史建模为有向依赖图的思想可迁移至其他需要风险追踪的场景（如多步推理链验证、RAG 检索路径可信度评估）。
2. **关系权重的统计学习策略**：基于归一化强度变异度自动校准关系可靠性的方法（公式 6）避免了人工调参，可在其他图结构化不确定性任务中复用。
3. **outcome-blind 校准范式**：所有超参数仅在不使用测试标签的训练轨迹上确定，保证了评估的公平性，这一设计值得在 UQ 研究中推广。
4. **Entropy 匹配分析协议**：通过控制局部置信度变量来分离图传播信号的独立贡献，提供了一种清晰的可解释性分析框架。
5. **与下游执行的闭环验证**：本文不仅评估 UQ 质量，还验证了其指导 Agent 采样选择的效果，为 UQ 研究的实用性评估提供了范式。

## 关键术语表
- **Uncertainty Quantification (UQ)**：不确定性量化，估计模型预测可靠性并支持风险决策的技术。
- **Trajectory-level UQ**：轨迹级不确定性量化，针对多步 Agent 执行序列整体风险评估，而非单步预测。
- **Relational Dependency Graph**：关系依赖图，将 Agent 执行历史建模为节点（执行事件）和边（依赖关系）的有向图结构。
- **Predictive Entropy**：预测熵，基于模型输出概率分布计算的本地不确定性度量。
- **Relation-aware Edge Weight**：关系感知边权重，根据依赖类型的结构重要性动态分配的传播系数。
- **Goal Alignment Score**：目标对齐分数，衡量当前推理步骤与原始任务目标的语义匹配程度。
- **Prefix-based Evaluation**：前缀评估，仅使用轨迹前半段信息预测最终成功/失败的早期风险检测方法。
- **Outcome-blind Calibration**：结果盲校准，在不调用最终标签的情况下确定模型超参数的校准策略。

## 可复现要素
- **数据集**：τ-2、Terminal-Bench-2、GAIA，均为公开基准。
- **代码/权重**：代码已开源，见 https://github.com/icip-cas/RUPA；未提及预训练权重。
- **关键超参**：可见窗口大小 5、时间衰减因子 δ=0.80、动量衰减 γ=0.75、图传播权重 ηg=0.70、动量权重 ηm=0.30、历史权重 λh=1.0、本地权重 λu=1.0；各关系类型强度详见 Table 6（Sequential: 0.35, Latest: 0.75, Repetition: 0.95, Feedback: 0.85, Progression: 0.65, Parallel: 0.45）。
- **嵌入模型**：bge-m3。
- **框架**：Harbor framework 用于轨迹执行与验证。
