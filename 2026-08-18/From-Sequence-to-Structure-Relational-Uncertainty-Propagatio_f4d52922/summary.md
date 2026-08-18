---
title: "From-Sequence-to-Structure-Relational-Uncertainty-Propagatio"
source: https://arxiv.org/pdf/2608.16002v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:39:29"
---

# 论文速读：From-Sequence-to-Structure-Relational-Uncertainty-Propagatio

## 一句话总结
论文针对现有LLM Agent不确定性量化(UQ)方法仅依赖局部token概率或线性序列、无法刻画长程依赖累积风险的缺陷，提出RUPA框架；该方法将Agent执行轨迹自动建模为含7类语义/反馈依赖边的有向图，通过关系感知的不确定性传播机制聚合历史执行风险，并结合局部置信度输出轨迹级不确定性估计，在τ-2、Terminal-Bench-2与GAIA三大基准上持续优于现有UQ基线。

## 研究问题与动机
- **核心问题**：LLM Agent在长Horizon多步推理与工具交互中，失败风险往往源于早期中间步骤的错误沿依赖路径逐步累积与放大，而非单步预测的孤立失误。
- **现有方法不足1**：传统UQ（Predictive Entropy、Sequence Probability、Verbalized Confidence）仅评估当前生成的局部置信度，完全忽略执行历史、工具调用与环境反馈中的风险传递，作者在τ-2上测得Airline域Seq-prob AUROC仅0.205，接近随机猜测。
- **现有方法不足2**：近期Agent专用UQ方法（SAUP、Tracer、UProp）虽引入历史上下文，但普遍将轨迹视为线性序列按时间距离或语义相似度聚合，无法刻画推理状态、工具调用、错误反馈之间的非时序、多跳依赖结构。
- **实证动机**：故障轨迹的结构化分析显示，高风险步骤均匀分布于执行全段，且伴随高频的重复行为(Repetition, 0.981)、停滞(Stagnation, 0.883)与反馈冲突特征，印证“关系型依赖结构”是风险演化的关键驱动，线性建模存在根本局限。

## 核心贡献（创新点）
1. **揭示Agent不确定性的关系演化本质**：明确指出执行失败风险源于推理节点间的拓扑依赖而非独立序列，为Agent UQ提供了超越单步置信度的分析视角。
   与已有工作的本质区别：不同于将Agent轨迹简化为时间序列或独立token级预测的传统做法，本文首次将“关系型依赖结构”作为不确定性累积的核心驱动因素进行系统性实证与建模。
2. **提出RUPA图结构不确定性量化框架**：将Agent执行前缀自动转换为包含7类依赖边（Sequential/Latest/Repetition/Progression/Parallel/Feedback/Goal Alignment）的有向轨迹图。
   与已有工作的本质区别：相比SAUP/Tracer等仅做序列聚合或点对点传播的方法，RUPA显式建模了多跳、非连续、带语义/反馈属性的图拓扑，使风险能沿真实执行路径流转而非机械时间衰减。
3. **设计关系感知的不确定性传播与动量融合机制**：通过关系可靠性、强度与时序衰减自适应计算边权重，结合指数动量保留长程趋势，最终融合局部熵与历史传播风险得到轨迹级置信度。
   与已有工作的本质区别：传统传播通常假设均匀衰减或固定图结构；RUPA的权重由数据驱动（基于未标注轨迹的关系强度方差校准），且专门引入Goal Alignment语义对齐得分替代普通边权重，更贴合Agent任务目标导向。
4. **验证UQ信号的可操作性与早期检测能力**：不仅报告AUROC/AUPRC提升，还展示RUPA分数可直接用于多采样动作选择与截断前缀检测，形成“评估-干预”闭环。
   与已有工作的本质区别：多数UQ工作仅停留在离线预测质量评测；本文进一步证明提升的UQ质量能转化为Terminal-Bench-2与GAIA上接近翻倍的Agent成功率增益，凸显实用价值。

## 方法详解
- **Relational Trajectory Graph Construction**：将Agent执行前缀表示为$\mathcal{G}=(\mathcal{V}, \mathcal{E})$，节点$v_i \in \mathcal{V}$涵盖用户指令、助手推理/动作、工具调用与环境观测。在受限历史窗口（Visible Window=5）内构建7类有向边：Sequential（紧邻前步）、Latest（最新环境/用户输入）、Repetition（高度相似推理/工具模式）、Progression（延续现有方案）、Parallel（同任务备选分支）、Feedback（指向失败/不稳定观测）、Goal Alignment（当前步骤与原始目标的语义依赖）。边类型通过文本嵌入距离（bge-m3）、词法特征匹配（如`next/therefore`→Progression，`error/timeout`→Feedback）及工具调用规范化签名确定，全程outcome-blind，不使用未来标签。
- **Relation-aware Uncertainty Propagation**：
  - 局部不确定性$U_t$：助手节点取predictive entropy；环境节点取执行失败/空响应/冲突反馈等可观测信号。
  - 边权重由关系可靠性$\rho_{\tau}$、关系强度$\tilde{r}_{it}$与时序衰减$\delta^{\text{age}(i,t)-1}$共同决定：$w_{it} = \rho_{\tau_{it}} \tilde{r}_{it} \delta^{\text{age}(i,t)-1}$。
  - Goal Alignment边不使用权重，直接计算对齐得分：$Q_{it} = 1 - S(y_t, x)$。
  - 图传播聚合：$G_t = \frac{\sum_{i \in \mathcal{N}(t)} w_{it}(P_i + Q_{it})}{\sum_{i \in \mathcal{N}(t)} w_{it} + \epsilon}$，其中$P_i$为历史节点已存储的传播风险。
  - 指数动量保留长程趋势：$m_t = \frac{\sum_{k < t} \gamma^{t-k} P_k}{\sum_{k < t} \gamma^{t-k} + \epsilon}$，融合为$H_t = \eta_g G_t + \eta_m m_t$。
  - 最终轨迹级不确定性：$R_t = \lambda_u U_t + \lambda_h H_t$，完整轨迹得分由各步$R_t$聚合，分数越高风险越高。
- **参数校准**：关系权重$\rho_{\tau} = |T| \frac{\exp(q_{\tau}/T)}{\sum_{\tau' \in \mathcal{T}} \exp(q_{\tau'}/T)}$，其中$q_{\tau} = \mathrm{Var}(\tilde{r}_{\tau}) / (\mathbb{E}(\tilde{r}_{\tau}) + \epsilon)$，仅基于未标注训练轨迹计算，固定后跨模型/数据集复用，无需测试标签微调。

## 实验与结果
- **数据集与模型**：τ-2（对话决策）、Terminal-Bench-2（命令行软件工程）、GAIA（开放域复杂推理）。使用6个开源LLM：Qwen3.5-27B、Qwen3.6-35B-A3B、Gemma-4-26B-it、Gemma-4-31B-it、GPT-OSS-120B、MiniMax-M2.7。
- **基线**：Entropy、Seq-prob、SAUP、Tracer、UProp。评估指标：AUROC、AUPRC、Best F1。
- **主要结果**：
  - RUPA在全部6模型×3基准组合上均取得最优平均AUROC
