---
title: "Beyond-Single-Turn-Confidence-Trajectory-Adapted-Uncertainty"
source: https://arxiv.org/pdf/2608.11552v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:39:01"
field: "大语言模型不确定性量化"
keywords: ["uncertainty quantification", "LLM agents", "trajectory confidence", "tool-use", "self-consistency"]
innovations: ["系统评估三类单轮UQ方法在多轮智能体轨迹上的迁移性", "提出轨迹级白盒聚合、黑盒一致性（TER/ASC/ADC/AEC）与反思式评分的统一框架", "揭示stuck-policy导致的一致性评分歧义及聚合策略的关键影响"]
benchmarks: ["BFCL-v4", "τ²-bench"]
---

# 论文速读：Beyond-Single-Turn-Confidence-Trajectory-Adapted-Uncertainty

## 一句话总结
本文系统评估了三类单轮不确定性量化的迁移性，将白盒概率、黑盒一致性与反思式方法适配到多轮工具调用智能体轨迹场景中，发现黑盒自一致性通常是最强方案（TER最高达0.868 AUROC），反思式是低成本有效基线，而白盒方法性能受跨轮聚合策略影响极大且不稳定。

## 研究问题与动机
- **单轮UQ方法的局限性**：现有不确定性量化方法主要评估单个生成响应，但智能体的观测单元是交互式轨迹，错误可能在多步决策中累积并传播。
- **缺乏系统性对比**：此前工作提出了传播方法（如SAUP、UProp）和轨迹级不确定性形式化框架（Oh et al., 2026），但未在匹配条件下对三类主流UQ方法进行跨模型、跨数据集的公平对比。
- **部署安全需求**：可靠的不确定性信号可使智能体在执行前放弃或升级，但当前方法在轨迹级预测成功/失败方面的可靠性尚不明确。

## 核心贡献（创新点）
- **首篇系统对比三类UQ方法在智能体轨迹上的迁移性**，涵盖判别力、校准性与选择性预测三个维度。
- **提出轨迹级适配方案**：将白盒方法扩展为跨轮聚合（均值/最小/首末轮），并设计新的黑盒一致性度量（TER、ASC、ADC、AEC）。
- **揭示AGENT-UQ的独特挑战**：证明轨迹不确定性不能简单视为"更长上下文的单轮不确定性"，三类方法在不同模型-数据集组合上表现出差异化的失效模式。
- **提供计算预算-性能权衡分析**：量化三类方法的额外开销，发现反射式方法在零额外轨迹采样下表现稳健，是实际部署的强基线。

## 方法详解
- **白盒评分器（White-Box）**：基于token概率，包括序列概率（SP）、长度归一化序列概率（LNSP）、平均token负熵（ATN@K），通过first/mean/min/last四种跨轮聚合策略生成轨迹级置信度。
- **黑盒一致性评分器（Black-Box Consistency）**：采样m条额外轨迹后测量一致性，包括最终消息一致性（NCP）、动作结构一致性（FAC/ASC/ADC/AEC）、轨迹等价率（TER，使用gemini-flash-lite作为Judge）。
- **反思式评分器（Reflexive）**：要求模型评估自身轨迹，包括P(True)（预测"True"的概率）和Verbalized Confidence（模型自报0-1置信度）。
- **评估指标**：AUROC为主，辅以AUPRC、选择性预测风险覆盖率（PRR）、期望校准误差（ECE）。

## 实验与结果
- **数据集**：BFCL-v4（200任务）、τ²-bench的airline（50）、telecom（114）、retail（114）。
- **模型**：Qwen2.5-7B、gpt-oss-20b、Qwen3.5-9B、MiniMax-M3、gpt-4o-mini。
- **关键结果**：
  - 白盒：SPmean在Qwen3.5-9B telecom达0.725，但同一模型同分数改min聚合降至0.228；91/240白盒单元格低于0.5。
  - 反思式：P(True)在BFCL-v4 gpt-4o-mini达0.85，但airline上差距大（0.225–0.659）。
  - 黑盒：TER在airline Qwen3.5-9B达0.868，ASC在BFCL-v4 gpt-oss-20b达0.819；NER在gpt-4o-mini telecom仅0.342。
  - 无单一方法在所有模型-数据集组合上稳定最优。
- **稳定性分析**：白盒聚合策略可导致AUROC剧烈波动（图1），一致性随样本数m增加而提升，但stuck-policy现象导致失败轨迹重复一致。

## 相关工作脉络
- **单轮UQ方法**：概率法（SP/LNSP/entropy）、一致性法（self-consistency/semantic entropy）、反射法（Kadavath et al., 2022；Tian et al., 2023）——本文将其系统迁移至轨迹级。
- **智能体不确定性传播**：SAUP（Zhao et al., 2025）和UProp（Duan et al., 2025）——本文提供对比控制组，发现其平均AUROC仅0.434，低于最佳固定聚合（0.53）。
- **轨迹不确定性形式化**：Oh et al. (2026)——本文在其框架基础上补充实证评估与跨模型对比。
- **多轮对话置信度**：Zhang et al. (2026a)聚焦逐轮校准——本文关注轨迹整体成功预测。
- **工具使用校准**：UALA（Han et al., 2024）、ProbeCal（Liu et al., 2024）——本文证明简单单轮方法的迁移性仍不可靠。

## 局限性与未来方向
- **数据集范围有限**：仅覆盖BFCL-v4和τ²-bench的工具使用基准，未测试web浏览或具身智能场景。
- **标签噪声**：轨迹成功标签由数据库状态、断言、LLM judge混合生成，上限约束了任何评分器的判别力。
- **模拟用户混淆**：一致性评分测量的是智能体-用户联合系统的变异性，而非纯智能体不确定性。
- **统计功效不足**：部分数据集（如airline仅50任务）bootstrap区间宽，微小差异可能不显著。
- **未来方向**：建议测试步级标签、探索在推理过程中利用评分进行放弃/澄清/升级的实际效果、扩展到更多智能体环境。

## 研究启发与可借鉴点
- **跨轮聚合策略需实证验证**：白盒方法的性能高度依赖聚合选择（mean/min/last），不可直接移植，需针对部署场景选择。
- **TER作为强基线的价值**：轨迹等价率结合LLM Judge比符号级动作匹配更鲁棒，可扩展到其他需要语义等价判断的场景。
- **反思式方法的实用优势**：仅需一次额外pass，AUROC可达0.85+，适合成本敏感场景。
- **stuck-policy诊断方法**：通过success-success/failure-failure配对分析揭示一致性评分的歧义性，可作为一致性方法的可解释性工具。
- **接口消融设计**：区分文本动作接口与原生工具调用，发现一致性指标在接口间转移良好，为方法鲁棒性提供证据。

## 关键术语表
- **White-box scorer**：利用模型token概率计算置信度的方法，需访问内部概率分布。
- **Black-box consistency**：通过多次采样测量响应一致性来估计不确定性的方法，无需内部概率。
- **Reflexive scorer**：让模型自我评估轨迹正确性或置信度的方法，如P(True)和VC。
- **Trajectory Equivalence Rate (TER)**：使用LLM Judge判断采样轨迹与参考轨迹是否达成相同任务结果的比率。
- **Action-Set Consistency (ASC)**：比较轨迹动作类型集合的Jaccard相似度，忽略顺序和重复。
- **Prediction Rejection Ratio (PRR)**：衡量选择性预测性能的指标，正值表示优于随机拒绝。
- **Stuck-policy**：智能体在失败任务上重复相同错误行为的模式，导致一致性评分虚高。
- **Cross-turn aggregation**：将多轮置信度聚合为轨迹级分数的策略，如均值、最小值、首末轮。

## 可复现要素
- **数据集**：BFCL-v4（Apache-2.0）、τ²-bench（MIT），均来自官方仓库。
- **代码/权重**：论文未明确开源声明，但提到数值网格已发布（"numeric grid is released alongside it"）。
- **关键超参**：采样轨迹数m=3，温度T=0.7；白盒top-K=5；NLI模型为microsoft/deberta-large-mnli；TER Judge为gemini-flash-lite。
- **评估协议**：1000次任务聚类bootstrap，95%置信区间。
