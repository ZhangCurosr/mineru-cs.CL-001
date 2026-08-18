---
title: "CERTIFY OR REFUSE: A CROSS-MODEL MAP FOR SELECTIVE RISK CONTROL WITH COVERAGE FLOORS UNDER COVARIATE SHIFT"
source: https://arxiv.org/pdf/2608.10893v1.pdf
model: agnes-2.5-flash
chunks: 7
summarized_at: "2026-08-18 08:29:39"
field: "分布外泛化与选择性预测"
keywords: ["covariate shift", "selective risk control", "conformal prediction", "coverage floor", "certificate", "cross-model"]
innovations: ["硬覆盖地板形式化：首次将覆盖率下界作为可验证证书引入协变量移位场景", "局部泛函替代全局ESS：用局部接受区域泛函追踪认证成本，Spearman相关从.026提升至.936", "四块分割Algorithm 1：分离风险测试/地板测试/权重估计，使样本效率比oracle-A提升128×"]
benchmarks: ["SQuAD→NewsQA", "Grid Audit (256 cells)", "Deep Audit (6 cells × 4000 repeats)"]
---

# 论文速读：CERTIFY OR REFUSE: A CROSS-MODEL MAP FOR SELECTIVE RISK CONTROL WITH COVERAGE FLOORS UNDER COVARIATE SHIFT

## 一句话总结
在目标域存在协变量移位（covariate shift）的场景下，本文提出一种**跨模型地图（cross-model map）**机制，通过"认证或拒绝（certify or refuse）"策略实现选择性风险控制；核心创新是引入**硬覆盖地板（hard coverage floor）**概念，以可计算的证书保证目标域风险与覆盖率的下界，并在 SQuAD→NewsQA 真实迁移任务中零违规验证通过。

## 研究问题与动机
- **协变量移位下的选择性分类困境**：当目标域输入分布 $Q$ 偏离源域分布 $P$ 时，基于源域训练的模型在目标域上的风险（误分类率）不可控；传统方法往往缺乏对"拒绝决策"本身的统计保证。
- **现有风险控制方法的缺陷**：Conformal prediction（如 SCoRE、Weighted conformal）在无标记目标数据上可能产生**违规（violations）**——即实际覆盖率低于承诺水平，且对权重分布剧烈变化敏感；DRO-box 类方法虽安全但过于保守（escalate-all 拒绝），无法区分"真正高风险"与"低质量区域"。
- **覆盖率地板的必要性**：在下游实际部署（如 RAG、问答系统）中，除了控制风险上界，还需要保证**覆盖率下界**——即至少有多少比例的输入能获得有效预测，否则拒绝策略将失去实用价值。

## 核心贡献（创新点）
1. **硬覆盖地板（hard coverage floor）形式化**：将可行性前沿（feasibility frontier）操作化为可验证的统计对象，区分标记源资源与无标记目标资源的认证成本；与已有 work 的本质区别在于首次将覆盖率作为"地板"而非"软约束"引入选择性风险框架。
2. **跨模型证书地图（cross-model certificate map）**：通过四块分割算法（Algorithm 1）同时输出风险证书和覆盖率地板证书，以局部泛函 $\frac{\mathbb{E}_P[w^2 S_\lambda]}{(\mathbb{E}_P[w S_\lambda])^2}$ 追踪认证成本；与 SCoRE 等全局 conformal 方法的本质区别是放弃 ESS 全局估计，转而使用局部接受区域泛函，Spearman 相关从 .026 提升至 .936。
3. **拒绝诊断标签体系**：将失效模式分为 risk-starved / floor-starved / nuisance-limited / frontier-infeasible 四类并固定优先级，使证书输出具有可解释的诊断信息；与 Plug-in 等方法无诊断能力的本质区别在于提供结构化失败归因。
4. **分层移位直方图估计量（Theorem 3 / Proposition 1）**：以概率 $\geq 1-\delta_w$ 同时控制所有格网区域的偏差，实现双轴收敛速率匹配；与 oracle-A（已知权重）的本质区别在于无需真实权重即可获得同阶保证，样本效率提升高达 128×。

## 方法详解
- **四块分割 Algorithm 1**：将数据分为四组——风险测试集（大小 $n_r$）、地板测试集（大小 $m_f$）、权重估计源域组（$n_w$）与目标域组（$m_w$），分别完成风险证书、覆盖率地板、IPW 权重估计三项任务，避免过拟合。
- **Model-B' 与 K 细胞分层移位模型**：预注册 $K$ 个细胞（cells），假设权重 $w \leq B$ 且有 $\kappa$-正则前沿边际，在每个细胞内估计局部风险与覆盖率；网格尺度控制在 $< s/4$（$s$ 为 LR_margin 参数），以确保边缘区域采样充分。
- **证书输出规则**：对于每个查询，计算风险分数对齐格网（LR_margin(s)）上的风险上界与覆盖率下界；若两者同时满足阈值则输出预测，否则输出拒绝，并附带四维诊断标签之一。
- **损失函数/目标**：核心目标是最小化加权风险 $\mathbb{E}_P[w S_\lambda]$ 同时最大化覆盖率 $\mathbb{E}_P[w \cdot \mathbb{1}[\text{accept}]]$，约束为覆盖率 $\geq \beta^*$（地板）且风险 $\leq \alpha$；定理 3 给出两种形式：条件 sharp 形式（依赖 $D_w$）和无条件确定形式。
- **干扰轴未匹配问题**：$n_w \asymp B^2 K p(A_{\lambda_s})/(\kappa s)^2$，$B^2 K$ 仅为充分非必要条件；未知 $\eta$ 是开放边界问题（Theorem 8）。

## 实验与结果
- **数据集与设置**：SQuAD → NewsQA 跨域迁移；评估样本 $N_{\text{eval}} = 4{,}212$；证书置信度 $\delta = 0.05$，每单元容差 $\delta_{\text{tol}} = 0.0625$；两个 leg 分别使用 Llama-3-8B 和 DeepSeek-V4-Flash 作为源模型。
- **网格审计（Grid Audit）**：4 强度 × 256 单元格，1,000 次重复，B1 网格验证 591/1,024 单元格，**0 违规**，Holm 校正后 0 拒绝；深审计 6 单元格 × 4,000 重复（共 48,000 次），CP-UCB 违规概率仅 $6.2 \times 10^{-5}$。
- **基线对比**：Floor-free 108 次 Holm 拒绝，Weighted-conformal 40 次拒绝，Plug-in 662 次拒绝，DRO-box certify 0 单元格（过度保守）；本文方法 B' 仅验证 8/1,024 单元格，但完全验证样本效率比 oracle-A 高 **64×**，首次验证样本比高 **128×**。
- **真实工作负载结果**：Leg 1 (Llama-3-8B) $\hat{\beta}^*(0.10)=0.050$（CI [0.035, 0.130]）；Leg 2 (DeepSeek-V4-Flash) $\hat{\beta}^*(0.10)=0.209$（CI [0.160, 0.281]）；地板 $\beta=0.60$ 在所有测试 $\alpha$ 下均未穿越；证书最终输出 **2,000/2,000 诚实拒绝，0 违规**，完美通过诚实性检验。
- **局部泛函验证**：K 价格指数指数为 $K^{0.955}$（95% CI [0.755, 1.154]）；边界发散斜率（log-log）为 **−2.002**（CI [−2.142, −1.862]）；局部泛函 Spearman 相关 **.936**（CI [.927, .944]），全局 ESS 相关仅 .026，印证局部泛函更适合作为认证成本代理。

## 相关工作脉络
- **SCoRE（Bai & Jin, 2026）**：conformal 风险调整 e-values，等构造对比臂；本文与之对比在于放弃等构造假设，使用局部泛函替代，规避其在协变量移位下的违规风险。
- **Weighted Conformal（Tibshirani et al., 2019; Barber et al., 2023）**：基于 IPW 的 conformal prediction；本文指出其在协变量移位下产生 40 次 Holm 拒绝（违规），而本文方法零违规，核心差异在于引入硬地板约束。
- **DRO-box**：分布鲁棒优化型拒绝方法，escalate-all 拒绝策略；本文证明其虽然保守但失去实用性（certify 0 单元格），本文方法在同等安全保证下释放更多可预测区域。
- **Plug-in 方法**：直接代入估计权重；本文实验显示其产生 662 次拒绝（严重违规），原因在于未考虑估计不确定性，本文通过四块分割和证书机制弥补此缺陷。
- **Conformal Prediction 传统框架（Vovk et al.）**：假设 i.i.d. 数据；本文明确在协变量移位（i.i.d. 假设破坏）下工作，将 conformal 思想推广至非平稳场景。

## 局限性与未来方向
- **干扰轴未匹配问题**：权重估计所需样本量 $n_w$ 的上界 $B^2 K$ 仅为充分非必要，未知 $\eta$ 下最优收敛速率仍是开放边界（Theorem 8）。
- **K 细胞数的先验依赖**：模型 B' 需预注册 K，若 K 设置不当（过大或过小）可能影响验证效率；论文未讨论自适应 K 选择机制。
- **仅验证二分类/简单结构化输出**：实验集中在 SQuAD→NewsQA 问答场景，对多分类、序列生成等复杂任务的推广性未验证。
- **拒绝诊断标签的优先级固定**：risk-starved/floor-starved 等四类优先级为固定设定，未探索数据驱动的动态优先级调整。

## 研究启发与可借鉴点
- **四块分割策略**可迁移至任何需要分离"训练/估计"与"证书/验证"的计算场景，避免过拟合污染证书有效性，值得在自定义风险评估管线中复用。
- **局部泛函替代全局 ESS**的思路：当全局估计不稳定时，用局部接受区域泛函 $\frac{\mathbb{E}[w^2 S_\lambda]}{(\mathbb{E}[w S_\lambda])^2}$ 作为代理指标，Spearman 相关从 .026 跃升至 .936，这一技巧可推广至其他加权风险估计任务。
- **拒绝诊断四维标签体系**可为下游系统的可解释性设计提供模板——将失败模式分类并赋予优先级，便于工程团队快速定位是数据问题还是模型问题。
- 可结合本团队已有的大模型 RAG pipeline：在检索增强阶段引入硬覆盖地板，自动过滤"低质量检索结果"区域，替代当前的启发式阈值策略。

## 关键术语表
**Covariate Shift（协变量移位）**：源域与目标域的输入分布 $P(X)$ 不同但条件分布 $P(Y|X)$ 相同的数据迁移设定。
**Hard Coverage Floor（硬覆盖地板）**：保证目标域中至少有比例 $\beta^*$ 的样本可获得有效预测（不被拒绝）的统计下界。
**Certificate / 证书**：基于抽样数据提供的、以概率 $1-\delta$ 保证的风险上界与覆盖率下界的数学声明。
**ESS（Effective Sample Size）**：加权样本的有效样本量，衡量权重分布的不均匀程度；本文指出全局 ESS 与认证成本几乎无关（Spearman=.026）。
**IPW（Inverse Probability Weighting）**：利用权重 $w(x)=dQ/dP$ 校正协变量移位的重加权技术。
**LR_margin（Risk-score-aligned Lattice Margin）**：风险分数对齐格网，用于将连续风险空间离散化为可验证的细胞单元。
**Feasibility Frontier（可行性前沿）**：在给定资源约束下可同时实现的风险-覆盖率 Pareto 最优边界。

## 可复现要素
- **数据集**：SQuAD（源域）和 NewsQA（目标域），均为公开数据集；评估集 $N_{\text{eval}}=4{,}212$ 由论文构造。
- **代码/权重**：论文未明确声明开源（截至阅读时间）；基础模型为 Llama-3-8B 和 DeepSeek-V4-Flash，权重公开可下载。
- **关键超参**：$\delta=0.05$（证书置信度），$\delta_{\text{tol}}=0.0625$（每单元容差），$s$（LR_margin 网格尺度，需 $<s/4$），$K$（细胞数），$B$（权重上界）；具体数值论文第 2 段未全部列出，需查阅原文补充。
