---
title: "Refusing-Intent-Not-Form-Wrapper-Based-Intent-Group-Supervis"
source: https://arxiv.org/pdf/2608.13304v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:50:20"
---

# 论文速读：Refusing-Intent-Not-Form-Wrapper-Based-Intent-Group-Supervis

## 一句话总结
本文针对大模型安全微调中因表面包装形式导致的“捷径学习”问题，提出基于意图组的自动数据增强方法 WIFA，并在此基础上设计了强调变换有害拒绝的 WIFA-Boost 两阶段训练范式，以及通过锚定组一致性正则化显著降低良性误拒的 A-GCRT 目标，实现了“拒意图而非拒形式”的安全对齐。

## 研究问题与动机
- 现有安全微调通常在孤立 prompt–response 对上进行监督，模型容易将 wrapper 形式（角色扮演、学术包装、格式约束等）本身学习为拒绝/服从的虚假标签。
- 仅补充包装后的有害示例会教出 wrapper-as-refusal shortcut：表面安全指标上升，但同等包装的良性请求会被过度拒绝。
- 推理时防御（self-reminder、意图分析等）与外部 guard 系统无法更新模型参数，难以从根本上切断形式与标签的虚假相关性。
- 安全调优需同时优化两个常被混淆的目标：提升对变换形式有害请求的鲁棒拒绝，以及保留形态相似良性请求的合理服从。

## 核心贡献（创新点）
- **意图组级别的自动数据增强（WIFA）：** 无需外部教师模型或逐 wrapper 人工标注，通过共享包装家族将有害意图组与结构匹配的良性意图组配对，使 wrapper 形式失去预测决策的能力。
- **双路线训练范式：** 在统一数据层上衍生出两条互补路径：WIFA-Boost 通过两阶段 SFT 聚焦高安全操作点；A-GCRT 引入组一致性方差约束与方向性 margin 锚定，精准调节安全与误拒的权衡。
- **可复现的操作点突破：** 在 Qwen 与 Llama 双设置及七类基准上验证，复现的 prompt-time 与 training-time 基线均无法达到同等性能；WIFA-Boost 取得最强变换有害拒绝，A-GCRT-M5 将 OR-Bench 误拒率从基座 25.7% 降至 17.4%。

## 方法详解
- **WIFA 数据构建：** 对每个源意图 $z$，保留直接形式提示 $d(z)$ 并应用 $|\mathcal{W}_m|=7$ 种固定包装（学术、虚构、安全顾问、假设、角色施压、人设施压、格式约束）。有害组 $G_h(z_h) = A_h(z_h) \cup \{w(z_h): w \in \mathcal{W}_m\}$（含 2 个直接拒绝锚点），良性组 $G_b(z_b) = \{w(z_b): w \in \mathcal{W}_m\}$。目标格式统一为 `[INTENT ANALYSIS] d(z) [/INTENT ANALYSIS] y(z)`，利用基础模型自身对直接形式的回答生成目标，实现零外部教师自蒸馏。
- **WIFA-Boost 两阶段训练：** $\theta_1 = \mathrm{SFT}(\theta_0, \mathcal{D}_W)$，$\theta_2 = \mathrm{SFT}(\theta_1, \mathcal{D}_{\mathrm{cal}})$。第一阶段 5750 条 WIFA 样本建立意图-形式结构；第二阶段 2750 条校准样本（原有害拒绝 + 500 条纯良性）恢复良性服从。顺序不可逆，反转会导致困难子集拒绝率坍塌。
- **A-GCRT 正则化目标：** $\mathcal{L} = \mathcal{L}_{\mathrm{SFT}} + \lambda_{\mathrm{gcr}}(\mathcal{L}_{\mathrm{var}} + \gamma \mathcal{L}_{\mathrm{anchor}})$。
  - **决策位置分数：** 在 `[/INTENT ANALYSIS]` 后首个响应 token 处，计算预设拒绝前缀集 $\mathcal{R}$ 与合规前缀集 $\mathcal{C}$ 的最大 logit 之差：$s_\theta(x) = \max_{r\in\mathcal{R}}\ell_\theta(r|x) - \max_{c\in\mathcal{C}}\ell_\theta(c|x)$。
  - **组内一致性损失（$\mathcal{L}_{\mathrm{var}}$）：** 同一意图组内各包装形式决策分数的方差，迫使相同意图的不同表面形式输出一致的拒绝/服从倾向。
  - **方向性锚定损失（$\mathcal{L}_{\mathrm{anchor}}$）：** 有害组要求 $\bar{s}_z \geq m$，良性组要求 $\bar{s}_z \leq -m$，以 hinge loss 将两组推向决策边界的相反两侧。Margin 与权重参数通过验证集选择操作点，而非单调调节旋钮。

## 实验与结果
- **设置与基线：** 主模型 Qwen2.5-7B-Instruct（250 条 AdvBench-style 有害种子），对照 Llama-3.1-8B-Instruct（250 条 HH-Inst 种子）。对比 3 种 prompt-time 防御与 3 种训练时防御（Vanilla Refusal-SFT, LookAhead Tuning, RATIONAL 等），评估涵盖 HarmBench、SORRY-Bench（5 突变子集）、OR-Bench-Hard、XSTest、StrongREJECT、MMLU、GSM8K 及 15 类未见攻击族。
- **Qwen 主结果：** WIFA-Boost 达到最强变换有害拒绝：SB-avg5 从 22.1 提升至 63.7，SB-mis 达 59.3，OR 为 56.0。A-GCRT-M5 实现低误拒操作点：OR-Bench 从 25.7% 降至 17.4%（低于基座与所有复现基线），SB-avg5 升至 46.7%，MMLU 与 GSM8K 保持接近基座（69.0 / 83.70）。
- **泛化与消融：** 未见攻击族测试显示 WIFA-Boost 平均 ASR 最低（9.5），优于 A-GCRT-M5（16.8）。消融证实匹配良性包装、意图分析目标、两阶段顺序及 A-GCRT 双损失项缺一不可；单纯更强 SFT 或单组件正则无法复现低 OR 点。Llama 结果印证了安全优先与低误拒操作点的分离趋势，但未重复低于基座 OR 的现象。

## 相关工作脉络
- **安全对齐与拒绝微调：** RLHF/DPO/指令微调多监督孤立样本对，未显式绑定同一意图的不同表面形式；本文以意图组为监督单元，从数据层切断形式捷径。
- **Jailbreak 与鲁棒安全基准：** HarmBench/SORRY-Bench 等强调表面形式鲁棒性，但仅追加变换有害数据易使形式成为标签
