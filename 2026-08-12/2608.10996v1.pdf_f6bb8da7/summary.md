---
title: "C<sub>o</sub>nR<sub>u</sub>b<sub>-</sub>M<sub>e</sub>d<sub>:</sub> R<sub>e</sub>inf<sub>o</sub>r<sub>ce</sub>m<sub>e</sub>nt L<sub>ea</sub>rnin<sub>g w</sub>ith C<sub>o</sub>n<sub>se</sub>n<sub>sus</sub> R<sub>u</sub>bri<sub>cs</sub> f<sub>o</sub>r O<sub>pe</sub>n<sub>-</sub>End<sub>e</sub>d Medical Question Answering"
source: https://arxiv.org/pdf/2608.10996v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 07:59:35"
---

# 论文速读：C<sub>o</sub>nR<sub>u</sub>b<sub>-</sub>M<sub>e</sub>d<sub>:</sub> R<sub>e</sub>inf<sub>o</sub>r<sub>ce</sub>m<sub>e</sub>n<sub>t</sub> L<sub>ea</sub>r<sub>n</sub>i<sub>n</sub>g w<sub>i</sub>t<sub>h</sub> C<sub>o</sub>n<sub>s</sub>e<sub>n</sub>s<sub>u</sub>s</sub> R<sub>u</sub>bri<sub>c</sub>s</sub> f<sub>o</sub>r O<sub>p</sub>e<sub>n</sub>-<sub>E</sub>n<sub>d</sub>e<sub>d</sub> M<sub>e</sub>d<sub>i</sub>c<sub>a</sub>l Q<sub>u</sub>e<sub>s</sub>t<sub>i</sub>o<sub>n</sub> A<sub>n</sub>s<sub>w</sub>e<sub>r</sub>i<sub>n</sub>g

## 一句话总结
ConRub-Med 针对开放式医学问答缺乏廉价、单一结果验证器的难题，提出基于多模型共识 Rubric 的可验证强化学习框架；通过聚类生成可解释的原子评分标准，结合三态评分与配对序列优势处理机制，在 Qwen3 基座模型上显著提升了健康推理能力。

## 研究问题与动机
- 数学/编程的 RLVR 依赖廉价且单一的答案校验器，而开放式医学问题答案常呈部分正确、不完整或含临床顺序错误，无法用单一函数自动验证。
- 专家手工编写 rubrics 临床 grounding 强，但每轮调用成本过高难以规模化；纯模型自动生成 rubrics 又与专家标注存在显著质量落差。
- 现有 GRPO 等策略梯度方法在组内响应获得相同最终奖励时，优势函数全为零，策略优化失去方向信号。
- 医学 RL 需要在奖励设计中保留“正确覆盖、缺失信息、错误声明”的有意义区分，并将事实错误明确惩罚为负分，而非传统做法中与未作答等同的零分。

## 核心贡献（创新点）
1. **多模型共识 Rubric 构建管道**：利用异构大模型独立生成原子候选标准，经语义聚类、去重与医学专家盲标筛选，产出兼具高临床相关性与低冗余度的稳定规则集。（本质区别：打破单模型生成或纯人工编写的二元选择，以“多样性输入+确定性过滤”平衡可扩展性与临床可靠性。）
2. **三态精细化评分机制**：将每条标准判定映射为 CORRECT(+1)/MISSING(0)/WRONG(−1)，错误声明被显式赋予负分，精确刻画医学回答的事实质量差异。（本质区别：突破二值对/错奖励，避免错答与未作答在梯度信号上无差别对待，缓解 RL 中的“负样本稀释”问题。）
3. **Exact-Reward Ties 配对序列优势模块**：仅在 GRPO 组内 8 响应最终奖励完全相等时触发，通过双向配对判决器生成非零序列优势 d_i ∈ {−1, 0, +1} 直接注入训练优势。（本质区别：不修改原始标量奖励函数、不引入额外 preference loss，以低侵入方式修复平局梯度消失。）
4. **分层判决器与审计工作流**：配套设计二元/三元/双向配对三种专用判决器，并集成 `audit keep_p` 治理管道与临床上下文回退策略，形成可复现的医学 RLVR 基础设施。（本质区别：从单一 prompt judge 升级为正交分工的多级判决架构，兼顾吞吐量、事实严谨性与系统容错。）

## 方法详解
- **共识 Rubric 构建**
  - 三个异构模型（GPT‑5 Mini、Gemini 2.5 Pro、Claude Sonnet 4.5）独立从提示与参考上下文生成 20–30 个原子候选标准，每候选带参考覆盖标签（仅用于构建，不作奖励标签）。
  - DeepSeek‑V4‑Pro 执行确定性检查剔除空、欠规范、内部重复输出；筛选领域错误、逻辑矛盾与参考冲突；对表达相同潜在命题的标准聚类。
  - 仅当语义簇支持集大小 |Sℓ| = 3 且参考覆盖标签在簇内一致时才保留；每簇以最长成员为初始代表，经 reviewer 决定保留/重写/排除。
  - 去重后定额保留最多 10 个内容标准，附加 2 个全局控制（事实正确性、响应适当性）。
- **三态评分与奖励计算**
  - Judge 输出 v_ij ∈ {CORRECT, MISSING, WRONG}，映射为 s_ij ∈ {+1, 0, −1}；WRONG 专指事实错误、捏造、矛盾、反转或替换（不准确本身不自动判为 WRONG）。
  - 标准平均分 r_crit = (1/|C(q)|) Σ s_ij；错误贡献负分而非零分。
  - 最终奖励 R_i = r(y_i; q) + p_len(L_i)，含软长度惩罚，防止模型通过堆砌文本刷分。
- **配对序列优势（Exact‑Reward Ties Handling）**
  - 触发条件：完整 8 响应组且所有最终奖励有限且精确相等（此时 vanilla GRPO 优势全为零）。
  - 随机分对后，每对 (y_a, y_b) 由配对 judge 在原始与交换两种顺序下双向评估，仅当两种顺序偏好一致时才赋予非零序列优势 d_i ∈ {−1, 0, +1}，直接写入训练 advantage。
  - 不改变标量奖励，不添加单独 preference loss。
- **优化目标**：采用带不对称 clip 的 GRPO 策略梯度损失；非平局组沿用 vanilla GRPO 优势。
- **判决器设计**
  - **二元标准判决器**：严格逐条判断 yes/no，字面解释原则，仅返回 JSON 字符串数组。
  - **三元标准判决器**：区分 correct（实际陈述或明确暗示且事实正确）、wrong（直接错误/反向/替代）、missing（完全不涉及该标准）；明确区分“传递错误信息=wrong”与“未传递有用信息=missing”。
  - **双向配对判决器**：对称调用两次，仅当双向映射识别出相同潜在
