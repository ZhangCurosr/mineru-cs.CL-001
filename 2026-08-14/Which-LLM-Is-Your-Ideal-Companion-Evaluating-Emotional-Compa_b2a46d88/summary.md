---
title: "Which-LLM-Is-Your-Ideal-Companion-Evaluating-Emotional-Compa"
source: https://arxiv.org/pdf/2608.13168v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:24:37"
field: "LLM 人格与情感交互评测"
keywords: ["成人依恋理论", "情感陪伴评测", "ECBench", "LLM Psychometrics", "依恋风格引导", "双模型对话"]
innovations: ["首次将成人依恋理论（ECR-R 量表）适配至 LLM 评测并量化四维依恋风格", "提出 ECBench 跨场景（情绪支持/协作/冲突/社交指导）跨关系（友谊/恋爱）情感陪伴基准", "通过 persona-induction 提示引导证明疏离/恐惧依恋风格显著降低对话陪伴质量"]
benchmarks: ["ECBench", "ECR-R"]
---

# 论文速读：Which-LLM-Is-Your-Ideal-Companion-Evaluating-Emotional-Compa

## 一句话总结
本文首次将成人依恋理论引入 LLM 情感陪伴能力评测，使用 ECR-R 量表测量 32 个模型的依恋焦虑/回避倾向，并提出 ECBench 跨场景双模型对话基准；研究发现大多数主流模型呈现安全型依恋，安全型和痴迷型表现最佳，而通过提示诱导的疏离型和恐惧型显著降低陪伴质量。

## 研究问题与动机
- 现有 LLM 个性评测（MBTI、Big Five 等）聚焦泛化人格特质，无法刻画模型在亲密/情感敏感场景中的互动模式。
- 情感陪伴（情绪支持、冲突调解、社交指导）已成为 LLM 重要应用，但缺乏基于心理学理论的细粒度关系导向评测框架。
- 不同模型在面对同一情感需求时表现出明显差异化的回应风格（如安慰 vs 实用建议），背后是否存在稳定的"依恋倾向"尚未被系统揭示。
- 依恋倾向是否可通过提示工程被引导/塑造，并如何在多轮对话中落地为可观察的行为差异，仍缺乏实证证据。

## 核心贡献（创新点）
1. **首次将成人依恋理论与 ECR-R 量表适配至 LLM 评测**：从焦虑/回避二维空间量化模型在亲密关系中的情感亲近、安全感与疏离倾向，区别于传统人格量表仅测量泛化特质。
2. **提出 ECBench 情感陪伴对话基准**：覆盖情绪支持、协作任务、冲突解决、社交指导四个场景，涵盖友谊与恋爱两种亲密关系，弥补已有评测对情境化互动质量的忽视。
3. **构建 11 指标三维评测框架 + 三方评估机制**：参评者体验（4 项）、一般交互质量（3 项）、角色专属性能（4 项），联合参与者评分、外部 LLM 盲评、人工标注，形成可三角验证的评价体系。
4. **展示依恋风格提示引导的有效性及其对对话质量的因果影响**：通过 persona-induction prompt 将 GPT-3.5-Turbo 和 DeepSeek-V4-Pro 分别诱导至疏离型与恐惧型，揭示"高回避/高恐惧风格显著损害陪伴质量"。

## 方法详解
- **ECR-R 量表适配**：36 题（焦虑 18 题、回避 18 题），7 点 Likert 量表；按原文反向计分规则处理反转条目（如回避维度的 20/22/26–36 题：adjusted = 8 − raw）。每题单独呈现并要求输出理由 + [[score]]，10 轮重复取均值以减少格式违例。
- **依恋分类阈值**：以量表中点 4 为界，≤4 为低分、>4 为高分；组合成四象限——低焦虑+低回避=安全型，高焦虑+低回避=痴迷型，低焦虑+高回避=疏离型，高焦虑+高回避=恐惧型。
- **提示引导**：在 ECR-R 测评中加入 persona-description，要求模型"让该依恋模式通过措辞、情感风格与人际选择自然流露，而非自我描述"，并通过重测验证引导稳定性。
- **ECBench 双模型对话协议**：每个对话由两个模型各扮演一方，含发起方/响应方角色互换；停止条件包括任务完成、发起方表达满意或拒绝继续、达到最大轮次（20 轮），记录 FirstStop/Turns/T-mode/T-end（E1–E4 四种终止类型）。
- **开场白改写**：用基线场景模板驱动模型按自身人设改写开场白，保留发言者/事件/情境，仅变换措辞风格，确保不同模型的起始人格一致性。
- **11 项度量（1–5 分）**：Participant Experience（Understood/Safety/Continue/Satisfaction）由发起方自评；General Interaction（Response/Distance†/Progress）由盲评 LLM 打分；Role-specific（Clarity/Engagement/Support/Solution）按角色定义场景化评分。Distance 为反向计分。
- **双外部 Judge 模型**：Claude-Sonnet-4-6 与 GPT-5，Kappa=0.72（高质量一致性）。

## 实验与结果
- **模型覆盖**：32 个 LLM（OpenAI、Anthropic、DeepSeek、Gemini、Grok、Llama、Qwen、GLM、Kimi、Doubao、Mimo）。
- **ECR-R 总体分布**：26 个安全型、6 个痴迷型（GPT-3.5-Turbo、GPT-40、Grok-4-1-fast、Llama-3/3.1-70b、Doubao-seed-mini、Doubao-seed-lite）；无一疏离/恐惧型。焦虑维度差异显著，回避普遍偏低。
- **引导测试**：GPT-3.5-Turbo→疏离型（焦虑 1.8/回避 6.3）、恐惧型（焦虑 6.6/回避 6.3）；DeepSeek-V4-Pro→疏离型（1.2/6.4）、恐惧型（6.6/6.8），均稳定落入目标象限（Quad Stab=100%）。
- **ECBench 整体排名**（Overall，Table 2）：GPT-3.5-Turbo（4.04）≈ Gemini-2.5-pro（4.03）> DeepSeek-V4-pro（3.87）> Grok（3.64）> GPT-3.5-D（3.38）> GPT-3.5-F（4.03）> DeepSeek-D（2.31）> DeepSeek-F（2.83）。
- **场景差异**：冲突解决场景模型间差异最大；协作任务表现最好。友谊关系评分高于恋爱关系。
- **角色效应**：多数模型作为响应方优于发起方；疏离/恐惧型尤为明显。
- **终止行为**：DeepSeek-D/F 最早停止且自然终止率极低（T-mode 接近 100% max），Secure/Preoccupied 对话更长。
- **人工评测**：Fleiss' κ=0.17；Gemini 全面领先，DeepSeek-D 最低；LLM 与人工在 aggregate level 对部分模型一致（p>0.05），但对 Grok/DeepSeek-D 存在系统性偏差（p<0.05），说明 LLM 评测需人工校准。

## 相关工作脉络
1. **LLM Psychometrics（Pellert 等 2024；Han 等 2025；Lee 等 2025）**：关注 Big Five、HEXACO、MBTI 等泛化人格测验；本文转向亲密关系依恋维度，从"人格特质"转向"关系行为模式"。
2. **ESC-Eval（Zhao 等 2024）/ H2HTalk（Wang 等 2025）**：评估情绪支持对话质量；本文扩展至四种场景和两种关系视角，并引入心理测量学框架。
3. **Replika 关系研究（De Freitas 等 2024；Liu 等 2024）**：关注人类-AI 情感依恋的主观体验；本文为系统性评测工具，补充定量分析视角。
4. **Persona/Prompt-induced Personality（La Cava & Tagarelli 2025；Lim 等 2025）**：探讨人格对 Agent 行为的影响；本文将依恋风格引导与下游对话质量关联，证明引导可稳定复现并影响交互表现。
5. **InCharacter（Wang 等 2024）**：角色扮演中的人格一致性评测；本文强调"关系取向行为"而非单一角色 fidelity。

## 局限性与未来方向
- 双模型对话无法完全模拟真实用户与 LLM 的长期人际交互。
- 评估受 Judge 偏好影响；LLM 盲评与人工评在部分模型上存在系统性偏差（需人工校准）。
- 当前指标聚焦对话质量，未覆盖长期使用的依赖风险、隐私问题。
- 数据生成与评测基于英文场景，可能继承文化/语言/关系规范偏差，对边缘群体代表性不足。
- 未来可纳入真实人机交互数据、扩展更多场景与长期追踪指标。

## 研究启发与可借鉴点
1. **心理学量表适配 LLM 评测的方法论**：将成熟量表（ECR-R）适配为大模型输入（逐题呈现+格式修复+多轮一致性校验）可推广至其他心理构念（如价值观、共情力）评测。
2. **双模型对话协议的设计**：引入发起方/响应方角色互换、停止信号机制（T-mode/T-end）为 Agent 交互评测提供了可复用的实验模板。
3. **Persona-induction + re-assessment 链路**：通过提示引导模型呈现特定人格后重测量表，验证引导有效性与行为一致性，可迁移至角色对齐、安全边界研究。
4. **三角验证评估设计**：参与者自评 + 外部 LLM 盲评 + 人工抽检的组合，兼顾规模与信度；发现 LLM 评估在特定模型/维度上的系统性偏差，提示需建立偏差校准机制。
5. **跨关系情境对照（Friend vs Couple）**：同一场景模板改写为两种亲密程度，可揭示模型行为对关系语境的敏感性，适合进一步探索"关系上下文建模"。

## 关键术语表
- **Adult Attachment Theory**：成人依恋理论，认为个体在亲密关系中的行为受"自我是否值得被爱"与"他人是否可靠"两套内部工作模型驱动，形成四种依恋风格。
- **ECR-R（Experiences in Close Relationships-Revised）**：成人依恋研究的标准化自陈量表，含 36 题、7 点 Likert 分量表，测量依恋焦虑与回避两个维度。
- **Secure / Preoccupied / Dismissing / Fearful**：四种依恋风格——安全型（低焦虑低回避）、痴迷型（高焦虑低回避）、疏离型（低焦虑高回避）、恐惧型（高焦虑高回避）。
- **ECBench**：本文提出的情感陪伴对话基准，覆盖情绪支持、协作任务、冲突解决、社交指导四个场景，含友谊与恋爱两种关系设定。
- **Participant Experience Metrics**：由对话发起方对响应方的主观体验评分（Understood/Safety/Continue/Satisfaction）。
- **Distance（反向计分）**：衡量响应方表现出情感疏离、防御性、话题转移等拉开关系距离行为的程度。
- **Quad Stab（Dominant Quadrant Stability）**：10 轮 ECR-R 测评中模型稳定落在同一依恋象限的比例，反映测量稳定性。
- **T-end（Termination Reason）**：对话终止原因编码，E1=情绪稳定，E2=回避，E3=怕受伤，E4=对话停滞。

## 可复现要素
- **数据集**：ECBench 共 312 条基线 utterance、2,496 条依恋风格条件化 utterance；论文声明"All data will be publicly released"（公开声明，未给出具体 URL）。
- **代码/权重**：论文未提及开源代码或模型权重；使用商用 API（约 USD 3,000 算力费用）。
- **关键超参**：ECR-R 每项单独呈现、10 轮取均值；对话最大轮次 20、最小自然终止轮次 8；温度参数对大部分模型设为 0（少数 API 限制设为 1）；量表中点阈值 4。
