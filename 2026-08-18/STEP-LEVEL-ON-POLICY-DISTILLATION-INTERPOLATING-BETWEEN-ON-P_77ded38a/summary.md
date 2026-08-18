---
title: "STEP-LEVEL-ON-POLICY-DISTILLATION-INTERPOLATING-BETWEEN-ON-P"
source: https://arxiv.org/pdf/2608.16333v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:55:05"
field: "大语言模型蒸馏与后训练"
keywords: ["on-policy distillation", "step-level supervision", "SFT-OPD interpolation", "black-box distillation", "reasoning", "agent tasks"]
innovations: ["提出SOPD，在学生轨迹的自然步骤起点生成教师目标，插值于SFT与OPD之间", "建立SFT与forward-KL OPD的理论联系框架", "无需教师logits的黑盒蒸馏支持"]
benchmarks: ["ALFWorld", "AIME24", "AIME25", "HMMT25-Feb", "HMMT25-Nov"]
---

# 论文速读：STEP-LEVEL ON-POLICY DISTILLATION: INTERPOLATING BETWEEN ON-POLICY DISTILLATION AND SUPERVISED FINE-TUNING

## 一句话总结
本文提出**Step-Level On-Policy Distillation (SOPD)**，通过将教师监督粒度从token级扩展到步骤级，在保留OPD on-policy状态分布的同时引入SFT式的连贯多token修正，在ALFWorld智能体和数学推理任务上均显著优于传统SFT与OPD。

## 研究问题与动机
1. **Token级OPD的碎片化修正缺陷**：标准OPD仅在学生轨迹的每个token位置查询教师分布，导致单次rollout中只能提供碎片化的单token修正，无法展开完整的正确修复路径。
2. **DAgger类方法的on-policy性质弱化**：现有DAgger风格方法通过插入教师响应改变学生生成的轨迹，破坏了训练状态分布与学生推理时的一致性。
3. **黑色盒蒸馏的实用性需求**：标准OPD需要访问教师logits，而SOPD仅需教师生成的文本响应，更适合黑盒蒸馏场景且每样本教师生成总量接近一个完整响应长度。

## 核心贡献（创新点）
1. **提出SOPD方法，插值于SFT与OPD之间**：在学生生成的完整轨迹上，于每个自然步骤起点由教师生成一个局部目标，兼具SFT的多token连贯监督与OPD的on-policy状态覆盖；与已有工作本质区别在于既不修改学生轨迹状态分布，也不依赖教师logits访问。
2. **建立SFT与OPD的理论联系框架**：证明SOPD在步长为整段响应时退化为SFT，在步长为单token时近似forward-KL OPD，为两种蒸馏范式的统一视角提供新认识。
3. **在Agent与Reasoning双领域验证通用性**：ALFWorld上将平均成功率提升13.4分（相对Vanilla OPD），数学推理四基准平均提升9.9分，且减少环境交互轮次。
4. **支持高效Black-box蒸馏与并行化**：教师调用只需生成单步输出，无需等待环境交互或工具执行，且各步骤教师查询可在学生轨迹完成后并行发出。

## 方法详解
**步骤划分与状态固定**：将学生轨迹$\tau_S = (s_1, s_2, \ldots, s_K)$按自然边界划分为步骤（数学推理用段落/Heading边界，ALFWorld用每个助手响应+解析动作），记录每个步骤前的学生前缀$h_k$，确保状态分布$d_k^{\pi_{\theta_r}}$由rollout学生固定。

**教师目标生成**：对每个$h_k$，教师独立生成$\tilde{s}_k \sim q_T(\cdot|h_k)$，目标仅以原始学生前缀为条件，不包含其他教师目标，保证各查询条件独立、可并行生成。

**Step-Balanced Loss**：
$$\mathcal{L}_{\mathrm{SOPD}}(B) = \frac{1}{|\mathcal{S}_B|} \sum_{(h_k, \tilde{s}_k) \in \mathcal{S}_B} \frac{1}{|\tilde{s}_k|} \sum_{j=1}^{|\tilde{s}_k|} -\log \pi_\theta(\tilde{s}_{k,j} | h_k, \tilde{s}_{k,<j})$$
内层对每目标token取平均防止长步骤主导，外层对所有步骤等权平均；支持packed attention mask实现并行高效训练。

**极限情况**：
- 一步极限（$K=1$）：退化为sequence SFT的teacher-forcing损失。
- 单token极限：等价于学生前缀上的采样forward-KL目标 $\mathbb{E}[-\log \pi_\theta(\tilde{y}_t|h_t^S)] = H(q_T) + \mathrm{KL}(q_T||\pi_\theta)$。

## 实验与结果
**ALFWorld Agent任务**（3B学生，7B RL教师）：
- Valid Seen：SOPD成功率84.29%，相对Vanilla OPD提升18.57分，交互轮次11.20（减少3.53轮）
- Valid Unseen：SOPD成功率82.09%，相对Vanilla OPD提升21.64分，交互轮次11.88（减少4.33轮）
- Hard：SOPD成功率10.74%，轮次28.15（最短交互轨迹）
- SOPD在Seen/Unseen上均优于TCOD（F2B方向），是student方法中最佳

**数学推理任务**（4B学生，4B RL Math教师）：
- AIME24：71.8%（相对OPD +9.9分）
- AIME25：67.4%（+10.4分）
- HMMT25-Feb：38.9%（+6.4分）
- HMMT25-Nov：52.7%（+13.1分）
- Avg₄：57.7%（相对OPD +10.0分），超越继续RL训练100步的对照组（46.9%）

**训练动态**：随着训练推进，学生生成的自然推理步骤数显著增加，每步长度稳定，教师监督token数从29增至45，体现模型逐步学会结构化推理。

## 相关工作脉络
1. **TRD（Trajectory-Refined Distillation）**：通过重写完整学生轨迹构建修复路径，属于off-policy修正；SOPD则保持学生轨迹不变，仅在每个自然步骤起点生成局部目标，维持更强的on-policy性质。
2. **TCOD（Temporal Curriculum On-Policy Distillation）**：使用时间课程控制多轮轨迹中学生/教师的控制区间，教师动作会进入环境执行改变轨迹；SOPD不插入教师文本到后续步骤的上下文中。
3. **OEC（On-Policy Expert Correction）**：在学生rollout中途切换至专家控制并SFT奖励过滤的专家后缀；SOPD覆盖完整轨迹的每一步而非仅后半段。
4. **G-OPD/ExOPD**：利用reward extrapolation允许学生超越单一教师；SOPD无奖励信号，纯依赖教师生成文本进行distillation。
5. **Black-box OPD（如GAD）**：通过generator-discriminator游戏实现黑盒蒸馏；SOPD直接查询教师生成响应，更简单实用且成本接近SFT。

## 局限性与未来方向
1. **步骤划分的语义依赖性**：数学推理依赖自然边界（段落/Heading），复杂推理可能缺乏明确步骤结构，需设计更鲁棒的自动划分策略。
2. **教师质量上限约束**：SOPD性能受教师模型能力限制，无法像ExOPD那样通过reward extrapolation突破教师边界。
3. **未探索自我蒸馏场景**：当前需独立教师模型，可与OPSD等self-distillation方法结合探索无额外教师成本的版本。
4. **多模态/代码生成扩展**：未在视觉或多轮对话场景验证，通用性有待进一步检验。

## 研究启发与可借鉴点
1. **Step-balanced归一化设计**：先对每个目标内部token平均再对所有步骤平均的loss结构，可有效防止长步骤主导训练，可迁移至任意序列生成任务的课程学习。
2. **Packed Attention Mask工程实现**：将学生轨迹、多个教师目标打包为单序列并设计掩码，实现并行高效训练，对长上下文训练有参考价值。
3. **Black-box友好的distillation思路**：仅需教师生成文本而非logits，大幅降低蒸馏部署门槛，适合工业场景中使用闭源API教师。
4. **SFT与OPD统一视角**：通过调整步长粒度在两者间插值，为后续设计自适应粒度蒸馏提供理论框架。
5. **训练动态监控指标**：跟踪学生步骤数、每步长度、教师监督长度的变化趋势，可有效诊断模型是否学会结构化推理，可作为通用训练诊断工具。

## 关键术语表
- **On-Policy Distillation (OPD)**：学生在自己生成的token位置上查询教师分布进行蒸馏的训练范式，确保训练状态与推理时一致。
- **Step-Level On-Policy Distillation (SOPD)**：本文提出的方法，在学生轨迹的每个自然步骤起点查询教师生成一个多token目标，插值于SFT与OPD之间。
- **Trajectory-Refined Distillation (TRD)**：通过重写完整学生轨迹构建修复路径的distillation方法，与SOPD的局部步骤修正形成对比。
- **TCOD (Temporal Curriculum On-Policy Distillation)**：使用时间课程控制多轮agent轨迹中学生/教师控制区间的distillation方法。
- **Step-balanced Loss**：先对每个教师目标内部token取平均、再对所有步骤等权平均的损失函数，防止长目标主导训练。
- **Packed Attention Mask**：将学生轨迹与多个教师目标打包为单序列并设计因果掩码，使各教师目标仅能 attend 到对应学生前缀的高效训练机制。
- **Black-box Distillation**：仅需访问教师生成文本而无需logits的蒸馏场景，SOPD天然支持此类设置。
- **Forward-KL vs Reverse-KL**：SOPD在单token极限下对应forward-KL $\mathrm{KL}(q_T||\pi_\theta)$，而标准OPD常用reverse-KL。

## 可复现要素
- **数据集**：ALFWorld（公开环境）、DeepMath筛选版（难度≥6，57K样本，作者自行筛选后公开）
- **代码/权重**：论文附录A提供完整训练超参，提交源包含figure生成脚本与聚合记录；模型权重未明确声明开源
- **关键超参**：
  - 数学推理：学习率$2\times10^{-6}$，batch size 1024 prompts，10 updates，最大响应32768 tokens
  - ALFWorld：学习率$10^{-6}$，batch 64 experiences，staleness 2，响应上限512 tokens
  - 教师解码：greedy（temperature=0）
  - 评估：temperature=0.4（ALFWorld）/1.0（数学），各32 samples
