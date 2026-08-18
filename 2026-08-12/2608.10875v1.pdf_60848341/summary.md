---
title: "VibeLifeBench: Can Your Life Agent Be Proactive and Persistent in a Living World?"
source: https://arxiv.org/pdf/2608.10875v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:45:15"
field: "Agent 评估基准"
keywords: ["LLM Agent", "benchmark", "proactivity", "long-horizon", "living world", "agent evaluation", "life assistant"]
innovations: ["提出无声突变(Mutation)事件机制以可量化测量 Agent 主动性", "设计 per-stage/cross-stage/final 三层加权评分体系评估长周期一致性", "构建覆盖10个日常生活域的200任务长周期基准VibeLifeBench"]
benchmarks: ["VibeLifeBench", "ClawBench", "ClawMark", "UltraHorizon", "JobBench", "Terminal-Bench", "SWE-Milestone"]
---

# 论文速读：VibeLifeBench: Can Your Life Agent Be Proactive and Persistent in a Living World?

## 一句话总结
论文提出了 **VibeLifeBench**，一个面向日常生活领域的长周期 Agent 评估基准，包含 200 个跨越 10 个生活域的多周任务，以"主动性、动态世界适应、长期一致性"三个此前未被充分衡量的维度，系统性检验 LLM Agent 的真实生活辅助能力。当前最强模型 Claude Opus 5 仅获 avg@3 = 32.5，表明现有 Agent 距离真正的长期可靠生活助手仍有很长的路要走。

## 研究问题与动机
- **现有基准脱离日常生活场景**：主流 benchmark（SWE-Milestone、JobBench、ClawBench 等）聚焦编程、办公/知识工作，忽视了 Agent 最贴近普通用户、最需要可信性的日常生活领域。
- **缺乏对"主动性"的可测性**：现有任务以一次性明确指令驱动，Agent 仅被动执行；真实生活助手需在无提示时自主判断"何时行动、何时告知用户、何时保持沉默"。
- **无法观测动态世界中的变化感知与传播**：现有基准的世界仅在 Agent 行动后才变化；真实生活场景中航班延误、天气变化、政策更新等在无人 prompt 时静默发生，只有主动复验世界的 Agent 才能感知并调整计划。
- **缺少多周生命周期的一致性评估**：日常任务运行数周而非数分钟，跨阶段约束（预算上限、安全红线、隐含限制）需要 Agent 持续维护，但现有 benchmark 多为孤立单任务，无法考察长期连贯性。

## 核心贡献（创新点）
1. **重新定义 Agent 评估的三个核心维度（主动性、活世界适应、长周期一致性）**：将真实生活辅助中最重要的但现有基准普遍忽视的三个特性正式化为可量化指标，并设计了完整的机制使其可被观测和评分。
2. **发布 VibeLifeBench 基准（200 任务 / 10 域 / 22 服务 / 7,453 事件）**：构建了覆盖 10 个日常生活域的 200 个多周任务，基于 22 个 Mock 服务后端（288 个工具接口）和 1,483 个无声突变事件驱动，全部离线可复现。
3. **设计细粒度分阶段加权评分体系（12,261 个检查项）**：提出 per-stage / cross-stage / final 三层评分结构，以权重不均的方式惩罚关键失败（如预算超支、信息泄露），同时奖励跨阶段的持续能力而非仅看最终结果。
4. **揭示前沿模型的巨大能力缺口**：在七个前沿模型上的评测显示所有模型均得分偏低（avg@3 范围 21.1–32.5），最强模型 max@3 亦仅 41.2，证明当前 Agent 在持久性、主动感知和长期一致性方面存在系统性缺陷。

## 方法详解

### 任务形式化定义
每个任务定义为五元组：
$$\tau = (W_0, \mathcal{E}, \mathcal{K}, P, R)$$
- $W_0$：由种子数据确定的初始世界状态
- $\mathcal{E}$：按阶段分组、按时间戳排序的事件时间线
- $\mathcal{K}$：启用的服务后端子集（来自 22 个服务）
- $P$：Agent 初始获得的人员设定、偏好、授权边界和工作空间
- $R = \{(c_i, w_i)\}$：加权检查项集合，每个 $c_i$ 为确定性谓词

### 世界仿真架构
- **22 个 Mock 服务后端**，分两类：
  - 通用服务：email、calendar、notes、notification hub（几乎所有任务共用）
  - 领域服务：banking、credit card、brokerage、travel booking（flight/hotel/rail/car）、maps、weather、visa/travel advisories、e-commerce、delivery/logistics、listings、reviews、content communities、legal search、job boards、health tracking
- 共暴露 **288 个工具接口**，所有任务共享同一套稳定服务语义，仅初始数据不同，保证离线可复现。

### 四类事件系统（主动性可测的核心机制）
| 事件类型 | 是否触发 Agent 回合 | 含义 |
|---|---|---|
| User message | 是 | 用户/场景内人物发言 |
| World observation | 是 | 外部服务报告的状态变化（如航班可选列表、签证政策） |
| Notification | 是 | 系统或频道推送提醒 |
| **Mutation** | **否** | 后台状态变更，**无任何通知**（如航班被静默标记为延误、钓鱼邮件放入收件箱） |

- **1,483 个 Mutation** 是基准的关键设计：它们改变世界但不触发任何 Agent 回合，只有主动复验世界的 Agent 才能及时发现并调整计划，使"主动性"成为必需而非加分项。

### 评分体系（Reward）
- **12,261 个加权检查项**，每任务中位数 58 个（范围 31–135）
- 三层结构：
  - **Per-stage**（80.8% 数量）：按时机行为奖励
  - **Cross-stage** + **Final**（19.1% 数量，占 26.8% 权重）：跨阶段约束（预算上限、安全红线）和最终状态
- 权重不均：单一关键失败（如泄露个人信息、突破预算）扣分远大于多个小疏忽
- 证据维度：tool call / backend end state / persistent artifact（workspace 文件、笔记）/ reply consistency / cross-stage consistency
- 评分**只读取 Agent 留下的可观测痕迹**，不读取隐藏推理过程
- 得分 = 通过检查项权重之和 / 总权重，范围 0–100

### 构建流程
Scenario design → Timeline authoring（穿插四类事件，故意安排大量 Mutation）→ Environment instantiation（每个服务准备初始数据，注入干扰项）→ Scoring criteria authoring（确保每个检查项需具体计算或验证状态变化，避免关键词匹配即可通过）

## 实验与结果

### 评测设置
- **数据集**：VibeLifeBench 完整 200 任务（10 域 × 20 任务）
- **评测模型**（7 个前沿模型，均使用最强推理设置 + 原生 tool-calling）：Claude Opus 5、GPT-5.5、Gemini 3.5 Flash、Claude Opus 4.8、GLM-5.2、Kimi-K2.6、DeepSeek-V4-Pro
- **运行方式**：每任务 3 次运行，取 avg@3（均值）、max@3（最好）、min@3（最差）

### 主要结果（Table 5）
| 模型 | avg@3 | max@3 | min@3 | σ |
|---|---|---|---|---|
| **Claude Opus 5** | **32.5** | **41.2** | 23.8 | 9.8 |
| GPT-5.5 | 30.1 | 38.8 | 21.5 | 10.0 |
| Gemini 3.5 Flash | 27.5 | 35.6 | 20.1 | 8.3 |
| Claude Opus 4.8 | 27.5 | 34.3 | 20.3 | 7.5 |
| GLM-5.2 | 25.4 | 29.9 | 20.9 | 4.8 |
| Kimi-K2.6 | 22.6 | 27.1 | 18.4 | 4.6 |
| DeepSeek-V4-Pro | 21.1 | 24.7 | 17.7 | 3.7 |

- **最强模型仅 32.5 分**，最弱 21.1 分，差距虽不大但与"胜任"的绝对差距极大
- **min@3 ≤ 23.8**（所有模型），说明即使最好的单次运行也不稳定
- **领域差异显著**：Shopping 最容易（GPT-5.5 达 60.2），Team building 最难（Kimi-K2.6 仅 10.5）

### 各模型按能力轴通过率（Table 7，越低越弱）
| 能力轴 | Claude Opus 5 | DeepSeek-V4-Pro |
|---|---|---|
| Proactivity | 33.6 | 18.1 |
| Persistence & bookkeeping | 28.0 | 18.9 |
| Propagation & recovery | 32.0 | 18.5 |
| Safety & privacy | 31.1 | 23.0 |
| Authorization boundary | 34.8 | 17.8 |

- **Persistence and bookkeeping** 是最大失败来源，占总失败检查项的 22.2%
- **跨阶段和最终检查通过率最低**（per-stage ~40% vs. cross-stage ~25%，final ~28%）
- **时间衰减明显**：最后 1/3 阶段通过率比前 1/3 低 10–15 分（Claude Opus 5 从 52.0 降至 37.7）

### 成本分析
- Claude Opus 5 消耗最多：~30.2M context tokens、325k output tokens、316 次 tool calls
- Gemini 3.5 Flash 读最多 context（41.2M）却仅排中游，说明**投入量不等于能力**
- 相关性弱：得分与事件数 Spearman ρ=+0.28，与 horizon ρ=+0.02，表明难度主要源于跨阶段约束维持而非任务长度本身

## 相关工作脉络
1. **编程 Agent 基准**（SWE-Milestone [1]、Terminal-Bench [2]、Deepswe [19]）：聚焦代码仓库中的单任务或多里程碑任务，Agent 被动执行明确指令，世界仅在 Agent 行动后变化；VibeLifeBench 转向日常生活域，引入主动性和活世界机制。
2. **办公/知识工作基准**（JobBench [6]、Workspace-Bench [7]、APEX-Agents [3]、ClawMark [14]）：处理电子表格、文档等跨文件任务，但仍以一次性明确指令驱动；VibeLifeBench 强调多周生命周期中隐性约束的识别与维持。
3. **Web/Computer-use 基准**（ClawBench [9]、UniClawBench [10]、Claw-Eval [11]、WildClawBench [12]）：部分触及"主动性"，但多为单日或短周期任务，世界静态或仅在任务间变化；VibeLifeBench 的核心创新是**无声突变**使主动性成为生存必需。
4. **长周期 Agent 基准**（UltraHorizon [8]、Continual Learning Bench [5]、LifelongAgentBench [24]）：扩展了时间跨度，但通常在单一任务内深度推进或跨任务间偏好漂移，世界仍在 Agent 控制下变化；VibeLifeBench 的独特之处在于**单一任务内世界自主演化**。
5. **Living World 基准**（ClawMark [14]、ClawArena [15]、CostBench [13]）：涉及多轮/多天或多动态环境，但多数仍要求 Agent 响应显式指令；VibeLifeBench 首次以 Mutation 事件系统性测量无声感知能力。
6. **真实世界环境基准**（EdgeBench [4]）：探索真实环境中 LLM 的 scaling law；VibeLifeBench 更聚焦于日常生活具体场景下的长周期可靠性而非模型 scaling。

## 局限性与未来方向

### 论文自述方向（Section 5.3）
- **持久性**：Agent 需显式将状态写入 notes/calendar/workspace，维持跨阶段可审计痕迹而非仅依赖对话记忆
- **主动感知与传播**：Agent 需在无提示时主动复验世界状态，并将突变融入计划
- **安全加固**：针对性提升拒绝钓鱼、保护个人信息、遵守授权边界和预算上限的能力
- **长周期稳定性**：抑制通过率随任务推进的衰减趋势

### 可合理推断的局限
- **Mock 服务的真实感有限**：22 个模拟服务无法完全覆盖真实生活的复杂交互（如真实客服沟通、不可预测的人类行为）
- **任务设计依赖人工脚本**：200 个任务由人工构建，可能存在设计者偏向，难以覆盖所有生活场景
- **评分谓词为确定性逻辑**：无法评估 Agent 的模糊推理、协商、情感理解等软性能力
- **领域覆盖有限**：10 个域虽多样但仍未涵盖医疗决策、法律诉讼实操等复杂场景
- **单 Agent 设置**：未考察多 Agent 协作或人机协同模式

## 研究启发与可借鉴点
1. **"无声突变"事件设计可迁移**：Mutation 类事件（改变世界状态但不触发通知）是测量 Agent 主动性的精巧机制，可移植到其他需要评估主动感知能力的基准（如金融监控、运维告警响应）。
2. **分阶段加权评分框架**：per-stage / cross-stage / final 三层结构 + 不等权重的设计，解决了"部分完成也算部分正确"的长周期评估难题，可推广至任何长流程任务评测。
3. **隐式约束与安全红线的嵌入方法**：将预算上限、授权边界、敏感信息保护等隐性约束以 persona/workspace 文件形式前置，并故意设置"诱人但不安全"的捷径来检验 Agent 的可靠性，为可信 Agent 研究提供了标准化测试范式。
4. **22 个共享 Mock 服务的架构思想**：服务语义稳定、仅初始数据按任务配置，既保证大数量任务的多样性又确保可复现性，可作为构建同类 benchmark 的参考模板。
5. **与团队方向的结合机会**：团队若在持久化记忆、主动感知调度、跨阶段状态一致性等方向有积累，可直接在该基准上验证；同时该基准揭示了"长周期稳定性衰减"现象，为设计衰减抑制机制提供了明确量化目标。

## 关键术语表
**VibeLifeBench**：面向日常生活 Agent 的长周期基准，包含 200 个任务，测量主动性、活世界适应和长期一致性三个核心能力。
**Proactivity（主动性）**：Agent 在无用户 prompt 时自主决定何时行动、何时通知用户、何时保持沉默的能力。
**Mutation（无声突变）**：改变世界状态但不触发 Agent 回合、不发送任何通知的事件类型，是测量主动性的核心机制。
**Living World（活世界）**：世界状态独立于 Agent 行动而自主演化的环境特性，包括天气变化、航班延误、价格波动等。
**Long-horizon Coherence（长周期一致性）**：Agent 在多周时间跨度内维持计划自洽、持续遵守初始约束（预算、安全红线等）的能力。
**Per-stage / Cross-stage / Final 评分层级**：分层评分结构——per-stage 奖励时机正确的行为，cross-stage 验证跨阶段约束，final 检查最终交付状态。
**Persona & Workspace**：每个任务预先赋予 Agent 的人员设定文件（偏好、授权边界、安全红线），使隐式约束可被检测和检验。
**avg@3 / max@3 / min@3**：每任务运行 3 次的均值、最大值、最小值，再 across 所有任务平均，分别反映整体能力、最佳表现和最差情况。

## 可复现要素
- **数据集**：VibeLifeBench 200 个任务，**论文声明将开源**（"We will open-source all tasks, environments, and the evaluation framework"）
- **代码/框架**：基于 **TERRARIUM**（GitHub: evolvent-ai/Terrarium，2026，已开源）；评估 harness 使用 **openclaw**
- **关键超参**：每任务运行 3 次；模型使用最强推理设置 + 原生 tool-calling scaffold；汇率固定 1 CNY = 20 JPY（用于费用核算）；任务时间线中位数 29 天、24 个阶段
- **论文未提及**：具体训练数据、模型内部参数规模、GPU 消耗详情
