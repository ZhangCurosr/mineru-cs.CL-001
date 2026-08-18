---
title: "Beyond-Retrieval-Query-Conditioned-Reuse-of-Long-Horizon-Age"
source: https://arxiv.org/pdf/2608.12847v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:56:20"
---

# 论文速读：Beyond-Retrieval-Query-Conditioned-Reuse-of-Long-Horizon-Age

## 一句话总结
本文指出长程Agent记忆的核心瓶颈已从“检索”转移到“后检索复用”阶段，并提出QCR（Query-Conditioned Reuse）格式：在固定检索结果与执行条件的边界下，将历史轨迹转化为包含工作流不变量、需重新锚定的绑定、适用条件与验证护栏的紧凑支持笔记。在WebArena、WorkArena与AppWorld共2,391个目标实例上，QCR以48.9%的在线Token开销实现62.3%的平均Success，较直接注入完整轨迹提升10.7个百分点。

## 研究问题与动机
- **检索质量≠复用效用**：现有记忆评测多止步于“能否检索/定位历史”，但长轨迹进入上下文后并不自动转化为可执行的操作计划。
- **源绑定陈旧导致直接回放失效**：成功轨迹携带大量源特定值（用户、路径、日期、环境状态等），直接注入会使Agent复制过时参数，产生Stale-Binding Error。
- **长轨迹的价值随长度与偏移量衰减**：直接注入原始轨迹在轨迹变长或源-目标绑定发生较大变化时，效用急剧下降，而通用摘要又过度压缩必要前置条件。
- **评测边界混淆**：存储、检索、选择、复用常被混在一起评估，缺乏隔离“候选选择”与“目标条件化支持生成”的控制实验。

## 核心贡献（创新点）
- **提出后检索复用评估框架**：固定冻结库、候选集、重排序、执行模型、解码策略与工具预算，仅改变传入Agent的支持对象表示，从而孤立测量复用环节的真实增益。
- **设计QCR四字段紧凑支持格式**：显式分离可迁移工作流、必须重获取的绑定、可放弃复用的适用条件与验证护栏，避免直接复制源值。
- **验证重排序显著提升可用记忆选择率**：轻量Ranker将Reusable-Memory准确率从Top-1检索的78.9%提升至94.8%，末端Success仅落后Oracle 1.8个百分点。
- **系统刻画轨迹长度与绑定偏移对复用的影响**：证明Full Trajectory在Very Long/ Large Shift场景下仅保留15.8%/8.2%的短期效用，而QCR仍保留60.3%与67.9%。
- **构建跨环境统一冻结库与Binding-Aware目标集**：汇集623条经原生验证器接受的成功轨迹，自动改写生成2,391个目标实例（平均每条源轨迹对应3.84个目标变体），提供可审计的评估账本。

## 方法详解
- **冻结库与目标构造**：从WebArena、WorkArena、AppWorld各取中等难度任务族，仅保留通过原生Checker的最终成功Rollout，构建623条统一混合库。针对每条源轨迹$\tau_s$，保留其目标工作流意图，改写1~多个Binding（实体/路径/日期/初始状态等），生成No/Small/Medium/Large四级偏移的目标任务，共2,391个。
- **固定检索与重排序**：使用BGE-M3基于源指令与轨迹描述符返回Top-5候选$Z_t$；随后由DeepSeek-V4-Pro Ranker基于紧凑描述符与目标查询$q_t$选定单条轨迹。$Z_t$与选定记录在四种条件间严格共享。
- **QCR支持合成**：给定$Z_t$中的选定轨迹、$q_t$与$o_{t,0}$，QCR Writer（温度0，确定性）生成四字段笔记$r_t$：
  1. **Workflow Invariant**：仅保留当前任务仍适用的动作模式与决策顺序（如“检查→校验→修改→验证”）。
  2. **Bindings to Re-obtain**：列出源中出现的实体/路径/标识/参数等依赖项，明确要求Agent从当前目标观测或工具调用中重新获取，禁止直接抄写源值。
  3. **Applicability Conditions**：记录源流程生效的前置约束与分支条件；若目标状态不满足则允许放弃复用。
  4. **Verification Guardrail**：提取源轨迹中用于确认完成的关键检查步骤，供目标侧校验最终状态。
- **对比条件**：NO MEMORY / GENERIC SUMMARY（离线源端摘要，长度与QCR对齐） / FULL TRAJECTORY（直接注入原始轨迹） / QCR。所有条件共享相同Target Query、初始状态、模型（DeepSeek-V4-Pro）、解码配置、工具预算与验证器。在线Token计入基础提示、传入记忆、支持合成输入输出与Actor输出，排除离线库构建成本。

## 实验与结果
- **数据集与模型**：WebArena、WorkArena、AppWorld；全链路使用DeepSeek-V4-Pro。每目标3次Seed-Matched运行取平均。
- **端到端性能（Table 1）**：QCR平均Success达62.3%，较Full Trajectory高10.7点，较No Memory高约23~25点；三个环境均保持Success与Milestone双第一。QCR在线Token为9.4k，较Full Trajectory的18.4k减少48.9%，API调用亦为最低。
- **检索与选择诊断（Table A10）**：Top-5 Reusable Coverage达97.8%；重排序后Final Paired Accuracy 91.7%，Final Reusable Accuracy 94.8%；仅5.2%的选定记忆无实际用处。直接Top-1检索成功率仅56.1%，随机选Top-5仅44.8%，Ranker选择逼近Oracle（64.1%）仅差1.8点。
- **长度敏感性（Table 2）**：随轨迹变长，Full Trajectory效用从Short的+18.4骤降至Very Long的+2.9；QCR从+21.9降至+13.2，仍保留短轨迹60.3%的增益。
- **绑定偏移敏感性（Table 3）**：Large Shift下Full Trajectory仅+2.2，QCR仍+20.1；QCR的陈旧绑定错误率从46.9%降至10.9%，正确重绑定率从31.7%升至77.8%，说明显式提示“需重获取”是抑制源值污染的关键。

## 相关工作脉络
- **长上下文与检索型记忆**（MemGPT、MemoryBank、HippoRAG、Mem0、A-MEM、RAG）：侧重存储、索引与上下文投喂，本文不改进存储与检索本身，而是隔离并度量“检索后如何用”。
- **轨迹与程序经验提取**（Reflexion、Voyager、ReAct、ExpeL、Agent Workflow Memory、SAM、Agentic Memory、OCR-Memory）：多从历史中提取Reflection/Skill/Script；本文保留原始已验证轨迹，仅在交付层做目标条件化转换，验证“形式压缩不如边界约束有效”。
- **长程Agent评测基准**（AgentBench、WebArena、WorkArena、AppWorld、τ-bench、Mind2Web、OSWorld、SWE-bench）：传统基准侧重单次执行打分；本文引入跨环境统一库与Binding-Aware改写，直接度量复用对当前任务的实际帮助。
- **长上下文评测**（LongBench、LoCoMo、LongMemEval、MemBench）：聚焦Retention与Access；本文指出Access≠Utility，主张将验证器与在线成本纳入记忆效用评估。

## 局限性与未来方向
- 仅评估已成功Rollout，未覆盖部分失败轨迹、自然重复任务历史、多记忆组合与开放式记忆获取场景。
- 未拆解QCR四个字段各自的独立贡献，仅验证整体干预的有效性。
- 在线Token节省不意味着所有任务都应最小化Token；安全敏感目标可能需要更多验证步骤。
- 仅测量验证器判定的Completion，未评估复用可能引发的不可逆副作用或策略违规。
- 未来可替换Embedding检索器、学习条件化Reuse Policy，并在保持边界会计的前提下扩展至多记忆与失败重试场景。

## 研究启发与可借鉴点
- **控制变量评测范式**：固定候选集、模型、解码与工具预算，仅改变支持对象表示，是剥离检索噪声、单独测量复用收益的可靠实验设计，可直接迁移至其他记忆系统评测。
- **Binding-Aware任务改写机制**：通过控制源-目标绑定偏移程度（No/Small/Medium/Large）量化迁移衰减，为记忆系统的“跨域/跨实例可迁移性”提供标准化压力测试方法。
- **四字段紧凑支持设计**：显式要求“保留什么、重获什么、何时放弃、如何验证”比单纯压缩或原样注入更有效，可推广至Skill/RAG记忆的条件化交付层。
- **非重叠在线Token计量**：将Base Prompt、传入记忆、支持合成、Actor输出分段统计，避免将“上下文窗口增大”误判为“免费增益”，为Agent效率评测提供更诚实的口径。
- **跨环境统一冻结库+去标签检索**：混合Web/API/App轨迹但不暴露环境标识，可检验工作流级泛化而非词汇级匹配，对多工具Agent的长期记忆库构建具有参考意义。

## 关键术语表
- **QCR (Query-Conditioned Reuse)**：基于目标查询与初始状态对选定历史轨迹进行条件化转换，生成紧凑的复用支持笔记。
- **Binding（绑定）**：智能体执行时必须锚定到当前上下文的实体值（如用户、路径、日期、标识符、参数等），源值直接沿用会导致执行错误。
- **Workflow Invariant（工作流不变量）**：从历史轨迹中提取的、可迁移到目标任务的动作模式与决策顺序。
- **Applicability Conditions（适用条件）**：明确源流程生效的前置约束与分支条件，允许目标状态不满足时主动放弃复用。
- **Verification Guardrail（验证护栏）**：记录源轨迹中确认任务完成的关键检查步骤，供目标侧执行完毕后校验。
- **Stale-Binding Error（陈旧绑定错误）**：Agent直接复制源轨迹中的特定值用于目标任务，与当前观测或查询冲突导致的执行失败。
- **Frozen Bank（冻结库）**：由
