---
title: "Executable-Code-Knowledge-Code-as-a-Native-Validation-Carryi"
source: https://arxiv.org/pdf/2608.16295v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:38:26"
field: "AI for Software Engineering / 编程代理知识表示"
keywords: ["AI coding agents", "executable code knowledge", "validation evidence", "code representation", "freshness checking", "patch impact analysis", "agent context"]
innovations: ["提出 ECKU：将选定代码单元直接表示为携带源绑定、契约、可执行证据与新鲜度状态的验证型知识对象", "揭示显式可执行证据对精确测试选择器恢复的决定性贡献（9/11 vs 1/11，p=0.0078）", "构建源-知识-契约-证据四维指纹的新鲜度检测机制，在 50 正向/17 负向扰动下达到 1.000 敏感度与特异性"]
benchmarks: ["python-dotenv / python-slugify / tomli 三仓库 26 受控补丁", "合成企业级三层服务（orders/pricing、subscriptions/entitlements、refunds/risk）18 补丁", "Qwen2.5-Coder-32B-AWQ / Qwen2.5-7B-Instruct 本地评测"]
---

# 论文速读：Executable-Code-Knowledge-Code-as-a-Native-Validation-Carryi

## 一句话总结
论文提出 **Executable Code Knowledge (ECK)**，将选定的高价值代码单元直接改造为携带领域语义、可执行证据、契约、溯源与新鲜度状态的验证型知识对象（ECKU），使 AI 编程代理无需仅依赖外部检索即可精准定位验证命令与变更影响；实验显示 ECKU 在 26 个受控补丁任务中实现 100% 精确变更行影响准确率，精确验证选择召回 9/11，并与检索/规则形成互补混合架构。

## 研究问题与动机
- **问题本质**：AI 编程代理处理仓库级问题修复时，仅需要"相似的代码片段"已不够，更需要知道哪段行为是业务规则、哪些测试是权威证据、哪个 API/数据关系受影响、以及上下文是否仍然新鲜。
- **现有方法不足**：当前系统通过检索、摘要、静态图、规则或反向规范等"事后提取"方式把普通代码转换为外部知识对象，这类提取管线存在**权威性缺失**——检索到的文本可以提示"去哪看"，但通常不声明"什么证据验证了该规则"；静态图能暴露结构，但不能区分领域不变量与偶然调用边；生成式摘要与逆向规范仍需要追溯、验证和保鲜。
- **本文追问**：代码本身能否**直接**表示 AI 代理可用的知识（而非仅作为被挖掘的文本）？特别是业务规则、安全校验、数据变换、API 契约等高价值单元。
- **动机案例**：一个简单的 `vip_large_order_discount` 函数，代理无法判断它是业务规则还是偶然实现细节、前置条件是什么、归属哪个 API/数据字段、应跑哪些测试、以及已有的文档是否已过期。ECKU 通过源绑定+可执行证据解决这一问题。

## 核心贡献（创新点）
1. **问题框架化**：首次明确将"代码直接作为 AI 代理可用知识表示"问题化，要求代码单元能同时表达计算、机器可读语义、证据、关系与新鲜度。
2. **ECKU 表示定义**：提出 ECKU = ⟨I, S, B, C, E, R, P, V, Q⟩，一次性集成身份、领域语义、可执行体、契约、证据、关系、溯源、验证状态与查询接口，区别于外部图表或摘要的"单一视图"表示。
3. **Python 原型与工具链**：实现基于装饰器的代码本地编写、manifest 导出、证据执行、精确变更行影响计算、AST 边界新鲜度检查、代理投影生成，以及可机器/审阅者读的补丁证据报告。
4. **机制级评测**：在 3 个真实仓库与 26 个受控补丁任务上，系统性评估显式证据恢复、独立标注的精确变更行影响（precision/recall/F1=1.000）、补丁审查投影保真度与新鲜度敏感度/特异性（均为 1.000/1.000），揭示"检索负责覆盖、ECK 负责源绑定与验证治理、投影负责交付"的混合架构优势。

## 方法详解
- **ECKU 结构**：
  - **I（身份）**：稳定唯一的标识符（如 `pricing.discount.vip_large_order`）。
  - **S（语义）**：领域语义，如业务规则名、阈值、折扣率等结构化属性。
  - **B（可执行体）**：绑定到原始函数/代码段；保留 Python AST 边界（lineno/end_lineno）以支持精确影响分析。
  - **C（契约）**：precondition/ensures 等约束（如 `user.level == 'VIP'`、`order.amount > 100` → `result.rate == 0.9`）。
  - **E（证据）**：一条或多条可执行命令（如 `pytest tests/test_pricing.py::test_vip_large_order_discount`），携带 id、kind、command 与 claim。
  - **R（关系）**：关联到 API 端点、数据实体、其他单元或文档（经规范化与规范序处理）。
  - **P（溯源）**：作者、时间、提交等元数据。
  - **V（验证状态）**：由哈希与执行结果维护，包含 fresh/supported/valid 判定。
  - **Q（查询接口）**：`build / query / show / validate / freshness / impact / context`。

- **验证与新鲜度判定**：
  - 四个关键指纹：`SourceHash = H(source(B))`、`KnowledgeHash = H(I, S, R, relevant(P))`、`ContractHash = H(C)`、`EvidenceHash = H(E)`。
  - **Fresh(u, t)**：四个哈希在时刻 t 均等于上次验证时的值。
  - **Executable(u)**：E 中所有携带可执行命令的 evidence 集合。
  - **Supported(u)**：Executable(u) 非空且每条证据命令 `Run(e)` 通过。
  - **Valid(u)**：Fresh(u) ∧ Supported(u)。明确区分"相关性"与"有效性"。

- **系统五层架构**：
  1. **Authoring Layer**：`@knowledge_unit`、`@contract`、`@evidence` 装饰器保持原始函数作为源跨度。
  2. **Discovery & Manifest Layer**：`eck build` 扫描仓库，导出 `.eck/knowledge_units.jsonl`、`.eck/evidence.jsonl`、`.eck/validation_state.json`。
  3. **Validation Layer**：执行证据命令，记录状态、各字段哈希、过期原因与时间戳。
  4. **Freshness Layer**：对比当前与持久化指纹，输出 `never_validated / source_changed / knowledge_changed / contract_changed / evidence_changed` 等细粒度原因。
  5. **Agent Context Layer**：`eck context <repo> "<task>"` 生成含意图、符号、源跨度、契约、可执行证据命令与关系的紧凑型上下文包，供代理消费。

- **关键设计选择**：
  - 装饰器不参与 source fingerprint，避免无谓触发知识变更；但知识哈希显式包含语义、关系等，能捕获代理-facing 语义漂移。
  - 关系顺序规范化消除"相同内容不同排列"导致的误报。
  - AST 边界精确到函数定义起止行，避免 whole-hunk 方案在边界处的假阳性。

## 实验与结果
- **数据集与仓库**：
  - `python-dotenv`（6 ECKUs，6 带证据，10 个受控补丁）
  - `python-slugify`（5 ECKUs，4 带证据，8 个受控补丁）
  - `tomli`（6 ECKUs，6 带证据，8 个受控补丁）
  - 合计 17 个 ECKUs，16 个带证据，26 个受控补丁。
  - 另含一个合成企业级三层服务（订单/定价、订阅/授权、退款/风控）的跨层基准（21 ECKUs，18 补丁）。
- **基线与模型**：
  - 基线：Plain / BM25 Chunk / EKP Retrieval / Rules Context（从 ECKU 渲染的 Markdown 规则）/ Direct ECK without Evidence。
  - 模型：Qwen2.5-Coder-32B-AWQ（主实验）、Qwen2.5-7B-Instruct（次实验），温度 0，top-k=6，8192 token 上下文，最多 800 生成 token。
- **RQ1 验证规划**（32B 模型，11 个有证据的任务）：
  - Exact Selector Recall：Plain=0 / BM25=0.455 / EKP=0 / Rules Context=**1.000** / Direct ECK (无证据)=0.091 / **Direct ECK=0.818**。
  - Executable Coverage Recall：**Direct ECK=1.000 / Rules Context=1.000 / BM25=0.455 / Plain=0**。
  - 配对 McNemar 检验 p=0.0078，说明显式证据对精确选择器恢复具有显著贡献。
  - JSON 解析失败：Plain 产生 12 个不可解析/空计划，BM25/Direct ECK 无失败。
- **RQ2 补丁影响**：
  - 独立人工标注的 12 条正向链接、0 假阳性、0 假阴性；Micro P/R/F1=**1.000**，Exact-set accuracy=**1.000**（所有补丁与仅正向补丁）。
  - 模型驱动补丁审查（Direct ECK Patch）：Parsed/Strict JSON/File Recall/ECK ID Recall/Exact Selector/Executable Coverage 均为 **1.000**；规则上下文在精确选择器召回上达 0.818，BM25 为 0.364。
- **RQ3 跨层投影**：
  - ECK Projection 以最少 token（4586 chars）实现 ECK ID=1.000、API=0.971、DB Field=**0.832**、Business Rule=1.000、Exact/Coverage=1.000。
  - Rules 虽也能达 1.000 验证指标，但 ECK ID=0.000、Business Rule=0.471，且长度更长。
- **RQ4 新鲜度**：
  - 真实仓库 50 个正向扰动 + 17 个同文件无关负例：Sensitivity=**1.000**，Specificity=**1.000**，FP rate=0.000。
  - Rules snapshot 在同一 50 个正向扰动上全部漏检（0/50）。
- **关键数字**：
  - 最强结果：直接 ECK 精确验证选择召回 **9/11**（paired McNemar p=0.0078），可执行覆盖 **11/11**；精确变更行影响 P/R/F1=**1.000**；跨层投影 DB 字段召回 **0.832**；新鲜度 S/Spe=**1.000/1.000**。

## 相关工作脉络
- **ContextBench [11] / CORE-Bench [17]**：强调 agentic coding 需仓库级搜索；本文并非竞争性检索基准，而是补充性表示层——在被检索前就让高价值单元自带可验证知识。
- **CodexGraph [12] / Code KG / Graph-RAG / Codebase-Memory [16]**：基于抽取的图或索引提供跨库导航；本文差异在于知识身份与验证溯源 reside 在源码本身，图/索引是其投影而非源头。
- **AGENTS.md / Cursor rules / Devin memories**：提供项目持久上下文；本文兼容但指出规则缺少源跨度、契约哈希、证据指纹与补丁影响能力，主张将规则视为 ECKU 的投影。
- **Reversa [4]**：将遗留系统逆向为可追溯运营规格；方向相反——Reversa 是从现有系统提取/重建，ECK 是作者直接在实现单元上编写声明式知识，二者可组合（逆向提出候选，人工审核后成为权威 ECKU）。
- **User as Code (UaC) [10] / Metis [3] / Code as Agent Harness [14]**：更广义的"可执行记忆/代理基础设施"；本文是其在软件工程 artifact 上的聚焦化实现：只针对仓库内选定单元绑定源跨度、契约、证据与新鲜度。
- **Protocol-Driven Development (PDD) [8] / SWE-Skills-Bench [7]**：PDD 强调不变量主权与可证明证据链；本文承认 SWE-Skills-Bench 版本失配风险，直接驱动 ECK 新鲜度设计。
- **Design by Contract [13] / Daikon [5] / safe regression test selection [15] / requirements traceability [6]**：继承契约与影响分析传统，但目标从"程序正确性"扩展到"代理可用知识对象"。

## 局限性与未来方向
- **规模与覆盖面**：仅 3 个小型 Python 库与 26 个受控补丁，无法代表大规模商业仓库或天然 commit 分布；ECKU 仅覆盖精选单元而非全仓。
- **语言限制**：当前原型仅限 Python，跨语言泛化未验证。
- **任务构建偏差**：issue 描述虽已与 patch 文件名解耦，但仍来源于已知补丁；更严格评估需天然 issue 或人工不查看补丁的独立任务。
- **人工标注单薄**：直接影响的黄金标签由单人标注；缺少多标注者 inter-annotator agreement。
- **影响分析范围有限**：仅支持 direct source-span 交集，不做关系展开式影响传播、依赖传播或安全回归测试选择；patch-review 与投影实验的"完美值"是**保真度量**而非独立影响发现。
- **端到端修复未验证**：评测停留在代理准备与补丁审查质量，未给出 SWE-bench 级别的最终补丁成功/接受率。
- **模型覆盖**：仅用 Qwen 家族 7B/32B；其他模型家族未见一致性质性验证。
- **新鲜度语义**：哈希变化是治理信号而非语义变更证明；作者/人类维护 ECKU 的真实负担未评估。
- **未来方向**：关系型组合、更大规模仓库/多标注者、天然 issue 评测、端到端修复成功率、跨语言扩展、版本失配下 ECK 对补丁接受率的提升、与 SWE-Skills-Bench 风格的 acceptance test 结合。

## 研究启发与可借鉴点
1. **"源绑定 + 可执行证据"可作为知识库设计的通用范式**：无论使用何种表示（图/规则/摘要），只要能把代理-facing 的知识声明反挂到原始代码的源跨度并绑定可执行验证命令，即可显著提升验证规划的精确性；建议在本团队的知识中间件设计中引入 ECKU-style 的契约/证据元字段。
2. **显式证据对精确选择器恢复的贡献可通过配对 McNemar 检验量化**：本文以 9/11 vs 1/11 + p=0.0078 的对照揭示"元数据/检索能定位文件，但显式证据才能定位精确测试选择器"的机制；这一对照思路可直接迁移到任何评测代理"应该跑哪个测试"的任务。
3. **新鲜度校验的多维哈希分离值得借鉴**：将 SourceHash / KnowledgeHash / ContractHash / EvidenceHash 分离并分别绑定 AST 边界与装饰器排除，既能捕获函数体改动，又能独立诊断"语义漂移"与"证据过期"；在实现自研 agent memory 刷新机制时，可照搬此分层指纹策略以避免同文件无关改动触发误告警。
4. **投影 vs 源层的架构分离**：本文结论"raw ECK 不一定是最佳交付格式，任务特定的 typed projection 能以更少 token 更高保真传递跨层字段"提示我们——在代理系统的知识栈里区分"源事实层"与"面向任务的投影层"，比单纯把 ECKU 原始 JSON 喂给 LLM 更有效。
5. **Rules-as-projection 的工业可迁移路径**：规则文件仍是代理高效消费形式，但可以把 ECKU 作为规则的权威来源，在 CI 中持续重新生成并附带 freshness 警告；这一"提取→人工审核 ECKU→CI 验证→再生成规则"的五步迁移路径，可在本团队的编码规范/合规知识沉淀流程中复刻。

## 关键术语表
- **ECKU (Executable Code Knowledge Unit)**：一种代码编写的知识对象，绑定源跨度、携带契约、可执行证据、关系、溯源与验证/新鲜度状态，供代理直接查询与消费。
- **Source-Knowledge-Contract-Evidence 四维哈希**：分别对代码体、代理 facing 语义、契约与证据计算哈希，用于细粒度新鲜度诊断与状态持久化。
- **Exact Selector Recall**：代理输出的测试命令（或类/文件级选择器）与金标选择器的精确匹配度；要求命令与测试选择器严格相等（pytest/python -m pytest 等价）。
- **Executable Coverage Recall**：允许代理输出更宽泛的命令（如文件/类级别）只要包含金标测试的选择器即算命中，用于评估覆盖能力而非精确度。
- **ECK Projection**：从 ECKU 派生的面向任务的紧凑投影（如 impacted ECK IDs、API、DB 字段、Business Rule、验证命令），作为代理上下文的最终交付格式。
- **Rules Context / Rules with IDs**：将 ECKU 渲染为 Markdown 规则的记忆格式；前者不含 ID，后者显式加入稳定标识符，二者对比揭示"身份可见≠源绑定"。
- **Patch Review Fidelity**：在给定补丁 diff 后，代理从 ECK 报告/规则/BM25 等上下文中识别受影响 ECK ID、符号、验证命令与风险标志的能力；本文该指标测量的是投影保真度而非独立影响发现。
- **Freshness Sensitivity / Specificity**：在 50 个正向扰动 + 17 个同文件无关负例上，AST 边界指纹识别真实变更的能力（敏感度）与不误报无关编辑的能力（特异性），两者均达 1.000。

## 可复现要素
- **数据集**：三个真实 Python 仓库（python-dotenv、python-slugify、tomli）与一个合成企业级三层服务；26 个受控补丁 + 独立标注的直接影响黄金集 + 50 正向/17 负向新鲜度扰动样本。**仓库与补丁为公开开源项目**，人工构建的补丁/标注与 ECK 实验脚本见 artifact。
- **代码/权重**：ECK CLI、测试、示例、patch 报告复现、实验脚本、黄金标签、保存的原始模型输出均已打包到 commit-pinned 匿名 artifact 仓库；模型权重使用本地部署的 Qwen2.5-Coder-32B-AWQ / Qwen2.5-7B-Instruct（vLLM 0.19.1，OpenAI-compatible 端点），论文未提供模型权重下载链接，实验需自备模型环境。
- **关键超参**：temperature=0，top-k=6 context 单元/chunks，8192-token 模型上下文窗口，最多 800 生成 token；AST 边界使用 lineno/end_lineno；装饰器不参与 source fingerprint；关系规范化后进行 KnowledgeHash。
- **硬件与环境**：主实验在单卡 NVIDIA H20（97,871 MiB）、driver 550.163.01、PyTorch 2.10.0+cu128、vLLM 0.19.1、OpenAI Python client 2.44.0 下复现；最小 CPU-only 复现仅需 `python -m pytest` + 示例 diff。
- **评估脚本**：规划实验 `experiments/run_agent_context_experiment.py`、补丁审查实验 `experiments/run_patch_review_experiment.py`、bootstrap CI 分析 `experiments/analyze_patch_review_results.py`、规则老化对比 `experiments/compare_rules_staleness.py`、真实仓库新鲜度 `experiments/run_eck_real_repo_freshness.py` 等均已开源在 artifact。
