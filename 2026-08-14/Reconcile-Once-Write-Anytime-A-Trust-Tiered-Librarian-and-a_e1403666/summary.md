---
title: "Reconcile-Once-Write-Anytime-A-Trust-Tiered-Librarian-and-a"
source: https://arxiv.org/pdf/2608.12984v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:49:13"
field: "可信AI系统与长文本生成"
keywords: ["trust-tiered retrieval", "point-in-time evaluation", "multi-agent report generation", "deterministic QC gate", "metric ledger", "write-back self-correction", "cross-section drift"]
innovations: ["将信任分层时间戳知识库与报告生成解耦，一致性成为存储属性", "通过不可变快照只读扇出消除多agent并发下的跨章节数值漂移", "六步确定性QC门禁经缺陷注入元评估达到Recall=1.0、Precision=1.0"]
benchmarks: ["SEC EDGAR filings (5397 sources)", "BLS macro releases (672 sources)", "Wikipedia (61 sources)", "self-collected production corpus: 6130 sources, 555926 evidence cards"]
---

# 论文速读：Reconcile-Once-Write-Anytime: A Trust-Tiered Librarian and a Multi-Agent Writer for Drift-Free, Point-in-Time Research

## 一句话总结
提出了一套部署导向的两阶段Agentic系统，将维护性的时间点知识库与长篇报告生成解耦：确定性"图书管理员"维护信任分层本体与共享指标账本，便携式多智能体"写作者"在任意时间点切面生成无漂移、可溯源的报告，并通过红队写回闭环实现自我修正。

## 研究问题与动机
- **数值漂移**：同一公司的同一指标在不同章节出现不同数值（因每段独立grounding）；
- **溯源丢失**：数字出现但无来源可追溯，读者无法区分经审计的披露与流畅的幻觉；
- **信任扁平化**：发布前谣言与已提交的10-Q文件以同等置信度被引用，系统缺乏源权威性概念；
- **现有RAG缺陷**：按查询检索原始片段并孤立grounding，无法维护一致且持续演化的判断体系，agentic writer虽协调章节但继承同样空白。

## 核心贡献（创新点）
1. **两阶段解耦系统**：确定性图书管理员（Phase A）与便携式多智能体写作者（Phase B）通过时间戳投影桥接，与现有RAG按查询检索的做法本质不同——一致性成为存储属性而非查询属性。
2. **共享存储协调的多智能体运行时**：异构agent（分节composer + 对抗性red-team prosecutor）在受控并发池下运行，通过信任分层存储间接协调而非直接消息传递，从根本上消除了并发writer的跨章节不一致问题。
3. **信任分层一致性机制**：官方优先的指标账本（tier → corroboration → recency）、类型化声明图（contradicts/supersedes/qualifies）、六步确定性QC门禁，与Self-RAG等"让同一模型评判自身输出"的方案本质不同。
4. **自修正写回闭环**：报告侧红队反驳映射为图书馆的source_override/claim_refuted，下一时间点自动自我修正，无需人工干预；同时检测"锚点置换"（approved evidence静默消失）并标记重新验证。
5. **机械式元评估验证的评估框架**：通过缺陷注入+负对照验证QC门禁的召回率和精确率均为1.0，且所有指标由机器计算、端到端可复现，区别于FinReasoning等LLM-as-judge方案。

## 方法详解
**Phase A：确定性图书管理员**
- 管道由确定性Python阶段组成，将公共源分类为信任层级（official > gov\_stat > media），记录真实发布日期；
- 维护三层递进蒸馏的本体：
  - **证据卡（Evidence Card）**：原始引文锚定的事实卡，携带source\_tier、as\_of、value\_norm等字段；
  - **指标账本（Metric Ledger）**：每个(company, metric)按tier → corroboration → recency选取权威值，保留竞争值作为alternatives并标记disagrees；
  - **声明图（Claim Graph）**：存储contradicts/supersedes/qualifies类型的有向边，追踪冲突与替代关系。
- LLM仅用于一个有界接缝：Haiku 4.5对歧义数值卡做精炼（修正value/unit或降为定性）；
- **时间戳投影（Point-in-Time Projection）**：给定截止时刻T，筛选as\_of ≤ T的证据生成四个工件（outline、evidence cards、ledger、claim graph），形成"无前瞻" seam。

**Phase B：便携式多智能体写作者**
- 固定DAG工作流：Slice → Compose → Normalize → Red-Team → Apply Verdicts → Bounded Rewrite → QC Gate → Render；
- **难度分层路由**：涉及未解决冲突的章节路由至Opus 4.8（高难），其余路由至Sonnet 5（常规），实现异构调度；
- **权力分离**：Composer满足section contract，Opus prosecutor独立red-team（不能给自己打分），仲裁者是确定性QC门禁而非LLM；
- ** stigmergic协调**：Agent间无直接消息传递，仅读写共享信任分层存储；
- **写回闭环**：Red-team反驳映射为source\_override（status=retracted），下一cutof继承修正；改写仅降级被反驳的源，不发明新值，所有变更追加至append-only audit log。

**QC门禁（六步确定性检查）**
(1) Orphan citation（孤儿引用）→ blocking error；(2) Unsourced number（无来源数值）→ warning；(3) Numeric drift（数值漂移）→ 由账本结构性消除；(4) Buried contradiction（埋藏矛盾）→ blocking；(5) Unregistered metric（未注册指标）→ blocking；(6) Cross-section contradiction（跨章节矛盾）→ blocking。

## 实验与结果
**数据集**：自建生产级公共语料库，6,130个来源，555,926张证据卡片（457,561张数值卡），2,589个权威公司-指标值，295个发行人，11个行业；来源包括SEC EDGAR文件（5,397）、BLS宏观发布（672）、Wikipedia文章（61）；时间跨度2023-06至2026-07。

| 实验 | 关键结果 |
|---|---|
| E1 跨章节漂移消除 | 无账本基线产生6,845个矛盾数值（2,105个指标）；本文0个；7个时间点重放4,732次数值变化100%有据可查 |
| E2 落地 grounding | 202/203（99.5%）数字行有证据引用，0个孤儿引用，0个未注册指标 |
| E3 信任分层 | 5,397官方文件全部正确分类为hard evidence，672 gov\_stat为supporting，61 media为routing-only；0张数值卡来自media；0个gov\_stat值顶替了公司自有official值 |
| E4 分层选择 vs 流行度 | 本文tier-first在22/22黄金案例上正确；popularity-first仅9/22；13个流行度陷阱（广泛重复的低层谣言）全部幸存 |
| E5 QC门禁元评估 | Recall=1.0（5/5缺陷全捕获），Precision=1.0，False-positive rate=0（3个负对照均未触发） |
| E6 并行+路由 | 有界并发并行比串行快3.7×；难度分层路由比全Opus节省4.1%成本；分层质量评分（grounding+conflict+contract）比全Opus高+0.079、比全Sonnet高+0.262 |
| E7 活库时间重放 | 7个时间点卡片从235,373单调增长至555,312；0个前瞻违规；捕获4,395个后到事件 |
| E8 写回自我修正 | 示例案例：红队反驳$19mn利息后，账本自动回退至$9mn备选值，0次人工编辑；批次中5/6写回成功自我修正 |

最强结果：跨章节漂移从6,845降至**0**；QC门禁Recall/Precision均为**1.0**；tier-first选择在黄金集上**22/22**正确（vs 流行度9/22）；分层路由质量超越全Opus**+0.079**。

## 相关工作脉络
- **RAG / Self-RAG** [10, 16, 3]：按查询检索原始片段、孤立grounding；Self-RAG让同一模型评判自身输出，而本文用确定性门禁+独立red-team，且维护跨时间一致的本体而非静态检索。
- **TimeQA / StreamingQA / RealTime QA** [4, 17, 14]：聚焦短答案的时间敏感QA，处理前瞻问题；本文面向长篇报告，额外解决跨章节一致性和源权威性治理。
- **GraphRAG / GFM-RAG** [7, 18]：在实体/声明图上推理但无溯源分层和时间戳投影，每次查询重新调解而非维护持久化账本。
- **MemGPT** [20]：持久化agent状态但无源权威性概念和无前瞻纪律。
- **MetaGPT / AutoGen / LangGraph** [11, 24, 15]：通用多智能体框架；本文不绑定外部框架，自建轻量headless编排器，且通过共享存储而非消息传递实现协调。
- **REGAL / OntoMetric** [1, 26]：共享确定性核心纪律，但前者基于遥测数据、后者聚焦单文档ESG图，本文扩展到跨文档非结构化证据的时间戳调解。

## 局限性与未来方向
- 语料库仅限英语且为三层设计（sell-side被省略，仅出现在E4黄金集中）；
- gov\_stat层（BLS宏观数据）无公司归属，设计上只能丰富上下文而不能进入公司级账本；
- 实体链接基于子串匹配，偶尔会过度归因（如"3M"可能匹配无关文本）；
- E4的跨层级黄金集是设计的而非采样的（部署语料库中无跨层级cluster），尽管层内规则在2,132个真实冲突上得到验证；
- E6使用确定性质量代理而非人工判断；E8写回追踪仅演示了小批次；
- E1的无账本基线是孤立实验臂，未与graph/语义层检索器等强竞争者对比。

## 研究启发与可借鉴点
- **确定性核心 + LLM边缘**：将选择、一致性、QC等可机械验证的逻辑置于确定性层，LLM仅用于有界的提炼/创作环节，使主指标可复现且门禁可元评估——可迁移至任何需要高可信度输出的长文本生成场景。
- **不可变快照 + 只读扇出实现并发安全**：通过bridge生成point-in-time冻结快照，所有composer只读并发访问同一快照，天然消除写-写冲突、脏读和死锁，无需锁或两阶段提交——为多agent并行编排提供了简洁的一致性保证。
- **stigmergic协调模式**：Agent间不直接通信，仅通过共享存储读写，将协调复杂度从通信拓扑转移到数据模型——适合需要严格隔离职责和审计追踪的系统。
- **QC门禁的元评估设计**：用缺陷注入（正样本）+ 负对照（clean perturbations）同时验证recall和precision，消解"自己批改自己作业"的问题，是评估生成系统gate的通用范式。
- **难度分层路由的反直觉结论**：全中等模型并非最优，在困难推理处投资强模型反而超越全Opus——提示在agent编排中"均匀省钱"可能损害最关键的合成环节。

## 关键术语表
- **Trust Tiering（信任分层）**：按源权威性对来源分级（official > gov\_stat > sell\_side > media），不同层级决定其数值是否可作为hard evidence被引用；
- **Point-in-Time Projection（时间点投影）**：将动态库按截止时刻T投影为只读快照（as\_of ≤ T），确保报告不"前瞻"未来信息；
- **Metric Ledger（指标账本）**：每个(company, metric)对按tier → corroboration → recency选取唯一权威值，消除跨章节数值漂移；
- **Claim Graph（声明图）**：存储数值间contradicts/supersedes/qualifies类型的有向边，使冲突可追溯且可调解；
- **Write-Back Loop（写回闭环）**：报告侧red-team反驳映射为图书馆的source\_override，下一cutof自动继承修正，无需人工干预；
- **QC Gate（质量控制门禁）**：六步确定性（无LLM）检查门禁，错误集非空则阻断交付，自身经缺陷注入元评估；
- **Anchor Swap（锚点置换）**：approved evidence静默消失但补填卡片计数不变的情况，系统会标记而非静默保留；
- **Stigmergic Coordination（痕迹协调）**：Agent通过共享环境状态间接协调，而非直接消息传递。

## 可复现要素
- **数据集**：自建公开可再分发语料库（6,130来源，555,926证据卡），来源于SEC EDGAR和BLS REST API（带provenance headers）；论文未提及代码/权重是否开源；
- **关键超参**：Conflict阈值>15%（单位等值的重述不触发假阳性）；并行worker数Ntrade latency与cost（E6）；DAG固定结构；所有run记录cost与latency；
- **评估平台**：E6在真实Bedrock后端运行3次取均值；确定性实验（E1/E3/E4/E5/E7/E8）离线stub后端运行；
- **复现材料**：附录A-G提供worked example、QC门禁规格、ontology schema、DAG图示、可视化图和复现步骤。
