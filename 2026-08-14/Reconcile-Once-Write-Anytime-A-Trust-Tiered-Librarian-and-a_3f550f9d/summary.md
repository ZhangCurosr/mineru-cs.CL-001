---
title: "Reconcile-Once-Write-Anytime-A-Trust-Tiered-Librarian-and-a"
source: https://arxiv.org/pdf/2608.12984v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:48:38"
field: "面向可信长文生成的多Agent系统与溯源架构"
keywords: ["trust-tiered retrieval", "multi-agent writing", "point-in-time evaluation", "cross-section drift", "deterministic QC gate", "long-form report generation", "metric ledger"]
innovations: ["信任分层账本+声明图解耦存储与生成，实现跨节零漂移", "六项确定性QC门禁经缺陷注入meta-eval验证", "红队refutation回写驱动图书馆自修正闭环"]
benchmarks: ["SEC EDGAR 6130 sources", "BLS macro releases 672", "Wikipedia 61 articles"]
---

# 论文速读：Reconcile Once, Write Anytime: A Trust-Tiered Librarian and a Multi-Agent Writer for Drift-Free, Point-in-Time Research

## 一句话总结
本文提出一个双层代理系统，将"点时知识图书馆"与长文报告生成解耦：确定性Librarian将带时间戳的公开源按信任分层收录至证据卡片、权威指标账本和声明图；便携多Agent Writer从中读取快照生成矛盾为零、可追溯的研究报告，并通过红队机制闭环自修正。

## 研究问题与动机
- **数值漂移**：同一条指标在不同章节引用不同数值，因各段落独立 grounding 导致跨节不一致。
- **来源失踪（provenance loss）**：数字无源可溯，审计文件与流畅幻觉难以区分。
- **信任扁平化**：发布前传闻与已提交 10-Q 被同等对待，系统缺乏"源权威"概念。
- **现有RAG/Agent方案不足**：RAG 逐查询检索原始 chunk，不维护一致演化的知识本体；现有 agentic writer 虽协调章节但仍继承上述缺陷。

## 核心贡献（创新点）
1. **双阶段解耦架构**：确定性 Librarian 维护时间戳知识本体（Phase A）与便携式多 Agent Writer（Phase B）分离，后者以切分点 T 读取快照生成报告，而非逐查询 RAG。
2. **信任分层一致性机制**：官方优先的 Metric Ledger（tier → corroboration → recency 排序）+ 有向声明图（contradicts/supersedes/qualifies 边），从存储侧保证跨节数值唯一，而非依赖调度或共识协议。
3. **六项确定性 QC 门禁**：无 LLM、语言无关的六项机器检查（孤儿引用、无源数值、数值漂移、埋藏矛盾、未注册指标、跨节矛盾），且门禁本身经缺陷注入 meta-eval 验证（recall=1.0, precision=1.0）。
4. **自修正回写闭环**：红队 refutation 映射为 source_override 写入图书馆，下一切分点自动回滚/替代，零手动编辑；锚点置换（anchor swap）触发重新校验。
5. **分布式 Writer 设计**：难度分级模型路由（冲突章节→Opus、其余→Sonnet）+ 有界并发扇出，较串行快 3.7×，质量超过 all-Opus 上限 +0.079。

## 方法详解
- **Phase A（Librarian）**：Python 管道按 true publication date 收录源，分配信任层级（official / gov\_stat / media）；提取引文级证据卡片（quote-grounded evidence card），含 `metric_value`、`value_norm`、`source_tier`、`as_of` 等字段；对含歧义数值的卡片用 Claude Haiku 4.5 做一次廉价 refine pass。
- **Metric Ledger**：每个 `(company, metric)` 三元组选出一条权威值，策略为 tier 优先 > 跨源互证数量 > as\_of 时效；冲突保留为 alternatives，>15% 阈值触发 flag。
- **Claim Graph**：在数值上构建 `contradicts / supersedes / qualifies` 三类有向边，支持时间线推演。
- **Bridge（点时投影）**：给定切分点 T，过滤 `as_of ≤ T`，重算 ledger，输出 outline / evidence cards / metric ledger / claim graph 四份 artifact，实现 no-look-ahead。
- **Phase B（Writer DAG）**：slice → compose（difficulty-tiered 路由）→ normalize → red-team（Claude Opus 4.8 "prosecutor" 逐节反驳）→ apply verdicts → bounded rewrite → deterministic convergence backstop → QC gate → render。
- **Write-back loop**：red-team verdict 映射为 `source_override`/`claim refuted`，追加至 append-only audit log；被否决源降级后，ledger 回落到已有同类备选值。
- **QC 六项检查**（Appendix B）：①orphan citation ②unsourced number ③numeric drift ④buried contradiction ⑤unregistered metric ⑥cross-section contradiction；均阻断交付，非事后打分。

## 实验与结果
- **数据集**：自建生产级公开语料，6,130 源 → 555,926 张证据卡片（457,561 数值），覆盖 295 发行人、11 行业（SEC EDGAR 5,397 / BLS 672 / Wikipedia 61），时间跨度 2023-06–2026-07。
- **E1 跨节漂移**：无 ledger 基线出现 **6,845** 条矛盾数值（2,105 个指标）；本系统输出 **0**。
- **E2 溯源覆盖**：202/203（**99.5%**）数字行带证据引用，零孤儿引用、零未注册指标。
- **E3 信任分层**：5,397 official / 672 gov\_stat / 61 media 全正确分类；457,561 张数值卡中 **0** 条追溯到 routing-only 源。
- **E4 分层 vs 流行度**：22/22 黄金案例正确（流行度基线仅 9/22）；13 个"流行度陷阱"全部由分层规则存活。
- **E5 QC 门禁验证**：recall = **1.0**（5/5 缺陷全捕获），precision = **1.0**，负样本误报率 0%。
- **E6 并行+路由**：有界并发较串行快 **3.7×**；难度分级较全 Opus 省 4.1% 成本；分级质量分超过 all-Opus **+0.079**、超过 all-Sonnet **+0.262**。
- **E7 活体库生长**：7 个切分点重放，**0** look-ahead 违规，卡片从 235,373 增至 555,312；4,732 次权威值变更 100% 有据可查。
- **E8 回写自修正**：单案 `$19mn→$9mn` 零手动编辑；批量 5/6 指标自修正，均回落至同类备选。
- **最强结果**：跨节漂移归零（6,845→0）；E4 选型准确率 22/22（vs 9/22）。

## 相关工作脉络
1. **RAG**（Lewis et al., 2020; Gao et al., 2023）：逐查询 grounding，不维护演化本体；本文在存储侧固化一致性。
2. **Self-RAG**（Asai et al., 2024）：LLM 自我批判；本文 QC 门禁是确定性、meta-validated 且阻断交付，非事后打分。
3. **GraphRAG / GFM-RAG**（Edge et al., 2024; Luo et al., 2025）：实体/声明图推理，无溯源分层与点时投影；本文在时间戳维度做维护式 reconciled store。
4. **Temporal/Streaming QA**（TimeQA, StreamingQA, RealTime QA）：关注短问答中的前瞻问题；本文面向长篇研究报告的跨节一致性挑战。
5. **STORM**（Shao et al., 2024）：合成百科类文章，无源权威分层、无指标账本、无点时纪律。
6. **多 Agent 框架**（MetaGPT, AutoGen, LangGraph）：本文自建轻量 headless runtime，以共享存储协同而非消息传递，避免并发分歧。

## 局限性与未来方向
- 语料仅英语、仅三级分层（sell\_side 未纳入）；gov\_stat 无法提供公司归属，仅 enrich 宏观上下文。
- 实体链接基于子串匹配，可能将公司短名误归到无关文本（如"3M"）。
- E4 跨层黄金集为设计覆盖而非采样（真实语料无跨层冲突）；E6 用确定性质量代理，非人工评判。
- 未与强 shared-state 竞品（如图语义层检索器）直接对比，E1 仅隔离 ledger 效应。
- 未来：扩展 sell\_side 层级、改进实体链接精度、对接 graph-based retriever 基线、引入人类主动评审提升机制。

## 研究启发与可借鉴点
1. **"权威账本"范式**：将一致性作为存储属性而非生成属性，适用于任何需要跨段落数值对齐的长文生成场景（研报、尽调备忘录、监管申报）。
2. **分层+互证+时效的三重排序策略**（tier → corroboration → recency）简洁且对抗流行度陷阱，可直接迁移至金融/医疗知识图谱构建。
3. **QC 门禁 meta-eval**（缺陷注入+负样本）是验证任何自动化审查模块的通用方法论，比 LLM-as-judge 更可靠。
4. **确定性核心+LLM 边缘**的架构（核心管道无 LLM，仅在 refine pass 用 Haiku）兼顾可复现性与灵活性，适合高可信生产部署。
5. **难度分级路由**在长文写作中不仅省钱，还可能提升质量（将强模型聚焦于冲突密集章节），与本团队多路由编排方向高度相关。

## 关键术语表
**Trust-tiered ontology**：按来源权威（official > gov\_stat > media）分层的知识本体， governs 数值引用而非 popularity。
**Metric Ledger**：每个 (公司, 指标) 三元组维护一条权威值及冲突 alternatives 的结构化账本。
**Evidence Card**：绑定原文引用的原子证据单元，含 `metric_value / value_norm / source_tier / as_of` 等字段。
**Claim Graph**：在数值间建模 contradicts / supersedes / qualifies 三类有向边的冲突图谱。
**Point-in-time projection**：以切分点 T 过滤 `as_of ≤ T` 的证据快照，实现 no-look-ahead 的时点报告。
**Write-back loop**：红队 refutation 映射为 source override 回写图书馆，驱动自修正闭环。
**QC Gate**：六项无 LLM、语言无关的确定性检查，阻断不合格报告交付。
**Anchor Swap**：已被批准的证据随后消失（即使卡片数不变），触发声明重新校验。

## 可复现要素
- **数据集**：自建公开语料（SEC EDGAR + BLS + Wikipedia），6,130 源 / 555,926 证据卡；论文声明为可公开获取，具体链接见附录 G。
- **代码/权重**：论文未提及开源仓库，Phase B 为自建 headless runtime（未绑定 LangGraph 等外部框架）；Phase A 核心为确定性 Python 管道。
- **关键超参**：冲突阈值 15%；并发 worker 数 n（可交易 latency/cost）；模型路由：Opus 4.8（冲突章节）/ Sonnet 5（其余）/ Haiku 4.5（refine pass）。
