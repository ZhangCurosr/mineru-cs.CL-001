---
title: "When-Your-Agent-Opens-the-Chat-App-Agent-Controlled-Search-o"
source: https://arxiv.org/pdf/2608.12888v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:24:32"
---

# 论文速读：When-Your-Agent-Opens-the-Chat-App-Agent-Controlled-Search-o

## 一句话总结
提出 ReFind，一种完全不构建语义索引的 Agent 可控搜索接口，仅对原始聊天记录做词法级 BM25 索引，并结合迭代查询循环与四项聊天原生控制（会话感知排名融合、局部上下文扩展、时间过滤、已检会话去重），在精确检索与事实追踪任务上媲美甚至超越依赖预构建图谱/树状结构的 Agent 记忆系统。

## 研究问题与动机
- 现有 Agent 记忆系统普遍在问题到来前将原始对话历史转换为摘要、向量、树或知识图谱，该“先验加工”会丢弃细节且需离线/增量维护成本。
- 核心疑问：已有工作报告的检索收益，究竟有多少来自“结构本身”，又有多少仅源于“对原始历史的有效检索能力”？
- 个人信息重找（refinding）实证表明，用户通常通过小步、上下文引导的迭代操作（关键词尝试、滚动浏览、时间/会话线索）找回目标，而非单次精准查询。
- 动机：探索“零语义结构、保留原始记录、将智能移至检索控制”的极端路径，验证其能否在精确证据检索场景下替代复杂的预构建记忆架构。

## 核心贡献（创新点）
1. **ReFind 接口设计**：提出无 LLM 预构建索引的 Agent 可控搜索框架；与已有工作的本质区别是将检索智能从“离线结构化构建”转移到“在线问答驱动的状态机控制”。
2. **RRF 双层重排序机制**：结合 Turn 级 BM25 排名与会话级聚合排名进行倒数融合；与通用 Passage Reranking 的本质区别是显式利用聊天档案的会话拓扑结构而非纯文档级信号。
3. **状态化多轮检索控制空间**：将上下文扩展、时间过滤、已检会话去重与迭代 Query 重构统一为 Agent 的状态动作空间；与 Generic Agentic RAG 的本质区别是引入对话历史特有的导航原语，而非通用多轮工具调用。
4. **实证结论：结构收益可被可控检索回收**：在匹配骨干模型条件下，ReFind 在多项基准上超越强结构化基线；与已有工作的本质区别是证明了“保真存储+在线可控检索”可替代“压缩存储+离线索引”。

## 方法详解
- **两阶段解耦架构**：Stage 1（检索）由 ReAct Agent 自主决定关键词与检索参数，最多 4 轮搜索并将关键片段保存为 Notes；Stage 2（推理）将 Notes 按会话分组、按时间排序后交给 Answer Model 生成答案，避免检索与推理争夺上下文窗口。
- **词法索引**：以 Turn 为粒度构建 BM25 倒排索引（`k1=1.2, b=0.75`），支持增量追加，新消息写入即可检索，无需 LLM 调用或摘要预处理。
- **RRF 双层重排序**：对每条 Turn 计算 turn 级排名 `r1` 与会话级聚合排名 `r2`（会话得分为其内所有 Turn BM25 分数之和），融合公式 `RRF(c) = 1/(k+r1(c)) + 1/(k+r2(c))`（`k=60`），输出 Top-K（默认 5）Turn。
- **四项聊天原生控制**：
  1. **局部上下文扩展**：命中 Turn 前后各 ±w（默认 2）条 Turn 作为检索块，截断不跨会话边界，恢复代词/省略指代。
  2. **时间过滤**：Agent 可传入 `date_from`/`date_to`，在 BM25 打分前按时间窗裁剪候选。
  3. **会话去重**：维护已返回会话集合 `S_t`，后续搜索自动排除，避免预算浪费于重复会话。
  4. **迭代查询重构**：Agent 根据上一轮 Observation 调整关键词、收窄时间或切换实体。
- **形式化表达**：检索过程为 `O_{t+1} = Π_w(RRF(BM25(k_t, H|τ_t, S_t)), K_t)`，其中 `τ_t` 为时间约束，`S_t` 为已见会话集合，`Π_w` 为上下文窗口扩展。标准 RAG 退化为 `T=1` 且各控制项失活的情形。

## 实验与结果
- **数据集**：MemoryAgentBench（6 个子任务，约 2800 题：SH-QA、MH-QA、LME、EventQA、FC-SH、FC-MH）；LongMemEval-S/M（50/15 题，~115k/~500k tokens）。
- **基线**：长上下文模型（GPT-4o-mini）、稀疏/稠密 RAG（BM25-RAG、Contriever、text-embed 系列）、结构化记忆（RAPTOR、GraphRAG、MemoRAG、HippoRAG 2、Mem0、Zep）、Agent 记忆（Self-RAG、MemGPT、MIRIX、GAM、STITCH）。
- **主干与评判**：MemoryAgentBench 统一使用 GPT-4o-mini；LongMemEval 使用 GPT-5-mini，评判器为 GPT-4.1-mini。
- **主要结果**：
  - MemoryAgentBench 均值准确率 **ReFind 58.2**，居首；次之为 HippoRAG 2（53.2）与 BM25-RAG（48.8）。ReFind 在 SH-QA（83.0）、MH-QA（69.0）、LME（51.3）、FC-SH（62.7）、FC-MH（8.8）五项第一。
  - LongMemEval：ReFind 达 **S: 93.2±3.3，M: 89.3±6.0**，显著超越 STITCH（86.0/80.0）、HippoRAG 2（80.0/66.7）及 GAM（70.0/60.0）。
- **消融结论**：移除聊天原生控制降至 Generic Agentic BM25（S: 78.7, M: 82.2）；单次搜索（One-search）比全方法低 8.5（S）/20.4（M）分；上下文窗口对 S 贡献最大，会话去重对 M 贡献最大；BM25 后端优于 Dense/Hybrid 后端。

## 相关工作脉络
1. **结构化 Agent 记忆（GraphRAG, HippoRAG 2, RAPTOR, Mem0, Zep 等）**：主张离线预构建摘要/图谱/向量库以提升抽象推理效率；本文定位为其互补面，证明在精确检索场景下原始记录+可控搜索可替代预构建结构。
2. **Agent 检索系统（MemGPT, GAM, Self-RAG, A-RAG）**：赋予模型多轮检索工具调用能力；本文区别于通用文档检索，首次将“会话拓扑、时间线、已检状态”系统性地纳入 Agent 动作空间。
3. **个人信息重找实证（Teevan 2004, Whittaker 2011, Cheng & Aflatoony 2022）**：发现用户依赖本地步进、时间/线索标记与搜索历史；本文将其四项行为模式工程化为可计算的控制原语并验证其效用。
4. **稠密/混合检索 RAG（Contriever, text-embed
