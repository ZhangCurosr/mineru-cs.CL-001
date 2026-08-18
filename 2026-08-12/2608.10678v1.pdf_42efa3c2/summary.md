---
title: "Auditing Chinese Web-scale Corpora via Sampled BPE Token Statistics <sup>\u0011</sup>Caution: this paper may include offensive and upsetting content."
source: https://arxiv.org/pdf/2608.10678v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:33:18"
---

# 论文速读：Auditing Chinese Web-scale Corpora via Sampled BPE Token Statistics <sup>⚠</sup>Caution: this paper may include offensive and upsetting content.

## 一句话总结
本文提出 SAMPLED-BPE，一种基于小比例采样与 BPE 词频统计的轻量级 token 级审计流水线，将 TB 级中文网络语料的污染审计成本从“数月”压缩至“数小时”，并在 0.25% 采样下仅产生约 4.25% 的类别相对误差；基于该流水线对 11 个开源中文语料库及 2021–2026 年 Chinese Common Crawl 快照的系统性审计揭示：上游中文网页污染已从边缘噪声演变为与正常内容相当的高占比（2026 年达 79.52%），且呈现强烈类别漂移与快速词汇更替特征。

## 研究问题与动机
- **核心问题**：如何以低成本、细粒度（token 级）且能适应动态变化的方式，审计 TB 级中文预训练语料库中的隐性污染（色情、赌博、垃圾信息等）。
- **现有方法不足**：
  1. **规模瓶颈**：主流中文语料库达数百 GB 至 TB 级，完整扫描/索引成本极高（例如 83TB 数据在 128 核 CPU 上需 99 天）。
  2. **粒度粗糙**：既有语料库审计多停留在文档级语言分布、来源域名或质量评分，无法暴露 token 级污染。
  3. **规则滞后**：中文网络污染常为隐晦缩写、语境依赖型表达且更迭迅速，固定关键词词表极易过时且遗漏新兴污染词。

## 核心贡献（创新点）
- **提出 SAMPLED-BPE 轻量级审计流水线**：通过流式采样+BPE 词频统计+网络搜索上下文映射，实现从“月级全量扫描”到“小时级近似审计”的成本跃迁，验证了采样逼近在精度与效率上的最优平衡点。
- **系统性揭示中文上游语料的污染图谱**：首次对 11 个主流开源中文语料库与 2021–2026 年 Common Crawl 快照进行 token 级对比审计，发现污染在语料库间高度不均衡，且上游网页污染已占比近八成并持续动态漂移。
- **发布分层中文网络 token 数据集**：构建包含 630,684 条记录与 92,972 棵子串树的层级化数据集，支持污染家族追溯与“正常词组合衍生污染”的结构化溯源。
- **与已有工作的本质区别**：既往研究多聚焦下游 LLM 词表反推污染源（事后观测）或文档级质量评估（粗放粒度），本文直接面向上游语料实施自动化 token 级审计，并以采样+BPE 统计替代全量扫描与静态规则，填补了可扩展、周期性中文语料安全审计的方法空白。

## 方法详解
SAMPLED-BPE 采用四阶段流水线设计（如图 3 所示）：
1. **流式采样 (Streaming Sampling)**：对目标语料执行单次顺序扫描并选取前缀子集，避免全量随机打乱所需的标识符枚举与内存物化，I/O 成本近似线性，精度与随机采样相当。
2. **BPE 训练与词频计数**：在采样子集上训练 Bytepair Encoding (BPE) tokenizer 并统计 token 频次。BPE 作为压缩方法会反复合并高频相邻单元，其 learned vocabulary 天然暴露语料的重复词汇模式；高频污染 token 可在无固定关键词的前提下被自动浮现。
3. **类别映射 (Category Mapping)**：针对污染 token 常为缩写、隐晦或依赖上下文的特点，将每个 token 与其网络搜索结果作为上下文联合输入微调后的 GLM-4-32B 分类器，映射至 6 类：Normal Content、Adult Content、Online Gambling、Online Gaming、Online Video、Anomalous。分类器基于 Zhang et al. (2025) 专家标注的 GPT 词表进行 train/test 切分微调，测试集准确率达 97.32%。
4. **语料画像聚合 (Corpus Profiling)**：在 token 级记录占比、预测类别、搜索证据与分类理由；在类别级聚合 token 占比得到各类污染比例，形成可跨语料/跨时间对比的 corpus profile。
- **精度-效率权衡**：实验证明采样可保留可用估计。0.25% 采样下 token 覆盖率 76.83%，共享 token 的 Spearman/Pearson 相关系数分别达 86.88%/99.93%，类别加权相对误差 (WRE) 仅约 4.25%；同时实现 148.4× 运行加速与 35.8× 峰值内存缩减。

## 实验与结果
- **数据集与基线**：审计 11 个主流开源中文语料库（OSCAR, mC4, HPLT, CulturaX, CWT, ROOTS, WanJuan, MAPCC, SkyPile, CCI3, WuDao）及 2021–2026 共 6 个 Chinese Common Crawl 快照；以 Full-corpus 全量审计结果作为 ground truth 对比。
- **主要结果**：
  1. **污染广泛但不均**：OSCAR 污染率最高（83.38%，Adult 占 76.10%），mC4 次之（25.41%，Gambling 占 23.17%）；WanJuan、MAPCC、SkyPile、CCI3、WuDao 均低于 1%。清洗策略能显著降污（如 CWT-Cleaner 从 2.35% 降至 0.50%），但无法清零，残余多为 Anomalous。
  2. **污染 token 高度语料特异**：11 个语料库中共检出 106,671 个污染 token，其中 96.386% 仅出现在单一语料库，仅 1.446% 出现于 ≥3 个语料库；共同核心（≥8 个语料库）仅 252 个，主要为长期稳定的黑话/模板（如“老司机”“威尼斯人”）。
  3. **
