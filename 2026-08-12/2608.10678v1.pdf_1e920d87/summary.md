---
title: "Auditing Chinese Web-scale Corpora via Sampled BPE Token Statistics <sup>\u0011</sup>Caution: this paper may include offensive and upsetting content."
source: https://arxiv.org/pdf/2608.10678v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:04:34"
---

# 论文速读：Auditing Chinese Web-scale Corpora via Sampled BPE Token Statistics ⚠️ Caution: this paper may include offensive and upsetting content.

## 一句话总结
本文提出 SAMPLED-BPE，一种基于流式采样与 BPE tokenizer 训练的轻量级审计流水线，以极低的计算成本（耗时缩短 148.4×、内存降低 35.8×）实现网页级中文语料库的 token 级污染画像，并同步发布包含 63 万余条记录的分层中文网络 token 数据集。

## 研究问题与动机
- **核心问题**：如何高效、精准地审计海量中文上游语料库中的隐性污染（如色情、博彩、垃圾信息等），以追溯其在 LLM 词表与输出中的污染来源。
- **现有方法不足**：
  1. **规模瓶颈**：主流中文语料库达 TB/PB 级，全量扫描成本极高（如 83TB 数据 indexing 需 99 天）。
  2. **粒度粗糙**：既有审计多停留在文档级语言分布、来源域名或质量分数，无法揭示 token 级别的隐蔽污染。
  3. **动态滞后**：中文网络污染具有高度隐晦性与快速迭代特征，预设关键词列表极易过时且难以覆盖新型变体。

## 核心贡献（创新点）
- **提出 SAMPLED-BPE 轻量审计流水线**：通过小比例流式采样 + BPE 训练 + 搜索证据辅助的 LLM 分类，将 TB 级语料库的 token 级审计从“月级”压缩至“小时级”，且保持可用精度（0.25% 采样下类别加权相对误差仅 4.25%）。
- **首个大规模中文上游语料库 token 级污染系统审计**：对 11 个开源中文语料库及 2021–2026 年共 6 期 Chinese Common Crawl 快照进行审计，首次量化揭示“污染普遍但分布不均”以及“上游中文网络内容高度污染且随时间剧烈波动”（如 2026 年快照中 Adult Content 占比高达 68.72%）。
- **发布分层中文网络 token 数据集**：公开 630,684 条 token 记录与 92,972 棵词树，每条记录包含网页搜索上下文、污染类别与判定依据，支持污染家族的溯源与组合型污染的发现。
- **与已有工作的本质区别**：区别于下游 LLM 词表反推或文档级质量审计，本文直接对上游语料进行 token 粒度采样与外部网页证据映射，实现可周期复用、可追溯词法演变的上游污染审计。

## 方法详解
SAMPLED-BPE 流水线包含四个核心阶段：
1. **流式采样 (Streaming Sampling)**：对目标语料库执行单次顺序读取，避免随机采样所需的全量标识符枚举或 shuffle，使 I/O 成本接近线性，同时保证统计代表性。
2. **BPE 训练与计数**：在采样子集上训练 BytePair Encoding (BPE) tokenizer 并统计 token 频次。选用 BPE 的原因在于其作为压缩算法能自然暴露语料中的高频共现词法模式，且无需预设关键词即可发现新型污染 token；训练基于局部相邻对频率与固定 merge budget，计数仅需一次线性 pass。
3. **类别映射 (Category Mapping)**：将 token 映射至六大内容类别：`Normal Content`, `Adult Content`, `Online Gambling`, `Online Gaming`, `Online Video`, `Anomalous`。为克服隐式/缩写 token 的歧义，引入互联网搜索结果作为上下文证据，并使用开源中文 LLM `GLM-4-32B` 作为基础分类器，基于 Zhang et al. (2025) 专家标注的 GPT 词表数据微调，测试集分类准确率达 **97.32%**。
4. **语料画像聚合 (Cor
