---
title: "Auditing Chinese Web-scale Corpora via Sampled BPE Token Statistics <sup>\u0011</sup>Caution: this paper may include offensive and upsetting content."
source: https://arxiv.org/pdf/2608.10678v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:38:45"
field: "大语言模型语料审计与数据质量"
keywords: ["语料审计", "中文大模型", "BPE tokenizer", "网络污染", "Common Crawl", "数据质量", "token 级分析", "采样估计"]
innovations: ["提出 SAMPLED-BPE 采样+BPE 训练实现 token 级语料污染轻量化审计；首次用层级 token 树结构表征组合污染与共享核心-专属尾部结构；量化 0.25% 采样率下的精度-成本权衡（148.4× 加速、35.8× 内存削减、4.25% WRE）"]
benchmarks: ["11 个公开中文语料（OSCAR/mC4/HPLT/CulturaX/CWT/ROOTS/WanJuan/MAPCC/SkyPile/CCI3/WuDao）", "Chinese Common Crawl 2021–2026 六期快照", "GLM-4-32B 微调分类器（97.32% test accuracy)"]
---

# 论文速读：Auditing Chinese Web-scale Corpora via Sampled BPE Token Statistics

> ⚠️ Caution: this paper may include offensive and upsetting content.

## 一句话总结
论文提出 SAMPLED-BPE，一种**轻量级 BPE token 级语料审计流水线**，通过采样+训练 BPE tokenizer+搜索引擎映射六类内容标签，实现 TB 级中文语料的污染估算；在 0.25% 采样率下仅 4.25% 相对误差即可把 1TB 语料的审计时间从数月缩短到小时级。

## 研究问题与动机
- **中文 LLM 词表已暴露上游污染信号**（ChatGPT 长 token 含垃圾/色情、Codex 输出插入口号式博彩词串），但现有工作只检查下游模型词表（Zhang et al., 2025），**未回溯到上游语料本身**。
- **全量扫描代价太高**：主流中文语料达数百 GB–TB 级（如 83TB 数据在 128 核 CPU 上索引需 99 天）；逐文档语义分析更不可行。
- **已有分析粒度太粗**：语言分布、来源域名、文档级质量分只能描述“宏观构成”，**无法暴露 token 级污染分布**。
- **中文网络污染隐式且快速演化**：固定关键词表既会漏掉新生变体又迅速过时（如博彩平台名迭代、成人站换肤）。

## 核心贡献（创新点）
1. **方法**：提出 SAMPLED-BPE 流水线——采样流式读取 → BPE 训练与计数 → 搜索引擎证据 + GLM-4-32B 微调分类器映射 6 类标签 → 聚合为 corpus profile；**与已有工作的本质区别**在于全程在 token 级运行、不依赖预定义词表，首次把"海洋污染监测式采样估计"引入语料审计。
2. **精度-成本权衡的量化论证**：系统展示不同采样率下 token coverage、Spearman/Pearson 相关、category weighted relative error（WRE）与 runtime/memory 的变化曲线；**核心数字**——0.25% 采样率下：token coverage 76.83%，Spearman 86.88%、Pearson 99.93%，category WRE ≈ 4.25%，同时带来 148.4× 加速与 35.8× 内存削减。
3. **11 个公开中文语料 + 6 年 Common Crawl 快照的实证审计**：揭示"污染广泛但不均衡"（OSCAR 83.38%、mC4 25.41% vs. WanJuan/MAPCC/SkyPile/CCI3/WuDao 均 < 1%）以及 2026 快照 Adult 占比高达 68.72% 且随年份波动。
4. **层级中文 web token 数据集**：发布 630,684 条 token 记录 + 92,972 棵树，每条含 web 上下文、类别、解释；**首次把"token 组合污染"（如"北京"+"赛车"→ 博彩）和"共享核心 vs. 语料专属尾部"** 以可追溯结构形式开放。

## 方法详解
### 流水线四步（Figure 3）
1. **Streaming sampling**：单遍顺序扫描目标语料，按流式前缀/概率抽子集；比 random sampling（需先物化 shuffled index）快 22.4× 且覆盖误差 < 1%（Appendix C）。
2. **BPE training & counting**：对子集训练 BytePair Encoding tokenizer（Sennrich et al., 2016），再用同一 tokenizer 线性过一遍做频率计数；选择 BPE 三因：(i) 高频 token 暴露语料特征词法；(ii) 新污染词无需预设词表即可浮出水面；(iii) 训练只依赖局部相邻 pair 频次+固定 merge 预算，计数只需单次线性 pass。
3. **Category mapping**：每个 token 以搜索引擎结果（网页摘要/标题）作为上下文证据，由 GLM-4-32B 微调分类器判定 6 类之一：Normal / Adult / Online Gambling / Online Gaming / Online Video / Anomalous；训练/测试集来自 Zhang et al. (2025) 的 GPT 中文词表专家标注，微调后 test accuracy = 97.32%。
4. **Corpus profiling**：token 级记录 ratio、预测类别、搜索证据、分类理由；category 级聚合为各类别 prevalence 与总污染比。

### 精度度量
- **Token 级**：coverage = |V_Full ∩ V_sample| / |V_Full|；Spearman & Pearson 相关在共享 token 上做。
- **Category 级**：weighted relative error (WRE)，分别在 shared tokens 和 all tokens 上计算；Figure 4 显示即使 coverage 不完备，最低采样率下 WRE ≈ 5%。

### 采样率选取（Appendix B, Table 3）
对 11 个语料分别取 0.19%（HPLT）– 14%（mC4）不等；log-linear 插值估计 full-token WRE 全部 < 4%，多数在 2%–3%。

## 实验与结果
### 基线对比
- **Full vs. Sampled**：不同采样率（Figure 4）——0.25% 下 token coverage 76.83%，Spearman 86.88%，Pearson 99.93%，category WRE ≈ 4.25%。
- **Streaming vs. Random sampling**（Table 4，CC 0.10%）：22.4× 加速，token coverage 仅降 0.57%，category WRE 略降 2.69%。
- **Runtime/Memory**（Figure 5）：0.25% 采样率下 148.4× 加速、35.8× 内存削减；更小样本代价更低但精度下降。

### 11 个公开中文语料审计（Table 1）
| 语料 | 总污染 | Adult | Gambling | Gaming | Video | Anomalous |
|---|---|---|---|---|---|---|
| OSCAR | **83.38%** | 76.10 | 0.56 | 0.16 | 5.02 | 1.53 |
| mC4 | **25.41%** | 0.31 | **23.17** | 0.46 | 0.32 | 1.15 |
| HPLT | 4.95 | 1.95 | 1.61 | 0.25 | 0.72 | 0.43 |
| CulturaX | 3.36 | 0.15 | 2.45 | 0.16 | 0.18 | 0.42 |
| CWT | 2.35 | 0.05 | 0.22 | 0.29 | 0.06 | 1.73 |
| ROOTS | 2.02 | 0.23 | 0.00 | 0.00 | 0.00 | 1.79 |
| WanJuan | 0.75 | 0.04 | 0.15 | 0.17 | 0.05 | 0.33 |
| MAPCC | 0.69 | 0.10 | 0.08 | 0.09 | 0.04 | 0.38 |
| SkyPile | 0.61 | 0.06 | 0.04 | 0.05 | 0.04 | 0.42 |
| CCI3 | 0.55 | 0.16 | 0.00 | 0.01 | 0.00 | 0.37 |
| WuDao | 0.50 | 0.10 | 0.01 | 0.06 | 0.01 | 0.31 |

关键发现：
- **污染"广泛但不均衡"**：多语种泛爬 pipeline（OSCAR/mC4/HPLT/CulturaX）污染最高；注重中文特供/可信源/质量过滤的语料（WanJuan/MAPCC/SkyPile/CCI3/WuDao）< 1%。
- **Cleaning reduces but does not eliminate**（Appendix D, Table 5）：CWT-Cleaner 从 2.35% → 0.50%；ROOTS-Public 2.02% → 1.37%，残留仍主要在 Anomalous 和 Adult。
- **污染类别非单一**：OSCAR 以 Adult 为主（76.10%），mC4 以 Gambling 为主（23.17%），干净语料的残留多呈 Anomalous。

### 跨语料共享 vs. 专属污染（Section 4.2）
- **方向性 coverage**：Cleaner 语料间互相覆盖 > 50%（如 MAPCC↔SkyPile）；泛爬语料间呈非对称包含（CulturaX 中 69.2% 的污染 token 出现在 mC4，反之仅 13.8%）。
- **整体结构**：11 语料中共 106,671 个污染 token，96.386%（102,816）只出现在 1 个语料；≥3 个语料共现的仅 1,542（1.446%）；≥8 语料共现的"共享核心"252 个，跨越全部 5 个污染类（Figure 8）。
- **共享核心示例**："老司机"（Adult 委婉）、"威尼斯人"（Gambling 赌场名）。

### 中文 Common Crawl 时间演化（Section 5, Table 2）
| 年份 | 总污染 | Adult | Gambling | Gaming | Video | Anomalous |
|---|---|---|---|---|---|---|
| 2021 | 35.35 | 24.73 | 5.31 | 0.40 | 3.93 | 0.98 |
| 2022 | 61.87 | 54.53 | 1.05 | 0.16 | 4.75 | 1.38 |
| 2023 | 62.90 | **55.86** | 0.54 | 0.23 | 4.89 | 1.37 |
| 2024 | 34.83 | 28.01 | 1.48 | 0.33 | 3.86 | 1.16 |
| 2025 | 38.01 | 30.53 | 0.74 | 0.22 | 5.05 | 1.47 |
| 2026 | **79.52** | **68.72** | 0.21 | 0.07 | **8.39** | 2.12 |

- **Adult 高且波动**：2023 年 55.86% → 2024 年 28.01% → 2026 年 68.72%。
- **Gambling 单调下降**：5.31%（2021）→ 0.21%（2026），与跨境赌博整治力度加强一致。
- **Video 缓升**：3.93% → 8.39%。
- **Token 层替换规律**（Figure 9, 10）：Normal/Gaming/Video 年度 Jaccard 高、5 年 persistent；Adult/Gambling/Anomalous 年度替换快、5 年 persistence 弱。
- **Top-k 敏感性**（Appendix F）：k=10 显稳定、k=100 与全词表显波动，Adult/Anomalous 尤其明显。

### 层级中文 web token 数据集（Section 6）
- 630,684 条 token 记录 + 92,972 棵树。
- **两类可追溯模式**：(i) 同族污染 → 根为最小 recurring subtoken（如 "浪小辉"）；(ii) 跨类组合污染 → 两个正常子 token 组合出污染（如 "菲律宾"+"申博" → 博彩品牌；"北京"+"赛车" → PK10 投注）。
- 附录 H（Figure 18–25）给出了完整搜索证据与树形展开。

## 相关工作脉络
1. **Corpus Curation & Auditing**（Dodge et al., 2021; Kreutzer et al., 2022; Soldaini et al., 2024; Penedo et al., 2023, 2024）：聚焦 source/domain/quality/dedup 等文档级或源级属性；本文定位差异：**首次把审计粒度下放到 token 级**并覆盖污染类别。
2. **Chinese Polluted Tokens in LLM Vocabularies**（Yang, 2024; OpenAI Developer Community, 2026; Zhang et al., 2025）：观察 ChatGPT/Codex 词表中出现的垃圾/成人/博彩 token；本文定位差异：**从下游词表回溯到上游语料**，建立污染源头档案。
3. **Chinese Web-scale Corpora**（OSCAR/mC4/HPLT/CulturaX/ROOTS/CWT/WanJuan/MAPCC/SkyPile/CCI3/WuDao；Yuan et al., 2021; Chen et al., 2023; He et al., 2023; Du et al., 2024; Wei et al., 2023; Wang et al., 2024）：本文首次对这批语料做**统一的 token 级污染横评**。
4. **Common Crawl 时间切片研究**（Li et al., 2025 Tic-LM；Hagar & Bandy, 2025）：关注时间持续性 benchmark 与实用审计工具；本文提供**中文 web 污染时间演化的首个系统量化**。
5. **Sampling & Streaming for Large Corpora**（MacAvaney et al., 2019; Jiang et al., 2021）：指出新词涌现与词表过时挑战；本文用 **BPE 高频 token 自动浮现新污染词**，避免硬编码词表。
6. **Marine Pollution Monitoring 类比**（Karydis & Kitsiou, 2013）：灵感来源——子样本估计整体污染；本文把该思路形式化为 **SAMPLED-BPE 流水线**并在 NLP 语境验证。

## 局限性与未来方向
- **仅覆盖中文**：其他语言未审计；迁移需多语言标注员、语言特定搜索证据与可能不同的污染类别定义。
- **分类器依赖搜索证据**：GLM-4-32B 在英文/其他语言上的搜索回退质量未知；token 级别"隐式/委婉/缩写"特征在低资源语言更难判别。
- **翻译可读性损失**：部分 token 的字面翻译会丢失网络语境中的污染 cues；作者自陈翻译仅作可读性辅助。
- **抽样本身的偏差边界**：极端低采样率（< 0.1%）下 coverage 与 WRE 恶化；对长尾稀有污染 token 可能漏检。
- **未覆盖的污染维度**：当前 6 类（Normal/Adult/Gambling/Gaming/Video/Anomalous）之外，其他类型（如政治敏感、隐私泄露、仇恨言论）未被系统化评估。
- **未来方向**：扩展至多语种、自动化 periodic re-audit 机制、探索 BPE tokenizer 与其他分词器（SentencePiece/unigram）的对比。

## 研究启发与可借鉴点
1. **"采样+子集 BPE 训练"作为一种通用语料审计原语**：不依赖硬编码词表即可让高频污染 token 自动浮现；可复用到任何语言的 web-scale 语料体检。
2. **streaming sampling 的工程收益明确**：比 random sampling 快 22× 且精度损失 < 1%；对 TB+ 语料管线是低成本替换。
3. **层级 tree 结构解决"token 组合污染"这一经典难点**：把"两正常子 token 拼出污染"（如"北京赛车"）以可追溯方式表达，比 flat keyword list 更适合下游清洗和 periodic audit。
4. **Jaccard 重叠 + top-k 敏感性双视角刻画词表演化**：区分"头部稳定、尾部快速迭代"与"全域持续变化"两类 category；可迁移到任何语言的内容毒性追踪。
5. **端到端"采样→BPE→搜索→分类→树形数据集"全开源流水线**：可被后续工作直接接驳，用于新语料的快速污染体检或作为 pre-training filter 的有效性验证基线。

## 关键术语表
- **SAMPLED-BPE**：本文提出的轻量级 token 级语料审计流水线，核心思想是采样子集训练 BPE tokenizer，以高频 token 浮现污染信号。
- **Streaming sampling**：单遍顺序扫描语料时按概率抽取子集，避免 random sampling 所需的 index 物化与 shuffle，I/O 近乎线性。
- **BPE (BytePair Encoding)**：重复合并高频相邻 unit 的子词压缩算法；本文用作 token 级"词法指纹"提取器，而非下游语言模型分词器。
- **Category mapping**：用搜索引擎结果作为上下文证据、经 GLM-4-32B 微调分类器把每个 token 归入 6 类之一（Normal/Adult/Gambling/Gaming/Video/Anomalous）。
- **Weighted Relative Error (WRE)**：按 token 频次加权计算的类别占比相对误差，用于衡量采样估计相对于 Full 语料的偏差。
- **Hierarchical Chinese web token dataset**：630k+ 条记录、92k+ 棵树的层级数据集，每条含 web 上下文/类别/解释，支持污染溯源。
- **Jaccard overlap (temporal)**：相邻年或跨 5 年的 token 集合交集比，刻画类别内词汇替换速度与持久度。
- **Shared core vs. corpus-specific tail**：11 语料间共现 ≥8 次的 252 个污染 token 构成"共享核心"，其余 96%+ 为各语料专属尾部的现象学结论。

## 可复现要素
- **数据集**：11 个公开中文语料（OSCAR/mC4/HPLT/CulturaX/CWT/ROOTS/WanJuan/MAPCC/SkyPile/CCI3/WuDao）及 6 个 Chinese Common Crawl 快照（2021–2026）均为公开可用；层级 token 数据集也已发布（论文标注脚注 1）。
- **代码/权重**：论文未明确给出仓库链接（全文未出现 GitHub URL）；SAMPLED-BPE 流水线描述完整，可据方法复现。
- **分类器**：GLM-4-32B（开源模型）微调；训练/测试集来自 Zhang et al. (2025) 的 GPT 词表专家标注，划分比例未详述。
- **关键超参**：BPE vocab size、merge ops 上限、采样率（0.19%–14% 因语料而异，见 Table 3）；实验硬件：96 核 CPU + 768 GB RAM。
- **度量与基线**：token coverage、Spearman/Pearson、category WRE、runtime、peak RSS；Appendix C/D/F 提供 streaming vs. random、clean vs. unclean、top-k 敏感性等补充实验。
