---
title: "Counting-Documents-Is-Not-Counting-Text-Unit-Bias-in-Web-PDF"
source: https://arxiv.org/pdf/2608.16390v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:27:53"
field: "大规模语料构建与评估"
keywords: ["PDF语料库", "数据质量评估", "截断损失", "token分布", "单位偏差", "Common Crawl"]
innovations: ["首次对Web-PDF语料进行token加权统计，揭示文档级与token级指标的严重偏差", "量化Common Crawl截断策略的文本损失（55-62%）及5MiB政策的增量收益", "提出双引擎提取对比方法精确测量截断恢复率"]
benchmarks: ["CC-MAIN-2021-31-PDF-UNTRUNCATED", "PyMuPDF", "PDFium"]
---

# 论文速读：Counting-Documents-Is-Not-Counting-Text-Unit-Bias-in-Web-PDF

## 一句话总结
本文揭示了当前Web-PDF语料库统计指标的"单位偏差"问题：各语料库虽以token数宣传规模，却以文档数为单位计算覆盖率、OCR路由等所有指标，导致对语料质量的评估严重失真——例如Common Crawl的截断策略实际损失了55–62%的文本而非表面所示的23%。

## 研究问题与动机
- **核心矛盾**：FinePDFs、CCpdf、PDFA等主流PDF语料库均报道token总量，但所有处理比率（过滤率、OCR路由率、覆盖率）均以文档为单位计算，忽略了PDF长度分布的极端偏斜。
- **已有工作缺陷**：FinePDFs自身已观察到其文档中位数字符数约68k（其他语料库仅11–13k），却仍以每文档成功率报告OCR路由；CCpdf报告每步处理文档数但从不给出token计数。
- **实际影响**：由于长文档承载不成比例的文本量，基于文档计数的统计会严重低估截断、过滤等操作对实际文本规模的损害。

## 核心贡献（创新点）
1. **首次对Web-PDF语料库进行token加权分析**：将每一 headline 统计指标（长度分布、工具链占比、截断影响）同时以文档数和token数呈现，揭示两单位间的巨大偏差。
2. **量化了文本浓度的极端不均衡性**：发现Gini系数达0.807，仅3.02%的文档占据一半token，50页以上文档占5%却含53.53%文本。
3. **首次量化Common Crawl截断限制的文本代价**：1 MiB上限导致23.06%文档被截断，但损失63.08%文本；即使提升至5 MiB，仍有30.19% token被截断。
4. **开源代码**：提供了可复现上述所有分析的代码。

## 方法详解
- **数据集**：CC-MAIN-2021-31-PDF-UNTRUNCATED，包含7,932,654个唯一PDF文档，32,570,135,761个tokens（Apache Tika 2.8.0计数，ICU whitespace分词）。
- **Token单位来源**：使用语料库元数据中的`tika_eval_num_tokens`字段，该值以10,000,000为右 censoring 上限（仅2个文档触及上限）。
- **截断分析框架**：利用provenance表中的`cc_truncated`、`fetched_status`、`fetched_length`及WARC字节范围，识别并配对截断文档与其从origin重获取的完整版本。
- **双引擎提取验证**：使用PyMuPDF（MuPDF）和PDFium（Chromium）两个独立解析器，在截断版本和完整版本上分别提取文本，比较恢复率——两引擎在完整文档上结果高度一致（差异<1.7%），但在截断文件上差异达8.4×，说明修复能力是截断恢复的关键变量。
- **工具链分类**：基于producer/creator字符串将文档归类为LaTeX、PowerPoint、Microsoft Word等工具链家族。

## 实验与结果
- **核心数据集**：CC-MAIN-2021-31-PDF-UNTRUNCATED（790万文档，326亿token）。
- **长度分布（表1）**：
  - >50页文档：占4.42%文档，含49.71%文本（11.24×均值）
  - 4–30页文章类：占31.64%文档，含26.87%文本
  - 1页文档（传单/表格）：占24.96%文档，仅含2.70%文本
- **工具链占比（§4.3）**：LaTeX工具链占1.66%文档但含4.05%文本（2.43×）；PowerPoint占2.24%文档仅0.82%文本；Word占24.95%文档、19.37%文本。
- **截断影响（表2）**：
  - 1 MiB上限：影响23.06%文档，占63.08%文本
  - 5 MiB上限（反事实推演）：影响5.94%文档，但仍有30.19%文本被截断
- **文本恢复率（§4.5）**：
  - PyMuPDF：恢复截断文本的11.4%（20.7%页面）
  - PDFium：恢复1.4%（1.9%页面）
  - 两引擎合计，约55–62%的语料库文本被截断策略永久丢失
- **5 MiB增量收益（表3）**：提升截断上限使5–10 MiB文档恢复率从4.88%升至22.44%（+17.6pp），但>25 MiB的文档仅提升2–3pp， diminishing returns明显。

## 相关工作脉络
1. **FinePDFs (Kydlíček et al., 2025)**：报道3T token规模，以文档数报告OCR路由率和截断恢复率，但未提供token加权分析——本文直接对比并修正其统计口径。
2. **CCpdf (Turski et al., 2023)**：按文档数和语言统计表列处理流程，不报告token总量；本文指出其无法反映文本层面的真实损失。
3. **PDFA (Montalvo & Wightman, 2024)**：基于相同时期语料，按文档大小（>100 MB）和渲染速度过滤，但不报告任何丢弃率——本文揭示了此类size-based过滤的文本代价。
4. **Nemotron-CC (Su et al., 2025)**：已采用token加权报告"+57.4%高质token"，本文扩展了这一思路至PDF专属场景。
5. **OmnidocBench (Ouyang et al., 2025)**：仅测量已解析文档的提取保真度，不报告覆盖率——本文补充了缺失的覆盖率维度。

## 局限性与未来方向
- **局限性**：
  - 恢复率仅覆盖67%的被截断文档（1.2M/1.83M），剩余33%因技术错误（OOM、hang）未被分析。
  - producer/creator字符串自报，8.08%无法分类，LaTeX占比为下界估计。
  - page_size仅记录首页，landscape分类近似。
  - 5 MiB结果为反事实推演（基于2021文件大小），非实测。
  - 仅比较文本层解析器，未评估OCR管道。
- **未来方向**：需开发size-aware的抓取策略（而非简单提高固定上限），以及对多模态PDF使用OCR管道进行类似分析。

## 研究启发与可借鉴点
1. **统计口径的双轨制**：任何涉及长度偏斜数据（如代码库、论文、PDF）的质量评估，必须同时报告文档级和token/字符级指标，避免误导性摘要。
2. **截断损失的量化方法**：通过配对"截断版本"与"完整重获取版本"，并用双独立引擎提取比较，可精确测量信息损失——此范式可迁移至任何有截断/截断风险的数据集评估。
3. **工具链标注的价值**：按producer/creator分类可揭示内容来源结构（如学术vs商业），为后续采样加权或分层处理提供依据。
4. **5 MiB政策的评估框架**：论文展示了如何用历史数据反事实推演新策略效果，并为团队提供了可复用的评估脚本模板。
5. **Gini系数的引入**：将经济学中的不平等度量（Gini=0.807）用于token分布，为快速诊断语料偏斜提供了一个简洁的单一数字指标。

## 关键术语表
- **Unit Bias（单位偏差）**：语料库统计中以文档数为单位报告比率，但实际文本量集中在少数长文档，导致评估失真。
- **Gini Coefficient（基尼系数）**：衡量不平等程度的指标，本文语料token分布Gini=0.807，表明极端集中。
- **Linearized PDF（线性化PDF/Fast Web View）**：Adobe优化格式，将首页和交叉引用表置于文件前端，使部分下载仍可渲染，但反而降低了截断恢复率。
- **Truncation Cap（截断上限）**：Common Crawl对单个WARC记录的存储限制（从1 MiB升至5 MiB），超出的文件被裁剪存储。
- **Re-fetch Recovery（重获取恢复）**：对截断文档从源URL重新完整下载的策略，本文测量其实际文本恢复率。
- **Text-bearing Document（含文本文档）**：成功提取出文本的PDF，区别于纯扫描/图片PDF。

## 可复现要素
- **数据集**：CC-MAIN-2021-31-PDF-UNTRUNCATED，来自Digital Corpora（https://digitalcorpora.org/corpora/file-corpora/cc-main-2021-31-pdf-untruncated/）
- **代码**：已开源于 https://github.com/lfoppiano/cc-wacky-pdf
- **关键超参**：Apache Tika 2.8.0、PyMuPDF 1.28.2、PDFium（pypdfium2 5.12.1）、token计数右 censoring 上限10,000,000
- **截断上限对比**：1 MiB vs 5 MiB
