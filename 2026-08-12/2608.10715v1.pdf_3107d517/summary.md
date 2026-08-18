---
title: "MOST BIOMEDICAL PUBLICATIONS SHOW SIGNS OF LLM-ASSISTED WRITING"
source: https://arxiv.org/pdf/2608.10715v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:40:39"
field: "科学计量学与AI伦理"
keywords: ["LLM辅助写作", "学术出版", "词频分析", "PubMed Central", "检测指标", "学术诚信"]
innovations: ["提出分母修正的无偏LLM使用率估计方法（从下界升级为近似真实值）", "首次在全文段落粒度系统刻画Discussions与Methods的LLM使用差异", "揭示LLM驱动的全球学术英语语言趋同现象"]
benchmarks: ["PubMed Central Open Access Subset (1,194,287 papers, 2017-2025)"]
---

# 论文速读：MOST BIOMEDICAL PUBLICATIONS SHOW SIGNS OF LLM-ASSISTED WRITING

## 一句话总结
本文提出了一种基于词频变化的无偏估计方法，用于测算生物医学文献中 LLM 辅助写作的普及率；应用于 PubMed Central（PMC）开放获取论文后得出：截至 2025 年底，**89% 的论文**显示出 LLM 辅助写作的语言痕迹。

## 研究问题与动机
1. **核心问题**：如何可靠估计学术出版物中 LLM 辅助写作的使用比例，以支撑科研伦理政策制定。
2. **现有方法不足（频率缺口法）**：Kobak et al. (2025)、Gray (2025) 等方法仅基于 marker 词频率缺口（$\Delta = q - \hat{p}_{\text{human}}$）给出下界，系统性低估真实使用率。
3. **现有方法不足（混合模型法）**：Liang et al. (2024b)、Geng & Trotta (2024) 通过 prompt LLM 生成对照文本再拟合混合模型，结果强烈依赖于所选模型和 prompt，不同工作对同一数据给出差异巨大的估计（如 arXiv CS 摘要：18% vs 35%）。
4. **现有方法不足（LLM 检测器）**：已有 detector（如 GPTKIT、Smodin 等）在 2023 年低使用率时期训练/校准，且多次被验证为不准确；分布重叠法（Siler, 2026）隐含假设人机词频分布无重叠，与事实不符。

## 核心贡献（创新点）
1. **提出无偏估计框架**：通过构造 marker 词集合 $G(T)$ 并最大化 $\hat{\beta}_{\text{LB}} = \max_T \frac{q - \hat{p}_{\text{human}}}{1 - \hat{p}_{\text{human}}}$，突破此前所有方法仅能提供下界的局限，实现对真实 LLM 使用比例的可靠估计。与频率缺口法的本质区别在于分母修正（除以 $1-\hat{p}_{\text{human}}$ 而非 1），使估计值显著大于 $\Delta$。
2. **首次在全文段落粒度上系统刻画 LLM 使用差异**：将方法应用于 PMC 论文的 Abstract、Introduction、Methods、Results、Discussion 各部分，发现 Discussion 段落 LLM 使用概率（68%）是 Methods 段落（32%）的两倍，即便在 Methods 内部整体使用率也超 50%。
3. **揭示跨国语言趋同现象**：母语英语国家与非母语国家在 LLM marker 词频率上从 2022 年的显著差异收敛至 2025 年几乎一致，表明 LLM 使用正在驱动学术英语的"语言均质化"。
4. **通过仿真实验验证估计方法的高精度**：在模拟数据（$n=100{,}000$ 文档、$m=500$ 个 marker 词）上验证，对所有 $\beta \in [0,1]$ 有 $|\hat{\beta} - \beta| < 0.02$，确认方法的可靠性。

## 方法详解
**步骤一：数据收集与预处理**
- 从 PMC Open Access Subset 下载，筛选 2017–2025 年英文论文，要求含 Introduction、Methods、Results、Discussion 四个部分，每部分不少于 250 词，最终得 1,194,287 篇论文。
- 379 个预定义的 LLM-associated marker 词（如 these, potential, delves, exacerbating, invaluable 等）来自前作（Kobak et al., 2025）。

**步骤二：词频时间序列建模**
- 对每个 marker 词，计算其在各月/年的文档使用频率 $q(t)$（至少出现一次的文档占比）。
- 以 2018–2022 年（ChatGPT 发布前）数据拟合线性回归，外推得到 $\hat{p}_{\text{human}}(t)$，即若无 LLM 时该词的预期人类写作频率。
- 频率缺口 $\Delta = q - \hat{p}_{\text{human}}$ 给出保守下界。

**步骤三：改进的下界估计量**
- 设 $\beta$ 为 LLM 辅助写作的论文比例，$p_{\text{LLM}}$ 为 LLM 文本中词频，则有：
$$q = (1-\beta)p_{\text{human}} + \beta p_{\text{LLM}} \quad \Rightarrow \quad \beta = \frac{q - p_{\text{human}}}{p_{\text{LLM}} - p_{\text{human}}}$$
- 以 $p_{\text{LLM}} \leq 1$ 替换，得到更强下界：
$$\hat{\beta}_{\text{LB}} = \frac{q - \hat{p}_{\text{human}}}{1 - \hat{p}_{\text{human}}} \geq \Delta$$
- 用 Delta 方法传播 $\hat{p}_{\text{human}}$ 和 $q$ 的不确定性，计算标准误，丢弃标准误 > 0.025 的阈值。

**步骤四：阈值优化与最终估计**
- 将 marker 词按 2024 年频率排序，选取频率低于阈值 $T$ 的词构成集合 $G(T)$。
- 在 19 个阈值网格上搜索，选择使 $\hat{\beta}_{\text{LB}}$ 最大的 $T$，取 $\hat{\beta} = \max_T \hat{\beta}_{\text{LB}}$ 作为最终估计。
- 该最大值通常在 $q$ 接近 1 时取得，此时 $p_{\text{LLM}} \approx 1$ 的近似更合理，因此下界较紧。

**步骤五：段落比较控制长度偏置**
- 因长段落更易包含 marker 词，额外使用随机 255 词片段（对应中位摘要长度）重新估计各段落使用率，实现公平比较。

**步骤六：仿真验证**
- 生成 $n=100{,}000$ 文档、$m=500$ 词的数据，设定已知 $p_{\text{human}}$（Gamma 分布）、$p_{\text{LLM}} = (1+\delta)p_{\text{human}}$（$\delta \sim U(0.5, 5)$）和真实 $\beta$，验证估计偏差 $|\hat{\beta} - \beta| < 0.02$。

## 实验与结果
**数据集**：PubMed Central (PMC) Open Access Subset，1,194,287 篇英文生物医学论文（2017–2025 年）。

**主要结果（Table 1，年度估计）**：

| 论文部分 | 2023 | 2024 | 2025 | 2025年12月 |
|---|---|---|---|---|
| Abstract | 9% | 31% | 53% | **68%** |
| Full paper | 19% | 52% | 77% | **89%** |
| Discussion | 14% | 43% | 68% | 78% |
| Methods | 11% | 29% | 46% | 54% |

**按 255 词片段的段内比较（消除长度偏置）**：
- Discussion 使用率最高：2025 年全年 57%，12 月达 **68%**。
- Methods 使用率最低：12 月仅 **32%**，但与全文估计（54%）差距最大，说明 Methods 内 LLM 使用不均匀（可能集中在特定子段落）。
- Abstract 紧随 Discussion：12 月达 **67%**。

**国家/地区差异（2025 年全文估计）**：
- 最高：韩国（85%）、中国（82%）、台湾（80%）。
- 最低：英国（28%）。
- 母语英语国家（澳、加、爱、新、南非、英、美）平均 $\hat{\beta}=0.37$；非母语国家平均 $\hat{\beta}=0.72$。
- 2022 年 marker 词频率与 2025 年 LLM 使用率呈负相关（$r<0$），表明 pre-LLM 语言水平较低的国家使用 LLM 弥补语言差距的程度更高。

**最强结果**：全文 2025 年 12 月估计 $\hat{\beta} = 0.89$（89%），显著高于此前所有文献报道的同时段估计（Table S1 显示 2024 年多数估计在 12%–36% 之间）。

## 相关工作脉络
1. **Kobak et al. (2025)**：识别 PMC 中 379 个 LLM marker 词并提出频率缺口法，本文在此基础上引入分母修正，从下界升级为近似无偏估计；二者方法本质区别在于是否用 $1-\hat{p}_{\text{human}}$ 归一化。
2. **Gray (2024, 2025)**：使用预定义 marker 词做频率缺口分析，估计 2024 年 full paper LLM 使用率为 16%，本文指出其方法系统性偏低（本文 2024 年为 52%）。
3. **Liang et al. (2024a, 2025)**：通过 prompt LLM 生成对照文本并用混合模型估计使用比例，依赖 prompt/model 选择；本文方法无需生成对照文本，避免了这一偏差来源。
4. **Siler (2026)**：基于 marker 词每千词使用率分布的重叠面积估计，隐含假设人机分布无重叠；本文方法不依赖分布形态假设，仅依赖线性外推。
5. **LLM detector 类研究（Akram, 2024; Liu & Bu, 2024; Picazo-Sanchez & Ortiz-Martin, 2024）**：依赖二分类检测器，文中多次被引用为不准确；本文完全绕过检测器路径，属不同方法论流派。
6. **Survey 类研究（Wiley, 2025/2026; Kwon, 2025; Liao et al., 2024）**：自行报告的使用率（如 Wiley 2026 报告 >70%）与本文估算一致，但存在自我报告偏差；本文提供了客观文本证据作为交叉验证。

## 局限性与未来方向
1. **线性外推的时效性**：$\hat{p}_{\text{human}}$ 依赖 2018–2022 年的线性趋势外推，距 ChatGPT 发布时间越远，假设越脆弱；随着人类作者模仿 LLM 风格或刻意避免 marker 词，基准线可能漂移。
2. **人为趋同效应未分离**：人类作者可能无意识模仿 LLM 写作风格（Yakura et al., 2024 引用），导致估计偏高；但刻意回避 marker 词又会低估，两者效应方向和大小尚不明。
3. **跨国比较使用固定 marker 集**：Figure 3 各国比较采用统一 marker 集（未针对各国优化阈值），部分估计可能为下界。
4. **Methods 段内使用不均**：255 词片段估计（32%）与全文估计（54%）差距大，说明 LLM 在 Methods 中使用不均匀（可能集中在 protocol 描述等非核心实验部分），需进一步细粒度分析。
5. **未覆盖非开放获取论文**：PMC OA Subset 仅代表部分生物医学文献，全量 PMC 及非生物医学领域的泛化有待验证。

## 研究启发与可借鉴点
1. **分母修正思路可迁移至其他领域**：$\hat{\beta}_{\text{LB}} = \frac{q - \hat{p}_{\text{human}}}{1 - \hat{p}_{\text{human}}}$ 这一从频率缺口到归一化估计的转换公式，可直接推广至任何可通过 marker 词追踪技术渗透率的领域（如 AI 辅助代码生成、AI 辅助翻译等）。
2. **阈值优化策略值得复用**：在词频排序基础上搜索最优 $T$ 使估计最紧，同时以标准误为约束排除不稳定估计，这一"最优下界→近似真实值"的思路可应用于其他基于词频变化的估计问题。
3. **段落粒度对比实验设计**：使用随机等长片段（255 词）消除段落长度偏置，使不同章节间可比——这一设计可推广至任何需比较文本不同部分特征的量化研究。
4. **仿真验证框架**：构建含已知 $\beta$、$p_{\text{human}}$、$p_{\text{LLM}}$ 的模拟数据集来验证估计方法精度，此验证范式可复用于评估其他新兴的 LLM 使用率估算方法。
5. **跨国语言趋同分析视角**：将 marker 词 pre-LLM 频率作为"语言掌握程度"代理变量，揭示 LLM 对全球学术英语的均质化效应——这一分析框架可为科技政策研究提供量化依据。

## 关键术语表
**LLM marker words**：在 LLM 生成文本中出现频率显著高于人类写作、可用于追踪 LLM 渗透率的虚词/功能词（如 these, potential, delves, exacerbating），本身无语义内容但具风格标识性。
**$\hat{p}_{\text{human}}(t)$**：基于 ChatGPT 发布前（2018–2022）词频线性趋势外推得到的反事实人类写作词频，代表"若无 LLM 干预时的预期使用率"。
**$\hat{\beta}_{\text{LB}}$**：经 $1-\hat{p}_{\text{human}}$ 归一化后的 LLM 使用率下界估计量，比原始频率缺口 $\Delta$ 更紧（数值更大），本文以其最大值作为最终估计。
**Threshold T**：用于筛选 marker 词集合 $G(T)$ 的频率阈值，控制参与估计的词数；过小则信号弱，过大则 $\hat{p}_{\text{human}} \approx 1$ 导致估计不稳定。
**Random 255-word crop**：从各论文段落中随机抽取的 255 词片段，用于消除段落长度差异对 marker 词检出率的偏置影响。
**Language convergence**：非母语国家学者使用 LLM 后，其 marker 词使用频率与母语国家趋同的现象，反映 LLM 对学术写作语言多样性的均质化效应。

## 可复现要素
- **数据集**：PubMed Central (PMC) Open Access Subset，完全公开（https://pmc.ncbi.nlm.nih.gov/tools/openftlist/），论文于 2026-01-23 基线版本下载。
- **代码**：已开源，GitHub：https://github.com/kobaklab/llm-usage-in-pmc
- **关键超参**：pre-LLM 回归期 2018–2022（60 个月）；标准误截断阈值 0.025；$\hat{p}_{\text{human}}$ 截断值 0.999；crop 长度 255 词；marker 词总数 379；阈值网格 19 个点（log-scale 均匀分布 + 0.03–0.7 区间加密）。
