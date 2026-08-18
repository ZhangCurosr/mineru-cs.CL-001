---
title: "MOST BIOMEDICAL PUBLICATIONS SHOW SIGNS OF LLM-ASSISTED WRITING"
source: https://arxiv.org/pdf/2608.10715v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:40:46"
field: "计算语言学与学术出版"
keywords: ["LLM使用率估计", "学术写作", "词汇频率分析", "生物医学文献", "语言变化检测"]
innovations: ["提出无偏估计方法突破仅下界限制", "跨章节/国家细粒度分析LLM使用模式"]
benchmarks: ["PubMed Central (PMC)", "频率差下界方法", "混合模型方法", "分布重叠方法"]
---

# 论文速读：MOST BIOMEDICAL PUBLICATIONS SHOW SIGNS OF LLM-ASSISTED WRITING

## 一句话总结
本文提出并验证了一种基于词汇频率变化、无偏估计LLM使用率的新方法，应用于PubMed Central (PMC) 开放获取生物医学论文，发现截至2025年底，**89%的论文显示出LLM辅助写作或编辑的迹象**。

## 研究问题与动机
- **核心问题**：准确估计学术界实际使用LLM的比例，为政策制定提供依据。
- **现有方法不足**：
  - 基于词汇频率差距的方法（如Kobak et al., 2025）只能给出**下界估计**，系统性低估真实使用率。
  - 基于混合模型的方法依赖特定提示词和LLM生成文本，结果对模型选择敏感。
  - 已有LLM检测工具在2023年前的数据上效果有限，且多次被证明不准确。
- **研究缺口**：缺乏一种无需假设、能够超越下界限制、适用于大规模真实语料的可靠估计方法。

## 核心贡献（创新点）
1. **提出无偏估计框架**：基于预训练-后训练词汇频率变化，通过优化阈值T选择最优marker words集合，突破已有方法的仅下界限制。
2. **跨章节细化分析**：首次系统比较LLM使用率在论文不同章节（Abstract、Methods、Results、Discussion）的差异。
3. **跨国别比较**：分析第一作者国家/地区与LLM使用率的关系，揭示语言障碍驱动的使用差异。
4. **大规模实证验证**：应用于119万篇PMC论文，发现89%的论文存在LLM使用迹象，高于先前所有文献报道。

## 方法详解
- **核心思想**：LLM辅助写作会引入特定"marker words"（如these, potential, delves）的过量使用，通过比较这些词在ChatGPT发布前后（2018-2022 vs 2023-2025）的频率变化，可推断LLM使用比例。
- **基本公式**：
  - 观测频率：$q(t)$ = 包含至少一个marker word的文档比例。
  - 人类文本反事实频率：$\hat{p}_{\mathrm{human}}(t)$，基于2018-2022年线性回归外推。
  - 频率差：$\Delta = q - \hat{p}_{\mathrm{human}}$，作为LLM使用率的下界。
- **改进估计量**：
  - 更强下界：$\hat{\beta}_{\mathrm{LB}} = \frac{q - \hat{p}_{\mathrm{human}}}{1 - \hat{p}_{\mathrm{human}}} \geq \Delta$
  - 最优选择：对marker words集合G(T)按阈值T优化，选择使$\hat{\beta}_{\mathrm{LB}}$最大的T（标准误差<0.025）。
- **标准误差计算**：通过delta方法传播$q$的二项方差和$\hat{p}_{\mathrm{human}}$的回归方差。
- **模拟验证**：用已知真实$\beta$的10万文档模拟数据验证，估计误差$|\hat{\beta} - \beta| < 0.02$。

## 实验与结果
- **数据集**：PubMed Central (PMC) Open Access Subset，1,194,287篇2017-2025年英文论文（含Abstract、Introduction、Methods、Results、Discussion）。
- **基线对比**：
  - Kobak et al. (2025) 频率差方法：2024年摘要估计14-16%。
  - Siler (2026) 分布重叠法：2025年57%。
  - 本文方法：2025年全年**89%**（完整论文）。
- **主要结果**：
  - 2025年底Abstract使用率：68%；Methods：54%；Discussion：78%。
  - 按255词随机片段比较：Discussion使用率最高（68%），Methods最低（32%）。
  - 跨国别：韩国最高（85%），中国（82%），台湾（80%）；英国最低（28%），美国（39%）。
  - 母语英语国家vs非母语国家：2025年分别37% vs 72%。
- **关键结论**：LLM使用率逐年上升（2023年仅19%，2025年达89%），且存在显著学科/章节/国家差异。

## 相关工作脉络
1. **Kobak et al. (2025)**：识别379个LLM关联marker words并提出频率差方法，但仅给出下界；本文在其基础上提出无偏估计并验证。
2. **Gray (2024, 2025)**：预选择marker words方法，同样仅提供下界估计。
3. **Liang et al. (2024a, 2025)**：混合模型方法，依赖LLM生成文本，对不同模型/提示敏感。
4. **Siler (2026)**：基于分布重叠法，隐含假设人机词汇分布不重叠，导致低估；本文方法更宽松。
5. **LLM检测工具研究**（Akram, 2024; Liu & Bu, 2024; Weber-Wulff et al., 2023）：证明检测器准确性不足，尤其对早期数据。
6. **跨国调查**（Liao et al., 2024; Lin et al., 2025; Prakash et al., 2025）：报道使用率差异，本文提供客观量化证据。

## 局限性与未来方向
- **假设依赖**：$\hat{p}_{\mathrm{human}}$基于线性外推，长期可能失效（人类可能模仿LLM风格或刻意回避marker words）。
- **marker words局限性**：基于词汇层面，无法捕捉内容层面的幻觉或结构性伪造。
- **国家比较偏差**：跨国分析使用固定marker words集，未针对各国优化，部分估计可能偏低。
- **未来方向**：结合内容级检测、追踪非英文文献、评估LLM对学术诚信的实际影响。

## 研究启发与可借鉴点
1. **方法论迁移**：基于词汇频率变化的无偏估计框架可推广至其他领域（如计算机视觉论文、社会科学文献）。
2. **实验设计借鉴**：阈值优化策略（grid search + 标准误差过滤）可作为稳健参数选择的范式。
3. **跨学科合作**：结合计算语言学与科学计量学，为科研政策提供数据驱动的决策依据。
4. **后续研究方向**：可探索LLM使用与论文影响力、错误率、可重复性之间的相关性。

## 关键术语表
**LLM-assisted writing**：使用大型语言模型辅助撰写或编辑文本的行为。
**Marker words**：在LLM生成文本中出现频率显著高于人类写作的词汇（如these, potential, delves）。
**Frequency gap**：观测词汇频率与反事实人类频率之差，作为LLM使用率的保守下界。
**Counterfactual frequency ($\hat{p}_{\mathrm{human}}$)**：假设无LLM影响时，基于历史趋势外推的预期词汇频率。
**Lower bound estimator ($\hat{\beta}_{\mathrm{LB}}$)**：更强下界估计量，通过$(q - \hat{p}_{\mathrm{human}})/(1 - \hat{p}_{\mathrm{human}})$计算。
**Threshold optimization (T)**：选择marker words子集的最优频率阈值，使估计量方差最小化。
**PubMed Central (PMC)**：美国国立卫生研究院维护的开放获取生物医学文献数据库。
**Linguistic convergence**：非母语研究者因使用LLM导致其写作风格向母语者趋同的现象。

## 可复现要素
- **数据集**：PubMed Central Open Access Subset（https://pmc.ncbi.nlm.nih.gov/tools/openftlist/），已公开。
- **代码**：GitHub仓库https://github.com/kobaklab/llm-usage-in-pmc，已开源。
- **关键超参**：marker words集合379个；阈值网格19个值；标准误差阈值0.025；回归期2018-2022；裁剪长度255词。
