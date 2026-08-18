---
title: "Can Released LLM Vocabularies Support Token-Level Estimation of Hidden Corpora?"
source: https://arxiv.org/pdf/2608.10690v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:36:14"
field: "大语言模型可解释性与数据审计"
keywords: ["LLM词表分析", "预训练语料估计", "BPE分词器", "分位数回归", "数据透明度", "token级估计"]
innovations: ["发现BPE token ID-ratio分布跨语料可迁移", "提出QGDE方法实现任意token级别语料比率估计"]
benchmarks: ["mC4语言设置", "FineWeb/Wiki/Code/Math领域设置", "SmolLM真实tokenizer验证"]
---

# 论文速读：Can Released LLM Vocabularies Support Token-Level Estimation of Hidden Corpora?

## 一句话总结
本文发现不同BPE分词器的token ID–ratio分布具有跨语料的可迁移稳定性，并提出QGDE（分位数引导密度估计）方法，利用已知语料迁移ID–ratio分布结构，实现从已发布词表中对隐藏预训练语料进行**细粒度token级别比率估计**。

## 研究问题与动机
- **核心问题**：已发布LLM的权重公开了分词器词表，但预训练语料构成往往仍是黑盒；能否利用词表信息，对隐藏语料进行比粗粒度类别比例更细粒度的**token级别**比率估计？
- **现有方法不足**：DMI（Hayase et al., 2024）只能从BPE合并规则层面推断粗粒度的语料类别混合比例；PoCTrace（Zhang et al., 2025）仅针对特定token组（如污染中文token）使用单一中位数趋势做范围估计，无法覆盖任意token的细粒度估计需求。

## 核心贡献（创新点）
- **发现ID–ratio分布跨语料可迁移**：证明不同语言/领域BPE分词器的token ID–ratio分布具有稳定的全局形状，为跨语料分布迁移提供了基础信号。与已有工作的本质区别在于：以往工作关注粗粒度类别混合或特定token组，本文建立了通用的token级别估计框架。
- **提出QGDE方法**：通过拟合多个分位数趋势近似共享的ID–ratio分布，并结合局部密度加权产生token级别的点估计。与PoCTrace使用单一中位数趋势的本质区别是：QGDE保留每个token ID处合理的ratio散布范围，而非退化为单个典型值。
- **细粒度预估验证**：在受控设置及SmolLM真实tokenizer验证中，QGDE在token级别达到3.00%的MRE，聚合为类别混合后达到3.08%的MRE，显著优于基线。

## 方法详解
- **建模假设**：BPE token ID反映合并顺序，从而编码语料统计信息；log-log空间下，ID与ratio呈近似线性关系（源于Zipf定律）。
- **分位数趋势拟合**：对已知语料的$(x_j, y_j) = (\log t_j, \log r_j)$点集，对每个分位数水平$\tau$用分位数回归拟合log-linear曲线$q_\tau(x) = a_\tau + b_\tau x$，得到一族候选估计。
- **分位数锚点选择（QAC）**：定义Quantile Anchor Coverage $C(\mathcal{T}_K) = \sum_{(x_j,y_j)\in\mathcal{P}} \mathbf{1}[\min_{\tau\in\mathcal{T}_K}|y_j - q_\tau(x_j)| < h_y]$，通过网格搜索选取覆盖最多的$K$个锚点，避免冗余或稀疏区间的趋势。
- **局部密度加权**：对目标token $t_i$，在邻域$\mathcal{N}_i$内用Gaussian kernel计算各候选趋势的局部支持度$W_{i,\tau}$，最终按归一化权重加权平均得估计值$\hat{y}_i = \sum_\tau \frac{W_{i,\tau}}{\sum_{\tau'} W_{i,\tau'}} z_{i,\tau}$。
- **类别聚合**：将token级别估计按token在各类别已知语料中的出现比例$\pi_{c,i} = \frac{n_{c,i}}{\sum_{c'} n_{c',i}}$分配至各类别，得到$\hat{\alpha}_c = \sum_i \frac{\hat{r}_i}{\sum_j \hat{r}_j} \pi_{c,i}$。

## 实验与结果
- **数据集**：语言设置用mC4（英/法/日/中）作源、OSCAR作目标；领域设置用FineWeb、Wiki、CodeParrot、OpenWeb-Math作源，目标分别为RedPajama-C4、BookCorpus、RedPajama-GitHub、FineWebMath；真实场景用SmolLM tokenizer（训练语料已知：FineWebedu 87.30%、Cosmopedia-v2 11.11%、Pythonedu 1.59%）。
- **基线**：直接ID-ratio迁移（Transfer）、PoCTrace、DMI（类别级）。
- **主要结果**：
  - Token级别：QGDE（K=11~14）在语言设置MRE低至**3.00%**，在领域设置最低约**5.94%**（单源）和**4.52%**（70%-混合）；SmolLM真实场景token级别MRE为**5.72~5.78**。
  - 类别级别：QGDE聚合后语言MRE约**3.09%**，领域MRE约**5.36~5.49%**；SmolLM场景下类别MRE为**5.90~6.08**，均优于DMI（语言9.09、领域15.14）。
  - 混源比单源更稳定，但具体主导成分差异影响较小；锚点数K>14后边际收益趋于饱和。

## 相关工作脉络
- **DMI（Data Mixture Inference, Hayase et al., 2024）**：从BPE合并规则推断语料类别层面的混合比例，属于粗粒度分类信号；本文定位为在其基础上实现**token级别细粒度**估计。
- **PoCTrace（Zhang et al., 2025）**：针对特定token组（污染中文token）利用单一中位数趋势做范围估计；本文将其扩展为**通用任意token的点估计**。
- **训练数据透明度研究（Carlini et al., 2021/2022; Dodge et al., 2021等）**：依赖模型输出/查询访问推断训练数据；本文利用词表本身的结构信号，无需查询访问，属不同技术路线。
- **BPE与词表信号研究（Sennrich et al., 2016）**：建立BPE合并顺序与词频统计的关系理论基础，本文沿用并推广至跨语料迁移场景。
- **数据配比研究（Hoffmann et al., 2022; Xie et al., 2023）**：强调预训练语料构成对LLM能力的影响，本文为其提供逆向推断工具。

## 局限性与未来方向
- **真实模型的评估受限**：ChatGPT、Qwen、DeepSeek等模型开放了词表但未公开训练语料，无法直接验证token级别估计的准确性，目前仅在SmolLM（罕见公开训练语料的模型）上验证。
- **领域差异带来更大误差**：领域类别（Code vs Wiki等）的ID–ratio关系对齐度低于语言类别，导致领域估计误差更高。
- **未来方向**：随着更多LLM训练语料的公开，可对真实发布模型进行更广泛的验证；同时可探索将方法推广至非BPE分词器（如SentencePiece）的场景。

## 研究启发与可借鉴点
- **分布迁移范式**：ID–ratio分布跨语料可迁移的发现提供了一个新的信号源思路——不仅可用于语料推断，还可启发对模型训练数据构成进行非侵入式分析的工作。
- **分位数+局部密度加权框架**：该方法将"范围估计"转化为"点估计"的设计具有通用性，可迁移到类似的数据推断场景（如估计代码/数学语料占比等）。
- **实验设计借鉴**：单源/混源/目标对齐的三阶段评估设置、QAC锚点选择与K消融的规范化分析，为类似分布迁移研究提供了可复用的实验范式。
- **与团队方向结合机会**：若团队关注模型透明度/训练数据审计，可将QGDE作为词表侧的信号工具，结合模型输出侧的信息进行交叉验证，构建更全面的语料推断pipeline。

## 关键术语表
- **ID–ratio分布**：BPE token ID与其在语料中占比的对数散点分布，反映词频-排名关系的结构化模式。
- **QGDE（Quantile-Guided Density Estimation）**：本文提出的分位数引导密度估计方法，通过多分位数趋势拟合+局部密度加权实现token级别比率估计。
- **QAC（Quantile Anchor Coverage）**：分位数锚点覆盖度指标，衡量所选分位数趋势对已知ID–ratio点的覆盖情况。
- **DMI（Data Mixture Inference）**：基于BPE合并规则的粗粒度语料混合比例推断方法（Hayase et al., 2024）。
- **PoCTrace**：基于token ID推测污染中文token比例的先前方法（Zhang et al., 2025）。
- **Mean Relative Error (MRE)**：平均相对误差，用于评估估计ratio与真实ratio之间的偏差程度。
- **分位数回归（Quantile Regression）**：对不同分位数水平$\tau$拟合回归曲线的统计方法，此处用于捕获ID–ratio分布的不同垂直层级。

## 可复现要素
- **数据集**：mC4（Xue et al., 2021）、OSCAR（Suarez et al., 2020）、FineWeb（Penedo et al., 2024）、Wikipedia dumps、CodeParrot（Hugging Face, 2021）、OpenWeb-Math（Paster et al., 2024）、RedPajama-C4/GitHub（Weber et al., 2024）、BookCorpus（Zhu et al., 2015）、FineWebMath（Allal et al., 2025）、SmolLM训练语料（Allal et al., 2025）。
- **代码**：已开源，地址 https://github.com/qingjiesjtu/QGDE。
- **关键超参**：分位数锚点数K（最优K≈11~14）；垂直带宽$h_y$、水平带宽$h_x$（论文未给出具体数值，见附录A/B）；分位数回归的非对称损失函数$\rho_\tau$（Appendix A, Eq. 11）。
