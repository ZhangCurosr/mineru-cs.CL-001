---
title: "Can Released LLM Vocabularies Support Token-Level Estimation of Hidden Corpora?"
source: https://arxiv.org/pdf/2608.10690v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:35:17"
field: "预训练数据透明度与词表分析"
keywords: ["BPE tokenizer", "corpus estimation", "quantile regression", "distribution transfer", "LLM transparency"]
innovations: ["发现BPE词表中token ID-比例分布的跨语料可迁移性", "提出QGDE分位数引导密度估计方法实现token级别语料比例估计", "在受控与真实场景中实现低至3%的平均相对误差"]
benchmarks: ["mC4/OSCAR语言估计", "FineWeb/Wiki/Code/Math领域估计", "SmolLM真实词表验证"]
---

# 论文速读：Can Released LLM Vocabularies Support Token-Level Estimation of Hidden Corpora?

## 一句话总结
本文证明了BPE词表在不同语料上训练后共享稳定的 token ID–比例分布，并提出了 QGDE（分位数引导密度估计）方法，仅凭发布的词表即可实现目标 token 级别的高精度语料比例估计，并在 SmolLM 真实场景下验证了有效性。

## 研究问题与动机
- **预训练语料组成是黑盒**：即使模型权重发布，具体语料构成仍不明确，影响对模型能力来源的解读。
- **现有方法粒度粗糙**：DMI（Hayase et al., 2024）只能估计语言/领域级别的粗粒度混合比例；PoCTrace（Zhang et al., 2025）仅针对特定 token 组（如污染中文 token）做粗略趋势估计。
- **词表蕴含细粒度信号**：BPE 词表按语料统计训练，token ID 反映了合并顺序和频率分布，可能承载更精细的语料比例信息。
- **核心问题**：发布的 tokenizer 词表能否支持对任意目标 token 的语料比例进行细粒度估计？

## 核心贡献（创新点）
- **发现 ID–比例分布的可迁移性**：证明不同语言和领域的 BPE 词表中，token ID 与语料比例的联合分布在 log–log 空间呈现稳定全局形状，可作为跨语料分布迁移的信号基础。
- **提出 QGDE 分位数引导密度估计框架**：通过多分位数趋势拟合近似 ID–比例分布，并结合局部密度加权将范围信号转化为 token 级别的点估计。
- **在受控与真实设置下均实现高精度估计**：token 级别平均相对误差低至 3.00%，聚合到类别级别后误差低至 3.08%，显著优于 DMI 和 PoCTrace。

## 方法详解
- **问题设定**：已知若干 source 语料的 token ID–比例对 $(t_j, r_j)$，目标是从 target tokenizer 的发布词表中估计每个 token $v_i$ 的未知比例 $r_i^*$。
- **分位数趋势拟合**：基于 Zipf 定律，在 log–log 空间对已知 ID–比例点进行分位数回归，拟合一族线性趋势 $q_\tau(x) = a_\tau + b_\tau x$，其中 $\tau$ 为分位数水平，覆盖不同比例假设（低/中/高）。
- **分位数锚点选择**：通过最大化 Quantile Anchor Coverage（QAC）选取 $K$ 个代表性分位数锚点 $\mathcal{T}_K^\star$，使各锚点趋势能覆盖尽可能多的已知 ID–比例点，避免冗余或稀疏区域。
- **局部密度加权**：对目标 token ID $t_i$，在其附近 ID 窗口内收集已知点，用高斯核计算每个分位数候选估计的局部密度支撑权重，最终以加权平均得到估计值：
  $\hat{y}_i = \sum_{\tau \in \mathcal{T}_K^\star} \frac{W_{i,\tau}}{\sum_{\tau'} W_{i,\tau'}} z_{i,\tau}$
- **类别聚合**：将 token 级别估计按 token 在各 source 类别中的出现比例分配，聚合得到类别级别混合比例。

## 实验与结果
- **数据集**：语言设置使用 mC4（source）和 OSCAR（target）的英/法/日/中；领域设置使用 FineWeb、Wikipedia、CodeParrot、OpenWeb-Math。真实验证使用 SmolLM tokenizer（训练语料：FineWeb-edu 87.30%、Cosmopedia-v2 11.11%、Python-edu 1.59%）。
- **基线**：Direct ID-ratio Transfer（直接复制 source 比例）、PoCTrace（单条中位数趋势）、DMI（源无关的粗粒度混合估计）。
- **主要结果**：
  - **Token 级别**：QGDE 在语言设置下单源平均 MRE 约 4.55%~5.80%，混源降至 3.00%~4.34%；领域设置单源约 5.94%~15.14%，混源降至 4.44%~8.41%。SmolLM 验证中 token 级别 MRE 为 5.72%~5.78%。
  - **类别级别**：QGDE 语言 MRE 约 3.08%~3.21%，领域约 5.24%~5.48%；SmolLM 类别 MRE 约 5.90%~6.08%，显著优于 DMI（语言 9.09%，领域 15.14%）。
  - **分位数锚点数**：K 从 3 增至 14 后误差显著下降并趋于饱和，K=14 为合理默认值。
  - **最强结果**：token 级别 MRE 最低 3.00%（语言混源），类别级别 MRE 最低 3.08%（语言混源）。

## 相关工作脉络
- **DMI（Hayase et al., 2024）**：利用 BPE merge rules 估计语料混合比例，但仅输出语言/领域级别的粗粒度向量，无法估计每个 token 的比例。
- **PoCTrace（Zhang et al., 2025）**：拟合单条 ID–比例中位数趋势，用于推测特定 token 组（污染中文 token）的频率，适用场景受限且为粗略范围估计。
- **数据透明度研究**：Carlini et al. (2021, 2022) 等方法通过模型输出/查询访问探测训练数据，依赖黑盒查询；本文仅需公开词表，更轻量。
- **BPE 与语料统计关系**：Sennrich et al. (2016) 奠定 BPE 合并顺序反映语料频率的基础；本文在此基础上挖掘 token ID 的全局分布可迁移性。
- **定位差异**：本文从"粗粒度混合估计"推进到"任意 token 级别的细粒度比例估计"，利用分布迁移而非单点趋势。

## 局限性与未来方向
- **缺乏真实 LLM 的 ground truth 验证**：ChatGPT、Qwen、DeepSeek 等发布词表但不公开训练语料，无法直接评估 token 级别估计精度；本文仅在 SmolLM 这一罕见案例上验证。
- **领域设置误差高于语言设置**：同领域不同语料库的采集管道差异导致 ID–比例关系对齐度较低， Code 领域尤为困难。
- **分位数锚点数量需调优**：K 增大后收益饱和，但最优 K 随 source 组合变化，缺乏统一设定准则。
- **未来方向**：扩展至更多已发布词表+语料的模型验证；探索跨语言/跨域迁移的边界条件；结合更多信号（如 merge rules）提升估计精度。

## 研究启发与可借鉴点
- **分布迁移思路可用于其他令牌化信号**：BPE merge rules、subword 频率等也可能存在跨语料可迁移分布，可拓展到词表安全性审计、数据污染检测等场景。
- **分位数回归+局部密度加权是通用估计范式**：该框架不依赖特定语料类型，可迁移到其他需要利用已知分布推测未知分布的任务。
- **实验设计值得借鉴**：从受控设置（单源/混源/目标相似）到真实案例（SmolLM）的渐进验证路径，兼顾严谨性与实用性。
- **可结合团队方向**：若团队关注预训练数据透明度、模型审计或数据去污染，本文方法可直接用于从发布词表反推语料组成，辅助数据治理决策。

## 关键术语表
- **BPE（Byte-Pair Encoding）**：一种子词分词算法，按语料中相邻符号对的频率逐步合并，生成的 token ID 隐含频率信息。
- **ID–比例分布**：token ID 与其在语料中频率比例的联合分布，在 log–log 空间呈现稳定的负相关形状。
- **分位数回归（Quantile Regression）**：拟合因变量在不同分位数条件下的回归线，此处用于捕捉 ID–比例关系的垂直 spread。
- **Quantile Anchor Coverage（QAC）**：衡量所选分位数锚点覆盖已知 ID–比例点的能力，用于自动选择代表性分位数水平。
- **局部密度加权**：以高斯核衡量目标 token 附近已知点的密度支撑，为各分位数候选分配软权重。
- **Mean Relative Error（MRE）**：平均相对误差，用于评估估计比例与真实比例的偏差。
- **DMI（Data Mixture Inference）**：基于 BPE merge rules 估计语料混合比例的源无关基线方法。
- **SmolLM**：Hugging Face 发布的轻量 LLM，其 tokenizer 和训练语料均公开，是本文真实场景验证的唯一案例。

## 可复现要素
- **数据集**：mC4、OSCAR、FineWeb、Wikipedia、CodeParrot、OpenWeb-Math、RedPajama-C4、BookCorpus、FineWebMath、SmolLM 训练语料；部分公开，部分需申请。
- **代码**：QGDE 代码已开源，见 https://github.com/qingjiesjtu/QGDE。
- **关键超参**：分位数锚点数 K（推荐 K=14），高斯核带宽 $h_x$、$h_y$（论文未明确数值，需参照实现）。
- **模型/权重**：无需预训练模型，仅需 BPE tokenizer 和已知语料的 token 计数统计。
