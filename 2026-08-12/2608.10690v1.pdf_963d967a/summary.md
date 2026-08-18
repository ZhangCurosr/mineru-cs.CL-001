---
title: "Can Released LLM Vocabularies Support Token-Level Estimation of Hidden Corpora?"
source: https://arxiv.org/pdf/2608.10690v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:35:49"
field: "LLM 数据透明度与可解释性"
keywords: ["LLM 词表分析", "语料构成估计", "BPE tokenizer", "token-level 估计", "分布迁移", "预训练语料透明度"]
innovations: ["证明 BPE 词表的 ID-ratio 分布在跨语料间可迁移，为词表逆向推断提供理论基础", "提出 QGDE，通过多分位数趋势拟合+局部密度加权实现任意 token 的细粒度语料占比估计", "在 SmolLM 真实发布词表上验证，token-level 误差低至 3.00%，category-level 聚合后 3.08%"]
benchmarks: ["mC4", "OSCAR", "FineWeb", "CodeParrot", "SmolLM"]
---

# 论文速读：Can Released LLM Vocabularies Support Token-Level Estimation of Hidden Corpora?

## 一句话总结
本文证明了不同语料上训练的 BPE 分词器共享稳定的 token ID–corpus ratio 分布结构，并据此提出了 QGDE（分位数引导密度估计）方法，仅利用已发布的 tokenizer 词表即可对隐藏预训练语料中任意目标 token 的占比进行细粒度估计，在受控实验和真实 SmolLM tokenizer 验证中分别达到最低 3.00% 和 3.08% 的平均相对误差。

## 研究问题与动机
- **核心问题**：即使 LLM 权重已发布，其预训练语料组成仍不透明；已发布的 BPE 词表是否能为任意目标 token 的细粒度语料占比估计提供有效信号？
- **现有方法不足**：Hayase et al. (2024) 的 DMI 只能估计粗粒度的类别级混合比例，无法给出每个 token 的独立占比；Zhang et al. (2025) 的 PoCTrace 仅针对特定 token 组（被污染的中文 token），使用单一中位数趋势线，丢弃了同 ID 区间的局部变异性。
- **关键观察动机**：BPE token ID 反映合并顺序，因而编码了训练语料的统计信息，但此前工作未系统验证跨语料 ID–ratio 分布的**可迁移性**。

## 核心贡献（创新点）
- **发现 ID–ratio 分布跨语料可迁移**：证明了不同语言（英/法/日/中）和不同领域（Web/Wiki/Code/Math）训练的 BPE 分词器之间存在稳定且可量化的 ID–ratio 分布相似性，为从已知语料到未知目标分词器的分布迁移提供了理论基础；与已有工作仅关注 merge rules（DMI）或单一 token 组（PoCTrace）的本质区别在于首次系统刻画并量化了这一跨分布可迁移性。
- **提出 QGDE 通用 token-level 估计器**：通过拟合多个分位数趋势近似共享的 ID–ratio 分布，并结合局部密度加权输出任意目标 token 的点估计；与 PoCTrace 使用单条中位数曲线的本质区别在于保留了同 ID 区间内 ratio 的纵向分布信息并转为细粒度点估计。
- **在受控设置和真实 SmolLM tokenizer 上验证了最高精度的 token-level 和 category-level 估计**：token-level 平均相对误差最低达 3.00%，category-level 聚合后低至 3.08%，全面优于 DMI 和 PoCTrace 两个基线；与已有工作的本质区别在于同时实现了从粗粒度 mixture 推断到细粒度 token-level 估计的跨越。

## 方法详解
- **QGDE 整体框架**：从已知语料（source）的观测 token ID–ratio 对中学习分布结构，将其迁移到目标分词器（target）的隐藏语料上，估计任意目标 token 的 corpus ratio。
- **分位数趋势拟合（Section 4.1）**：受 Zipf 定律启发，将 token ID 和 ratio 作 log–log 变换后呈近似线性关系；对已知 ID–ratio 点集 $(\log t_j, \log r_j)$ 拟合一族分位数回归曲线 $q_\tau(x) = a_\tau + b_\tau x$，其中 $\rho_\tau$ 为不对称损失函数，每条曲线覆盖不同纵向水平的分布区域。
- **分位数锚点选择（Section 4.2）**：定义 Quantile Anchor Coverage（QAC）目标——在已知点集上，某候选锚点集合覆盖的点数（垂直带宽 $h_y$ 内的点计一次），最大化 $C(\mathcal{T}_K)$ 以选出 $K$ 个最具代表性的分位数锚点 $\mathcal{T}_K^\star$，避免冗余或稀疏区域的趋势；实验显示 $K=11\sim14$ 时覆盖度趋于饱和。
- **局部密度加权（Section 4.3）**：对目标 token $t_i$，收集其 ID 邻域内的已知点 $\mathcal{N}_i$，用高斯核计算每个分位数候选 $z_{i,\tau}$ 的局部支持权重 $W_{i,\tau}$，最终以归一化加权平均得到估计值 $\hat{y}_i = \sum_\tau \frac{W_{i,\tau}}{\sum_{\tau'} W_{i,\tau'}} z_{i,\tau}$。
- **类别级聚合（Section 7.1）**：将 token-level 估计按 token 在各已知源类别中的出现比例 $\pi_{c,i} = n_{c,i} / \sum_{c'} n_{c',i}$ 分配至各语言/领域类别，得到 $\hat{\alpha}_c = \sum_{i \in \mathcal{I}} \frac{\hat{r}_i}{\sum_j \hat{r}_j} \pi_{c,i}$。

## 实验与结果
- **数据集**：语言设置使用 mC4（源）和 OSCAR（目标）的英文/法文/日文/中文切片；领域设置使用 FineWeb/Wikipedia/CodeParrot/OpenWeb-Math（源）与 RedPajama-C4/BookCorpus/RedPajama-GitHub/FineWebMath（目标）配对；真实验证使用已发布的 SmolLM tokenizer（语料：FineWeb-edu 87.30%、Cosmopedia-v2 11.11%、Python-edu 1.59%）。
- **评估基线**：Direct ID-ratio Transfer（直接复制源 token 位置的比例）、PoCTrace（单条中位数趋势）、DMI（源无关的粗粒度 mixture 推断）。
- **主要结果**：
  - **Token-level**：QGDE（$K=11$，语言 70%-mixed）平均相对误差 **3.00%**，显著低于 Transfer（27.95%±6.92）和 PoCTrace（17.42%±7.56）；Domain 设置 QGDE 在 $K=14$ 时单源平均降至 5.94%（PoCTrace 为 29.19%±5.96）。
  - **Category-level**：QGDE 聚合后语言误差 **≈3.08%**，Domain 约 5.36%，全面优于 DMI（语言 9.09%，Domain 15.14%）。
  - **SmolLM 真实验证**：QGDE token-level 误差 5.72–5.78%，category-level 误差 5.90–6.08%，均低于各自基线。
- **核心结论**：更多分位数锚点有帮助但收益快速饱和（$K\approx11\sim14$ 后边际增益接近零）；混合源语料比单源更稳定；语言类别估计比领域类别更容易（语言 3.00% vs 领域 ~5.9%）。

## 相关工作脉络
- **DMI（Hayase et al., 2024）**：将 BPE merge rules 视为混合信号，估计粗粒度的语言/领域比例向量；本文与之定位的差异在于 DMI 止步于类别级粗比例，QGDE 进一步推至 token-level 细粒度估计。
- **PoCTrace（Zhang et al., 2025）**：利用单条 ID–ratio 中位数趋势估计特定 token 组（被污染中文 token）的流行度；本文与之定位的差异在于 PoCTrace 仅适用于特定 token 子集且无法捕获纵向分布，QGDE 面向任意通用 token 并提供点估计。
- **训练数据提取/去重类工作（Carlini et al., 2021, 2022；Dodge et al., 2021）**：依赖模型输出或查询访问来检测记忆化样本和 benchmark 污染；本文定位差异在于完全不依赖模型查询，仅从公开词表推断。
- **数据透明度审计（Bommasani et al., 2023；Longpre et al., 2024）**：通过问卷调查和文档分析评估模型透明度；本文提供的是基于词表的自动化定量推断方法。
- **SmolLM 语料组成验证（Allal et al., 2025）**：本文为少数能使用真实已发布 tokenizer + 真实训练语料同时做验证的研究；相比其他仅理论分析的工作，提供了实证支撑。

## 局限性与未来方向
- **真实发布模型的 ground truth 稀缺**：ChatGPT、Qwen、DeepSeek 等主流模型仅发布词表而未公开训练语料，限制了直接验证；未来需等待更多模型开源训练数据才能做更全面评估。
- **领域类别估计精度低于语言类别**：不同领域的 vocabularies 重叠度更高（如 Code vs Wiki 结构差异大），ID–ratio 关系对齐程度较弱，token-level 误差相对较高。
- **聚合后 anchor 增益减弱**：分类别聚合会平滑或重新分配 token-level 改进的收益，category-level 估计对 anchor 数量不敏感，限制了进一步优化空间。
- **未来方向**：拓展到更多已发布 LLM 的验证（如有训练语料公开）、探索跨语种混合目标（非均匀 mixture）的估计性能、以及将该信号用于 corpus contamination 检测等下游任务。

## 研究启发与可借鉴点
- **ID–ratio 分布迁移的可复用范式**：将"跨分布稳定结构 → 分位数拟合 → 局部密度加权"的方法论可迁移至其他从离散发布信号推断隐藏统计量的场景（如从发布词表推断多语言覆盖度、从模型卡信息反推数据组成）。
- **QAC（Quantile Anchor Coverage）选择策略**：基于最大化覆盖度的贪心/网格搜索选锚点方法简洁有效，可作为多维度分布拟合中选择代表线的通用技巧，适用于任何具有排序关系的分布估计任务。
- **Token-level 到 category-level 的双阶段设计**：先做细粒度 token 估计再做加权聚合，相比直接做 coarse-grained mixture 推断能保留更多信息；这种"细到粗"的两阶段思路可用于数据溯源、模型审计等需要多尺度输出的任务。
- **SmolLM 作为真实世界验证基准**：本文利用了少数拥有完整训练语料的已发布模型，团队可探索是否有类似可用基准来验证其他从发布信号推断的方法。

## 关键术语表
- **ID–ratio 分布**：BPE token ID（反映合并顺序）与其在训练语料中占比之间的联合分布，是本文推断的核心信号。
- **QGDE（Quantile-Guided Density Estimation）**：本文提出的分位数引导密度估计方法，通过多分位数趋势拟合 + 局部密度加权实现 token-level 语料占比估计。
- **QAC（Quantile Anchor Coverage）**：分位数锚点覆盖度目标函数，用于选择最优分位数锚点集合，最大化已知 ID–ratio 点的覆盖比例。
- **DMI（Data Mixture Inference）**：Hayase et al. (2024) 提出的基于 BPE merge rules 的粗粒度语料混合比例推断方法。
- **PoCTrace**：Zhang et al. (2025) 提出的基于单条 ID–ratio 中位数趋势的特定 token 组流行度估计方法。
- **Mean Relative Error（MRE）**：平均相对误差，本文用于量化 token-level 和 category-level 估计精度的主要指标。
- **Directional Transfer Similarity**：基于 KL 散度和目标熵归一化的方向性分布相似度度量，用于量化源词表到目标词表的迁移质量。
- **BPE（Byte-Pair Encoding）**：字节对编码，LLM 常用的子词分词算法，通过迭代合并高频相邻 token 对构建词表。

## 可复现要素
- **数据集**：mC4（Xue et al., 2021）、OSCAR（Suarez et al., 2020）、FineWeb（Penedo et al., 2024）、Wikipedia dumps、CodeParrot、OpenWeb-Math（Paster et al., 2024）、RedPajama-C4/GitHub、BookCorpus、FineWebMath（Allal et al., 2025）、SmolLM 训练语料（FineWeb-edu、Cosmopedia-v2、Python-edu）——均为公开数据集。
- **代码**：QGDE 代码已开源，见 https://github.com/qingjiesjtu/QGDE。
- **关键超参**：分位数锚点数量 $K$（最优约 11–14）、垂直带宽 $h_y$、水平带宽 $h_x$——论文未给出具体数值，需在代码或附录中查找。
- **基线实现**：PoCTrace 和 DMI 作为对比基线，具体实现细节论文未提及，需参考各自原始代码。
