---
title: "LabelFusion-TS-Fusing-Large-Language-Models-Transformer-Enco"
source: https://arxiv.org/pdf/2608.11753v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 11:17:11"
---

# 论文速读：LabelFusion-TS: Fusing Large Language Models, Transformer Encoders, and Financial Time Series for Monetary-Policy Stance Classification

## 一句话总结
本文在联邦公开市场委员会（FOMC）货币政策立场分类任务上，将公开的市场时间序列作为独立模态接入原 LabelFusion 架构，构建 LabelFusion-TS 系统；实验表明该融合模型在仅使用约 240 句人工标注数据时即可超越零样本大语言模型，概率集成后加权 F1 达到 70.2%。

## 研究问题与动机
- **核心问题**：金融文本分类器几乎仅依赖纯文本输入，而人类分析师评估央行声明、财报或新闻时必然同时参考利率、通胀、资产价格等市场背景。
- **现有方法不足**：
  1. 既有金融多模态研究多聚焦“文本特征辅助预测价格/波动率”，反向的“时间序列辅助文本分类”探索极少。
  2. 央行沟通语言高度模糊与对冲（如 “act as appropriate”），相同措辞在不同市场周期下含义相反，纯文本模型在时序划分测试集中无法获知上下文环境。
  3. 高质量人工标注稀缺（训练期仅千级句子），纯 LLM 零样本在严格时序评测下存在明确性能天花板。

## 核心贡献（创新点）
1. **多模态可插拔的 LabelFusion-TS 架构**：将金融时间序列 Transformer 作为独立专家接入原 LabelFusion 的中层融合框架，实现文本、LLM 语义与市场周期的三方投票融合。
2. **银标自训练 + 金标微调的两阶段 RoBERTa 预训练**：利用同模型生成的 13,017 条银标句子学习任务分布，再用少量人工标注完成金标校准，有效缓解低资源场景下的表示退化。
3. **严格的时序验证与公开时间序列流水线**：基于 FRED 库构建 6 维日频市场窗口（滞后 45 天避免未来信息泄露），在 “train-on-past, test-on-future” 切分下证实时间序列模态提供文本模型缺失的环境先验。
4. **低预算超越零样本 LLM 的实证**：仅 20%（≈240 句）人工标注即实现交叉超越，27 组种子概率集成额外提升近 3 个百分点，为“弱独立模态强协同融合”提供新证据。

## 方法详解
- **整体融合机制**：三个专家独立训练后冻结，输出拼接送入小型 MLP 投票网络 $\hat{y} = f_{\text{MLP}}([h; z; m]) \in [0,1]^3$，其中 $h$ 为 RoBERTa CLS embedding，$z$ 为 LLM 零样本投票向量，$m$ 为时间序列 Transformer 输出；移除 $m$ 即退化至原 LabelFusion。
- **Text Expert**：RoBERTa-large，max 128 tokens。第一阶段（silver）：13,017 条银标句子训练 2 epochs，学习任务形状；第二阶段（gold）：人工标注数据（含 96 条无法精确日期但 ≤2014 的附加句）fine-tune ≤6 epochs，混合精度 fp16，class-weighted cross-entropy 损失。
- **LLM Expert**：gemma4:31b，temperature=0，直接复用 Shah et al. (2023) 的零样本分类提示词输出 $\{0,1\}^3$；同一模型同一提示词同步用于生产银标。
- **Market Expert**：标准 Transformer Encoder（4 层、宽 64、8 heads、FFN 128），输入为 126 个交易日窗口（联邦基金利率、2年期国债收益率、期限利差、纳斯达克指数对数、CPI同比、失业率），每通道取窗口内相对首日变化量并在训练集上标准化。采用 30 天利率变化率的三分类进行市场周期预训练（1962–2015），再在目标任务上 fine-tune。
- **训练与集成策略**：三项训练均使用随机种子 {1, 2, 3}，产生 $3 \times 3 \times 3 = 27$ 套系统；报告均值，并构建 probability ensemble（逐样本平均三分类后验分布）作为单一可部署模型。
- **数据构建**：Shah et al. (2023) 基准原始 2,379 句，去重/冲突后 2,312 句；通过精确匹配+归一化包含匹配恢复 1,692 句（73%）精确发布日期；按 2015年9月切分，训练 1,274 句，测试 418 句（2015–2022）。

## 实验与结果
- **数据集与指标**：FOMC 立场分类基准，主指标为加权 F1 $F1_w = \sum_c \frac{n_c}{N} \cdot \frac{2P_cR_c}{P_c+R_c}$，中性类占 48% 对平均分主导。
- **基
