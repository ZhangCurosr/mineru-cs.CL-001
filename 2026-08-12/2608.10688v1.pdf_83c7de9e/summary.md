---
title: "Leveraging Human Reading Behavior for Keyphrase Extraction: A Webcam-based Eye-tracking Corpus"
source: https://arxiv.org/pdf/2608.10688v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:33:45"
field: "认知NLP / 关键短语提取"
keywords: ["Keyphrase Extraction", "Eye-tracking", "Webcam-based", "Chinese Academic Text", "Reading Behavior", "CLIS-ET", "Attention Mechanism"]
innovations: ["基于低成本网络摄像头的中文学术阅读眼动数据收集平台", "构建CLIS-ET字符级眼动语料并验证其对KPE的增益", "揭示早期加工(FFD)与累积加工(TFD)特征在不同数据规模下的互补效应"]
benchmarks: ["Abstract-320 (五折交叉验证)", "Abstract-5190 (4:1划分)"]
---

# 论文速读：Leveraging Human Reading Behavior for Keyphrase Extraction: A Webcam-based Eye-tracking Corpus

## 一句话总结
本文提出一种低成本网络摄像头眼动追踪方法，采集中文图书情报学（LIS）学术摘要阅读行为数据，构建**Chinese LIS Eye-Tracking Corpus (CLIS-ET)**，并将字符级眼动特征（FFD、FN、TFD）融入多种关键短语提取（KPE）模型，验证了人类阅读行为信号对KPE性能的显著提升作用。

## 研究问题与动机
1. **现有KPE方法忽视人类阅读行为**：当前KPE研究主要基于文本内部线索（词频、位置、句法模式、上下文语义），基本未考虑关键短语与读者注意力分配和认知加工之间的关联。
2. **中文学术领域眼动语料匮乏**：既有公开眼动语料（如ZuCo、GECO）多来源于社交媒体、新闻、小说等通用文本，缺乏针对学科性较强的学术文献的眼动数据，且中文资源更少。
3. **传统眼动设备成本高、部署受限**：实验室级眼动仪（如EyeLink 1000 Plus）价格高达$25,000–40,000+，需要专门硬件、校准流程和受控实验室环境，难以支撑大规模数据采集。
4. **低质量/小规模数据导致模型性能波动**：在数据有限场景下（如Abstract-320，仅320篇摘要），如何有效利用少量高质量眼动信号提升KPE性能，尚缺乏系统研究。

## 核心贡献（创新点）
1. **开发了轻量级网络摄像头眼动数据收集平台**：基于Flask框架集成开源SearchGazer库，仅需消费级网络摄像头即可实现实时同步的多参与者眼动数据采集，大幅降低硬件与部署成本。与以往依赖昂贵实验室设备的方案本质不同。
2. **构建了首个面向中文LIS学术摘要的字符级眼动语料CLIS-ET**：包含10名参与者的3个核心眼动指标（FFD、FN、TFD），填补了中文学术阅读眼动资源的空白，区别于现有英文通用文本语料。
3. **系统性评估了单特征与组合眼动特征在多类KPE架构中的增益效果**：覆盖BiLSTM系列、BERT家族（BERT/MacBERT/RoBERTa）及T5风格模型（Randeng-T5-784M），揭示了不同模型结构下眼动信号的差异化贡献模式，区别于以往仅聚焦单一模型的工作。
4. **证实了早期加工特征（FFD）与累积加工特征（TFD）的互补价值**：小样本下早期特征增益显著，大数据量下累积特征更有效，为认知驱动NLP的任务设计提供了实证依据。

## 方法详解
- **数据收集平台**：基于Flask + SearchGazer（JavaScript眼动库），采样频率最高60Hz（间隔约16.67ms），支持浏览器端实时同步多参与者眼动坐标。实验前进行9点校准，阅读过程中鼠标跟随光标进一步细化校准。
- **有效注视阈值**：设定有效注视区间为16.7ms–1500ms（对应2–91个连续采样点），保留有效注视数据。
- **字符级特征提取**：计算每个字符的**FFD（首次注视持续时间）**、**FN（注视次数）**、**TFD（总注视时间）**，跨参与者取均值；若某字符有效数据少于50%参与者，则标记为无效。
- **KPE模型架构**：
  - 递归神经网络类：BiLSTM（基线）、BiLSTM+CRF、Att-BiLSTM、Att-BiLSTM+CRF；眼动特征作为外部特征向量引入，或在Attention模型中作为注意力指示符引导序列标注。
  - 预训练语言模型类：BERT-base-chinese、Chinese-RoBERTa-wwm-ext、MacBERT-base、Randeng-T5-784M（中文适配的T5架构）。
- **数据集构建**：
  - **Abstract-320**：320篇摘要（1,215句，64,969字符），眼动特征字符级覆盖率94.69%，采用五折交叉验证。
  - **Abstract-5190**：5,190篇摘要（15,999句，804,210字符），眼动覆盖率54.65%，按4:1划分为训练/测试集。
- **评估指标**：以F₁@K（K=3/5/10）为核心指标，真阳性仅限于出现在摘要中的关键短语（不含作者标注但摘要未出现的短语）。

## 实验与结果
- **最佳单特征结果**：在Abstract-320小样本上，FFD对BERT模型带来最大提升（F₁@3从37.05%升至39.39%，+2.83%）；在Abstract-5190大样本上，RoBERTa+FN达到最优F₁@5=44.51%。
- **最佳组合特征结果**：在Abstract-5190上，**FN+TFD组合**在Att-BiLSTM+CRF模型中表现最佳——F₁@5从33.80%提升至36.79%（**+2.99%**），F₁@10从31.40%提升至35.46%（+4.06%）。
- **稳定性规律**：小数据集上FN+TFD和FFD+TFD组合提升更明显；大数据集上FFD+FN和FFD+FN+TFD组合效果更稳定。
- **统计显著性**：通过配对t检验（五折交叉验证），在Abstract-320上，多数特征组合在BiLSTM系列和BERT家族中达到p<0.05甚至p<0.01显著水平（见表9）。

## 相关工作脉络
1. **ZuCo语料（Hollenstein et al., 2018）**：英文Wikipedia内容的眼动+EEG多模态语料，被Zhang & Zhang (2021)用于微博KPE，但未覆盖中文学术词汇。
2. **Yan et al. (2024)**：基于ZuCo语料融合眼动与EEG特征增强微博KPE，本文在方法思路上延续但转向学术领域与中文场景。
3. **AKEGIS（Scholz et al., 2019）**：利用用户搜索日志提取关键词，为基于行为的关键词提取提供了下游参考，但与眼动数据的认知精细度不同。
4. **GECO/GECO-CN语料**：双语小说语料，内容非学术领域，无法直接迁移至学术摘要KPE任务。
5. **传统眼动设备对比**：Table A1对比了EyeLink 1000 Plus、Tobii Pro Spectrum等高端设备与SearchGazer方案的成本/部署差异，本文选择轻量化路线而非高精度路线。

## 局限性与未来方向
- **领域单一、样本量有限**：仅针对LIS学科10名参与者，泛化到其他学科和多群体尚需验证。
- **眼动覆盖率随数据规模衰减**：Abstract-5190仅54.65%字符有有效眼动特征，限制了大规模模型的充分利用。
- **网络摄像头精度低于实验室设备**：SearchGazer最高60Hz采样，无法替代EyeLink等千赫兹级设备的高精度眼动实验需求。
- **三特征叠加未带来持续提升**：FFD+FN+TFD三者组合有时反而不如两两组合稳定，特征融合策略有待优化。
- **未来方向**：扩展至更多学科领域、提升追踪质量与验证流程、探索更强预训练架构下的精细化特征融合策略。

## 研究启发与可借鉴点
1. **低成本眼动数据采集框架可迁移**：基于SearchGazer + Flask的方案无需昂贵设备，可用于其他学术领域（如医学、计算机科学）的中文阅读眼动语料构建。
2. **字符级眼动特征避免中文分词问题**：对中文文本而言，直接在字符级别计算FFD/FN/TFD绕过了分词误差的影响，值得在其他中文NLP任务中借鉴。
3. **早期加工与累积加工特征的互补设计**：FFD反映初始认知加工深度，TFD反映反复加工与回归行为，两者组合在不同数据规模下各有优势，可作为多特征融合的设计参考。
4. **眼动信号作为注意力机制的引导信号**：在Att-BiLSTM中用眼动特征作为attention indicator而非简单拼接，为认知信号与注意力机制的交互提供了新思路。
5. **可与本团队方向结合**：若团队研究面向中文学术文本的信息检索或自动摘要，可将CLIS-ET的字符级眼动特征作为额外信号接入现有模型，验证跨任务迁移效果。

## 关键术语表
- **Keyphrase Extraction (KPE)**：从文本中自动识别并抽取具有代表性关键短语的NLP任务。
- **First Fixation Duration (FFD)**：读者首次注视目标字符的持续时间，反映初始认知加工负荷。
- **Fixation Number (FN)**：对目标字符的总注视次数，反映该字符受到的注意力聚焦频率。
- **Total Fixation Duration (TFD)**：对目标字符的所有注视时长之和，反映深度语义加工与回归行为。
- **CLIS-ET**：Chinese Library and Information Science Eye-Tracking Corpus，本文构建的中文LIS学术摘要眼动语料。
- **SearchGazer**：开源JavaScript眼动追踪库，利用消费级网络摄像头实现浏览器端实时 gaze 推断。
- **Att-BiLSTM+CRF**：融合注意力机制的双向LSTM结合条件随机场的序列标注模型，本文KPE任务的最强基线之一。

## 可复现要素
- **数据集**：CLIS-ET（眼动数据）与Abstract-5190（KPE标注数据）；代码仓库：https://github.com/yan-xinyi/Reading_ET_System
- **眼动特征**：字符级FFD、FN、TFD，跨10名参与者取均值
- **模型**：BiLSTM、BiLSTM+CRF、Att-BiLSTM、Att-BiLSTM+CRF、BERT-base-chinese、Chinese-RoBERTa-wwm-ext、MacBERT-base、Randeng-T5-784M
- **超参数**：max_length=512；Abstract-320上BiLSTM类epochs=30/lr=0.01、Attention类epochs=65/lr=0.005；预训练模型epochs=10/lr=5e-5；Abstract-5190上BiLSTM类epochs=30/lr=0.003、预训练模型epochs=8/lr=5e-5
- **硬件**：NVIDIA A100 GPU（主实验）；NVIDIA RTX 4090 GPU（Randeng-T5补充实验）
