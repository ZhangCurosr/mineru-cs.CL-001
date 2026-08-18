---
title: "Falsehood-and-Impossibility-Are-Diferent-Directions-in-an-AI"
source: https://arxiv.org/pdf/2608.12852v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:59:04"
field: "语言模型可解释性"
keywords: ["mechanistic interpretability", "truth probing", "impossibility", "sparse autoencoders", "semantic anomaly", "representation geometry"]
innovations: ["发现AI内部真理性与不可能性方向近似正交，不可能命题不表现为极端虚假", "构建五条件对照刺激集首次实验检验模型对不可能性的表征"]
benchmarks: ["Modality Set (15 topic families × 5 conditions)", "Philosophical Set (17 seed families, 85 prompts)"]
---

# 论文速读：Falsehood-and-Impossibility-Are-Different-Directions-in-an-AI's-Representation-of-Language

## 一句话总结
本研究通过探针实验发现，Gemma 3 4B IT 模型在内部激活空间中，真实性与不可能性的表征方向近似正交——不可能命题并非虚假命题的极端情况，而是更接近语义异常（semantic anomaly）；但模型的语言输出却将两者混为一谈。

## 研究问题与动机
- 哲学上"虚假"（contingent falsehood）与"不可能"（necessary falsehood/impossibility）是两类不同的表述失败，AI模型内部是否也如此区分尚不清楚。
- 现有truth probing研究表明，事实性命题的真假可从模型残差流中线性解码，但"不可能性"是否仅作为"极端虚假"被编码，此前从未检验。
- 实践层面，语言模型能生成流畅且具有说服力的虚假或矛盾陈述，若其内部未区分这两类失败，则难以可靠检测AI生成的误导内容。
- 当前方法缺乏对"不可能性"这一概念的操作性定义与对照实验设计，无法直接检验模型是否具备此类表征。

## 核心贡献（创新点）
1. **构建"不可能性vs虚假"的对照刺激集**：设计了15个主题族、每个包含五种条件（真实、偶发虚假、不可能、语义异常、概率性可能）的标准化数据集，首次使该问题可实验检验。
2. **发现真理性与不可能性方向的正交性**：在线性探针实验中发现，truth probe在区分不可能与虚假时处于随机水平（AUC=0.20），而impossibility probe完美分离两者（AUC=1.00），两方向余弦值<0.12。
3. **揭示必要虚假命题的表征邻近性**：不可能命题在激活空间中更接近语义异常而非偶发虚假，这与模型口头分类行为（将虚假标为contradiction）形成双重解离。
4. **在稀疏自编码器特征中复现探针几何**：Gemma Scope 2的SAE特征在layer 15同样显示不可能选择性，且这些特征同时对异常句响应，但不响应偶发虚假，与探针结论一致。
5. **提出"模型自身表征地理学"的研究视角**：建议从"模型如何划分可说与不可说"出发而非检验模型是否复现人类分类，为机制可解释性提供新思路。

## 方法详解
- **刺激设计**：哲学集（85条，17个家族）与模态集（75条，15个主题族×5条件）；每种条件对应不同语义/逻辑状态。
- **模型与提取**：Gemma 3 4B IT（2,560维残差，16层），逐提示记录最后一个prompt token后的残差状态（160×35×2560）。
- **线性探针**：L2正则化逻辑回归（C=0.1，类别平衡权重），5折交叉验证按主题族分组holdout；三类探针：truth（false vs true）、impossibility（impossible vs true+false+improbable）、anomaly（anomalous vs true+false+improbable）。
- **方向正交性检验**：在每一层拟合两组探针，计算决策向量的余弦相似度。
- **跨数据集迁移**：在模态集训练impossibility probe，直接应用于哲学集，衡量AUC。
- **表面基线**：使用字符n-gram TF-IDF（1-2gram词级、3-5gram字符级）、token长度作为对照。
- **统计显著性**：对impossibility peak层标签置换1,999次（保留每族至少一条不可能），Bonferroni校正35层后P=0.018。
- **稀疏自编码器分析**：使用官方Gemma Scope 2 JumpReLU SAE（layer 15, 16k字典），比较不同capacity checkpoint（小：~16 active features/prompt；大：~90）中的选择性特征。

关键公式描述：
- 探针目标：$f(x) = \text{sign}(w^T x + b)$，其中$w$由L2正则化logistic regression求解。
- 性能指标：balanced accuracy（各折均值）、AUC（decision value曲线下面积）。
- 正交性度量：$|\cos(w_{\text{truth}}, w_{\text{impossibility}})| < 0.12$（depth>10之后）。

## 实验与结果
- **模型口头分类**：哲学集总准确率55.3%；模态集中15条偶发虚假有12条被标为"contradiction"，与"married bachelor"等不可能命题归为同一类。
- **truth probe**：峰值balanced accuracy 0.93（depth≈10），但对impossible vs false仅AUC=0.20（随机水平）。
- **impossibility probe**：峰值balanced accuracy 0.97（depth 16 / transformer layer 15），impossible vs true AUC=0.98，impossible vs false AUC=1.00，对false vs true AUC=0.51（无信息）。
- **anomaly probe**：impossible vs false AUC=0.96，anomaly可早期（depth 8）解码。
- **方向几何**：truth与impossibility方向在各层余弦≤0.12，近似正交；impossibility与anomaly方向余弦约0.4，部分重叠但可区分（impossible vs anomalous AUC=0.89）。
- **跨集迁移**：impossibility probe从模态集→哲学集AUC最高0.72；反向0.79。
- **SAE特征**：大capacity checkpoint中，Feature 14761在impossible上 firing rate=0.67，Feature 9201=1.00；两者在false上分别为0.07和0.80（后者误报较高），在anomalous上分别为0.40和0.67。
- **最强结果**：impossibility probe在held-out families上balanced accuracy=0.97（Bonferroni P=0.018），是本文最稳健的发现。

## 相关工作脉络
- Azaria & Mitchell (2023) [11]：证明LLM内部可检测谎言，开启truth probing先河——本文扩展至"不可能性"维度。
- Marks & Tegmark (2024) [12]：发现true/false数据集的线性结构——本文指出该结构无法区分虚假与不可能。
- Hewitt & Liang (2019) [8]、Belinkov (2022) [9]：讨论探针的解释边界（可解码≠因果使用）——本文作者明确承认此局限。
- Cunningham et al. (2023) [14]、Lieberum et al. (2024) [15]、McDougall et al. (2025) [16]：SAE特征提取方法——本文沿用Gemma Scope 2并发现其重复探针几何。
- Chomsky (1957) [3]："colorless green ideas"——本文的semantic anomaly操作化基准。
- Wittgenstein (1922) [2]：《逻辑哲学论》中sinnlos（无意义）与unsinnig（荒谬）区分——本文结果与之相关但不完全对应。

## 局限性与未来方向
- 样本量有限：仅15个主题族、单一模型（Gemma 3 4B IT），结论外推需谨慎。
- 探针仅证明可解码性，无法证明模型在推理时主动使用该方向（因果关系未验证）。
- 刺激均为英语单模板句子，缺乏多语言、多模态、自然语境的检验。
- "impossibility"操作化定义混合了矛盾、定义违反、循环关系等多种类型，可能隐藏异质性。
- 未系统研究模型规模效应——更大模型是否仍保持正交性或合并方向？
- 未测试因果干预（如引导模型沿某方向移动）对输出的影响。

## 研究启发与可借鉴点
1. **五条件对照设计可迁移**：真实/虚假/不可能/异常/概率可能这一分类框架可用于检验其他模型的概念表征能力。
2. **方向正交性检验可作为新指标**：不仅报告单一探针性能，还可测量不同概念方向的关系，揭示模型内部语义结构。
3. **探针+SAE双验证策略**：线性探针提供方向，SAE提供稀疏特征解释，两者交叉印证增强结论可信度。
4. **跨数据集迁移评估**：在构造集训练、哲学集测试（或反之）可检验概念的泛化性，避免过拟合刺激形式。
5. **"模型分类 vs 人类分类"对比视角**：若模型自身组织方式与人类不同，这种差异本身即是有价值发现，而非仅视为缺陷。

## 关键术语表
- **Contingent falsehood（偶发虚假）**：在特定事实下为假，但在其他可能世界中可为真的命题（如"巴黎是德国首都"）。
- **Necessary falsehood（必要虚假/不可能命题）**：在所有可能世界中均为假的命题（如"已婚的单身汉"、"比自己更高的山"）。
- **Semantic anomaly（语义异常）**：语法正确但语义组合违反选择限制的表述（如Chomsky的"colorless green ideas"）。
- **Linear probe（线性探针）**：在模型隐藏状态上训练的线性分类器，用于测试某标签是否可从表征中解码。
- **Double dissociation（双重解离）**：两个方向分别对两个对比敏感但不互相覆盖的现象，证明它们代表独立的信息维度。
- **Sparse autoencoder（稀疏自编码器）**：学习将高维激活分解为少量活跃特征的无监督方法，用于提取可解释的语义单元。
- **Balanced accuracy（平衡准确率）**：各类别召回率的平均值，消除类别不平衡带来的偏差。
- **Residual stream（残差流）**：Transformer中各层输出的累加残差连接，承载逐步累积的语义信息。

## 可复现要素
- 数据集：公开，提供philosophical和modality两套JSON文件于 https://github.com/sixticket/representing-the-impossible
- 代码：公开，含提取、探针、SAE编码及绘图全部脚本，同上URL
- 权重：Gemma 3 4B IT与Gemma Scope 2 SAE权重从Hugging Face获取
- 关键超参：L2正则化C=0.1；类别平衡权重；1,999次置换；Bonferroni校正35层；TF-IDF n-gram范围1-2（词）/3-5（字符）
- 环境：Python 3.11, PyTorch 2.13.0, Transformers 5.14.1, scikit-learn 1.9.0, SAELens 6.47.1
