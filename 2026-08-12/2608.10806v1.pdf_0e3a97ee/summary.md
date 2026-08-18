---
title: "Assessing Reliability of BERT-Based Models on Question Answering Tasks"
source: https://arxiv.org/pdf/2608.10806v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:41:18"
field: "NLP模型可靠性评估"
keywords: ["Reliability Estimation", "Monte Carlo Dropout", "BERT", "Question Answering", "Robustness", "Paraphrasing", "SQuAD", "QuAC"]
innovations: ["双维度可靠性评估框架：内部随机性(MCD)与输入扰动(改写)正交评估", "首次系统对比四个BERT变体在QA任务上的准确性-可靠性解耦现象", "提供dropout率选择的实证依据(10%为平衡操作点)"]
benchmarks: ["SQuAD v2.0", "QuAC"]
---

# 论文速读：Assessing Reliability of BERT-Based Models on Question Answering Tasks

## 一句话总结
本文系统评估了BERT及其变体（RoBERTa、ALBERT、DistilBERT）在问答任务上的可靠性，通过蒙特卡洛Dropout引入内部随机性、通过改写引入输入扰动两个维度，揭示准确率与可靠性存在解耦现象，发现RoBERTa综合稳定性最优，而ALBERT和DistilBERT在特定条件下存在显著预测不一致。

## 研究问题与动机
1. **准确性≠可靠性**：现有文献大量报道BERT变体在SQuAD、QuAC等数据集上的准确率，但对其在扰动条件下的预测一致性缺乏系统评估，无法判断高准确率是否意味着高可信度。
2. **内部随机性评估空白**：Miok等（2022）证实MCD BERT可用于分类任务可靠性评估，但在QA任务上尚未有针对多个BERT变体的系统性实验和dropout率选择的实证依据。
3. **输入扰动敏感性未知**：真实应用场景中输入常伴随措辞变化，但BERT类编码器模型在面对语义保持的改写时是否会保持答案一致性尚不清楚。
4. **准确性-可靠性解耦**：高准确率模型在扰动下可能出现低可靠性，反之低准确率模型也可能保持稳定，现有评估体系未涵盖这一维度。

## 核心贡献（创新点）
1. **提出双维度可靠性评估框架**：同时从内部配置扰动（Monte Carlo Dropout）和输入扰动（语义改写）两个正交维度量化模型稳定性，而非仅依赖准确率单一指标。
2. **系统实证对比四个BERT变体**：首次在QA任务上全面比较RoBERTa、BERT-Base、DistilBERT、ALBERT在两种扰动下的可靠性差异，发现准确率排名与可靠性排名并不一致。
3. **提供dropout率选择的实证依据**：通过10%→35%的dropout率扫描，证明5%不足以引入有意义的随机性、≥15%会导致信息损失，确立10%为平衡操作点，填补了Miok等工作中缺乏实证 justification 的空白。
4. **发现准确性与可靠性的解耦现象**：以QuAC低准确率数据集为例，部分低准确率模型在无答案问题上仍保持高可靠性，证明两者并非强相关。

## 方法详解
**评估框架设计**：

1. **内部随机性评估（Algorithm 1）**：
   - 在推理阶段启用dropout（Monte Carlo Dropout），对每个输入重复采样N=50次
   - 每次采样得到预测答案后，与ground truth计算cosine similarity和F1 score
   - 汇总所有采样的指标均值与标准差，标准差越小表示内部稳定性越高
   - 同时对比MCD开启前后的准确率相关性（Table 5），验证dropout不破坏推理动态

2. **输入扰动评估（Algorithm 2）**：
   - 使用预训练BART paraphrasing模型（eugenesiow/bart-paraphrase）对问题进行改写
   - 用Sentence-BERT（all-MiniLM-L6-v2）计算改写前后问题的余弦相似度
   - 筛选相似度在[0.75, 0.98]区间的改写（经验验证此区间平衡了语义保持与词汇变异，见表2）
   - 对符合条件的改写输入运行模型，计算答案与ground truth的cosine similarity和F1 score

3. **统计检验**：
   - 采用Welch's t-test（α=0.05）检验RoBERTa与DistilBERT在内部扰动下的性能差异
   - 采用Wilcoxon signed-rank test + Bonferroni校正（α≈0.0021）进行多模型配对比较
   - 人工评估：3位标注员对10%样本进行标注，用Fleiss' Kappa度量标注者间一致性

4. **评估指标**：
   - Cosine similarity：$cos\theta = \frac{\vec{a}\cdot\vec{b}}{\|\vec{a}\|\cdot\|\vec{b}\|}$，衡量嵌入向量的语义对齐
   - $F_1$ Score：精确率和召回率的调和平均
   - 精确匹配标准：cosine similarity ≥ 0.95视为正确

## 实验与结果
**数据集**：
- SQuAD v2.0：1204篇context，11873个问题（含5945个无答案），高准确率基准
- QuAC：1000篇context，7354个问题（含1486个无答案），对话式QA，准确率较低

**核心结果**：

| 数据集 | 模型 | 准确率(%) | MCD Acc(%) | Cosine Sim(avg±std) | F1(avg±std) |
|--------|------|-----------|------------|---------------------|-------------|
| SQuAD-Total | **RoBERTa** | **78.57** | 78.34 | **0.8481±0.1543** | **0.8098±0.1772** |
| SQuAD-Ans | DistilBERT | 77.75 | 77.82 | **0.8992±0.0906** | **0.8496±0.1149** |
| QuAC-Total | RoBERTa | 21.39 | 21.50 | 0.4307±0.1761 | 0.2830±0.1761 |
| QuAC-Total | DistilBERT | 22.93 | 23.14 | 0.4265±0.1470 | **0.2881±0.1360** |

**改写输入下最佳结果**（Table 6）：
- SQuAD总得分：RoBERTa最优（cosine 0.4895, F1 0.3812）
- QuAC总得分：ALBERT最优（cosine 0.3217, F1 0.2333），RoBERTa次之（0.3206, 0.2135）

**统计显著性**：
- SQuAD上RoBERTa与DistilBERT差异显著（p<0.001）；QuAC上无显著差异（p=0.4547）
- Wilcoxon检验显示多种模型对在改写扰动下存在显著差异（Bonferroni校正后）

**MCD不影响推理动态**：
- 所有模型 unperturbed与MCD预测的cosine similarity相关性均>0.85（Table 5）
- DistilBERT相关性最高（SQuAD: 0.9464 cosine, 0.9423 F1）

**人工评估**：
- RoBERTa：正确45.79%，部分正确6.54%，错误47.66%（Kappa=0.7947）
- DistilBERT：正确39.25%，部分正确8.41%，错误52.34%（Kappa=0.7671）
- 自动指标低估了语义正确的回答

## 相关工作脉络
1. **Miok et al. (2022)**：提出MCD BERT用于分类任务可靠性评估，本文将其扩展至QA任务，并补充了dropout率扫描的实证依据。
2. **Raj et al. (2022, 2023)**：从语义一致性角度评估LLM可靠性，本文聚焦BERT类编码器模型在结构化QA中的稳定性，方法论互补。
3. **Ozkurt (2024)**：比较多种transformer在SQuAD v2上的准确率，但未涉及可靠性维度。
4. **Pearce & Zhan (2021)**：多数据集BERT变体对比研究，本文在此基础上增加扰动稳定性分析。
5. **Van Aken et al. (2019)**：逐层分析BERT隐藏状态以理解问答机制，本文关注端到端输出的稳定性而非内部表示可解释性。

## 局限性与未来方向
1. **任务与模型范围有限**：仅评估encoder-only的BERT变体，未扩展到generative LLM（附录中Tiny Llama表现差说明生成式模型在extractive QA中不适用本框架）。
2. **数据集局限**：仅使用SQuAD和QuAC两个英文数据集，跨语言和跨领域泛化性未验证。
3. **扰动方法单一**：内部扰动仅用MCD，输入扰动仅用改写，未探索其他扰动类型（如噪声注入、对抗攻击）。
4. **未来方向**：探索贝叶斯神经网络、集成学习等更先进不确定性量化方法；将框架扩展到更多NLP任务；研究领域自适应微调以提升稳定性。

## 研究启发与可借鉴点
1. **双维度可靠性评估可复用**：内部随机性（MCD）+ 输入扰动（改写）的正交设计可直接迁移至团队的其他模型评估流程，形成标准化的"鲁棒性报告"。
2. **dropout率选择需实证扫描**：不能直接沿用文献值，应根据具体模型和数据集通过10%-35%扫描确定最佳操作点，避免过高（信息损失）或过低（无随机性）的问题。
3. **准确性-可靠性解耦的启示**：在模型选型时不应仅看准确率排名，需同步评估标准差/方差指标；低准确率模型在高可靠性场景下可能仍有价值。
4. **人工评估与自动指标结合**：严格基于精确匹配的自动评估会低估语义正确的回答，建议引入人工标注（如三分法：correct/partially correct/incorrect）进行校准。
5. **相关性分析验证扰动有效性**：通过计算unperturbed与perturbed预测的correlation（Table 5），可快速判断某种扰动是否破坏了模型推理动态，作为可靠性评估的前置检查。

## 关键术语表
**Monte Carlo Dropout (MCD)**：在推理阶段保留dropout随机性，通过多次前向传播采样估计模型不确定性，本文用于内部扰动评估。

**Cosine Similarity**：衡量两个向量夹角的余弦值，范围[-1,1]，用于评估预测答案与ground truth在嵌入空间的语义对齐程度。

**Semantic Consistency**：模型在输入或配置扰动下保持答案语义一致的能力，本文用它量化可靠性。

**Extrac

tive QA**：从给定上下文中抽取答案片段的问答任务，区别于生成式QA，是本文研究的任务类型。

**Unanswerable Questions**：上下文中无足够信息回答的问题（SQuAD 2.0特有），模型应输出空答案，是可靠性评估的重要子集。

**MCD Accuracy**：开启Monte Carlo Dropout后 averaged across N次采样的准确率，用于验证扰动不破坏推理动态。

**Fleiss' Kappa**：衡量多名标注者间分类一致性的统计量，本文用于验证人工评估的可靠性（κ≈0.78表示substantial agreement）。

## 可复现要素
- **数据集**：SQuAD v2.0（公开）、QuAC（公开），链接见论文Data Availability
- **代码**：GitHub开源 https://github.com/uncertainity-quantification/reliability-estimation-qa1
- **关键超参**：
  - MCD采样次数N=50（10-100扫描后选定）
  - Dropout rate=10%（经验验证为平衡点）
  - 改写相似度筛选区间：[0.75, 0.98]（基于Jaccard和cosine联合验证）
  - 精确匹配阈值：cosine similarity ≥ 0.95
- **模型权重**：HuggingFace标准预训练权重（RoBERTa, BERT-Base, DistilBERT, ALBERT）
- **改写模型**：eugenesiow/bart-paraphrase
- **嵌入模型**：all-MiniLM-L6-v2 (Sentence-BERT)
