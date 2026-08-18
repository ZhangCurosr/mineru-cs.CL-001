---
title: "Assessing Reliability of BERT-Based Models on Question Answering Tasks"
source: https://arxiv.org/pdf/2608.10806v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:42:04"
field: "NLP模型可靠性与稳定性评估"
keywords: ["Reliability Estimation", "Monte Carlo Dropout", "BERT Models", "Question Answering", "Paraphrase Perturbation", "SQuAD", "QuAC"]
innovations: ["提出同时从内部MCD随机性与输入改写语义扰动两条路径评估BERT类QA模型可靠性", "为MC Dropout推理阶段提供数据驱动的10%dropout率选择依据并验证其对推理无破坏性", "结合语义相似指标、Wilcoxon/Bonferroni统计检验与人工评估给出模型可靠性的综合判别体系"]
benchmarks: ["SQuAD 2.0", "QuAC"]
---

# 论文速读：Assessing Reliability of BERT-Based Models on Question Answering Tasks

## 一句话总结
本研究系统评估了 BERT、RoBERTa、ALBERT 与 DistilBERT 四种编码器模型在 QA 任务中的**可靠性**，通过内部随机扰动（Monte Carlo Dropout）与输入语义扰动（机器改写/Paraphrase）双重条件，证明**高准确率不等于高稳定性**，且 RoBERTa 与 DistilBERT 在跨数据集均表现出更强的预测一致性。

## 研究问题与动机
1. 现有工作对 Transformer QA 模型的评测以准确率等静态指标为主，缺乏对模型**在受控扰动下输出稳定性**的系统分析；  
2. 实际部署场景中，输入表述变化与模型内部推断随机性普遍存在，但研究者不清楚这些扰动是否会导致语义偏离；  
3. 不同 BERT 变体在参数规模、训练目标与结构优化上差异显著，其可靠性排序仍未明确；  
4. 准确率与可靠性之间的关系尚未被实证检验，可能存在“高准低稳”或“低准高稳”的非单调关系。

## 核心贡献（创新点）
1. 提出同时从**内部配置扰动**与**输入语义扰动**两条路径量化 BERT 类模型在 QA 上的可靠性。  
2. 系统性地为 MC Dropout 推断阶段的 dropout 率选择提供了**数据驱动的阈值依据**（确定为 10%），并验证其对推理过程无破坏性。  
3. 建立包含 cosine similarity 与 F1 的多指标评估框架，结合 **Welch’s t-test** 与 **Bonferroni 校正的 Wilcoxon 配对检验**，给出模型差异的统计显著性结论。  
4. 引入基于 Sentence-BERT 的**人工语义对齐验证**与多标注者一致度分析，指出传统精确匹配指标会低估部分语义正确的输出。  
5. 对比 SQuAD（高准确率）与 QuAC（低准确率/对话式）两个代表性数据集，揭示模型可靠性具有**任务/数据集依赖性**。

## 方法详解
1. 模型与数据集：选用 BERT-Base、RoBERTa、DistilBERT、ALBERT 四种 BERT 系列预训练模型，在 SQuAD 2.0 与 QuAC 两个公开 QA 数据集上进行评测。  
2. 评估指标：同时采用  
   - **Cosine Similarity**（基于 S-BERT `all-MiniLM-L6-v2` 计算预测答案与标准答案的语义相似度）；  
   - **F1 Score**（词汇层精确匹配）；  
   用于衡量语义一致性与字面一致性。  
3. 内部扰动评估（MC Dropout）：在推理阶段开启 Dropout，对每个样本独立运行 N=50 次前向，得到多个语义/词汇输出，并统计各指标的平均值与标准差；通过对比启用 MCD 前后的 Accuracy 及预测相关性（correlation coefficient）判断对推理动态的影响。  
4. 输入扰动评估（Paraphrase）：使用 HuggingFace 中基于 BART 的改写模型生成原问题的改写版本，再利用 Sentence-BERT 计算改写前后句子的 cosine similarity，保留落在 **[0.75, 0.98]** 区间的改写样本以避免语义漂移或过于雷同；对改写输入进行 QA 推理后与原答案计算语义与词汇相似度。  
5. 统计验证：使用 Welch’s t-test 检验 RoBERTa 与 DistilBERT 的分布差异；使用 Pairwise Wilcoxon signed-rank test 进行所有模型组合在两数据集两指标上的差异比较，并通过 Bonferroni 校正控制多重比较误差。  
6. 人工评估：采用分层抽样对 10% 样本进行三重分类（正确/部分正确/错误），使用 Fleiss’ Kappa 度量标注一致性，并与自动指标结果对比。

## 实验与结果
- **SQuAD 上 MC Dropout 结果**：  
  - RoBERTa 总体 cosine 0.8481 ± 0.1543，F1 0.8098 ± 0.1772；  
  - DistilBERT 在可答题目上表现最优：cosine 0.8992 ± 0.0906，F1 0.8496 ± 0.1149；  
  - BERT-Base 与 ALBERT 的稳定性与平均相似度相对较差，ALBERT 在可答题目上一致性波动最大。  
- **QuAC 上 MC Dropout 结果**：  
  - RoBERTa 总体 cosine 0.4307 ± 0.1761，F1 0.2830 ± 0.1761；  
  - DistilBERT 总体 cosine 0.4265 ± 0.1470，F1 0.2881 ± 0.1360；  
  - ALBERT 在无答案题目上 cosine 达 0.8309，但标准差 0.2820，稳定性显著弱于 DistilBERT 的 0.1775。  
- **统计显著性**：SQuAD 上 RoBERTa 与 DistilBERT 性能差异显著（p<0.05）；QuAC 上两者差异不显著（p=0.4547）。  
- **MCD 相关性**：四种模型未扰动与扰动后预测分数相关系数普遍 >0.85，说明 MCD 不会破坏推理表征。  
- **输入改写评估**：  
  - SQuAD 上 RoBERTa 总体最佳（cosine 0.4895，F1 0.3812）；  
  - QuAC 上 ALBERT 在可答题目上优于其他模型（cosine 0.3311，F1 0.2825）。  
- **人工评估**：自动指标判定的错误样本中，RoBERTa 有 52.33% 被人类判为正确或部分正确，DistilBERT 为 47.66%，说明纯词汇指标会低估语义一致性。

## 相关工作脉络
1. **Miok et al. (2022)**：使用 MC Dropout 评估 BERT 在分类任务中的可靠性，本文将其扩展至 QA 任务，并给出 dropout 率的系统化确定方法。  
2. **Ozkurt (2024) / Pearce & Zhan (2021)**：在 SQuAD 等基准上对比 BERT 变体的准确率表现；本文补充了准确率之外的稳定性维度。  
3. **Raj et al. (2022/2023)**：面向 LLM 提出语义一致性可靠性度量思路，本文针对编码器型 QA 模型进行对照验证。  
4. **Miok et al. (2019)**：早期在 LSTM 上验证 MC Dropout 可靠性评估可行性，为本文方法迁移提供先例。  
5. **Huang et al. (2025)** 等关于 LLM hallucination 的研究：提醒依赖大模型做事实性 QA 存在一致性风险，强化了精确评估小参数量模型可靠性的必要性。

## 局限性与未来方向
1. 仅在 SQuAD 与 QuAC 两个数据集上验证，样本域与语言复杂度有限；  
2. 内部扰动只采用 MC Dropout，未覆盖 dropout 以外的随机性来源；  
3. 输入扰动仅使用一种 BART 改写器，未测试其他生成/噪声扰动策略；  
4. 未将 LLM 类模型纳入系统性对比（附录 Tiny Llama 实验因基线准确性过低而无法有效评估）；  
5. 可靠性结论主要基于定量指标，仍需更大规模人工验证。  
未来方向包括引入贝叶斯神经网络、集成学习等不确定性建模方法，并将框架拓展到复杂对话型或多模态 QA 场景。

## 研究启发与可借鉴点
1. **双路径可靠性评估范式**：同时考察“模型内随机性”与“输入语义扰动”，可为其他 NLP 下游任务（抽取、分类、生成）提供统一评估模板。  
2. **MCD 推理可用性实证**：为 10% 左右的 dropout 率在 QA 任务上既能产生可区分方差又不破坏语义表示提供了可直接复用的经验值。  
3. **引入 Sentence-BERT 作为语义等价判定工具**：在自动评测中缓解 strict exact match 偏严格问题，并建议结合人工语义标注作联合评估。  
4. **Wilcoxon + Bonferroni 的模型对比流程**：在多模型、多指标、多数据集情况下可用于保障统计结论可信度。  
5. **准确率与可靠性解耦观察**：提示团队在项目选型中应避免“唯准确率论”，在关键业务场景中补充稳定性评估以控制线上风险。

## 关键术语表
**Monte Carlo Dropout (MCD)**：在推理阶段保留 Dropout 机制，通过对同一输入多次采样评估模型输出分布稳定性。  
**Cosine Similarity (S-BERT)**：利用 Sentence-BERT 提取答案向量并计算夹角余弦，衡量预测与标准答案的语义相似度。  
**Paraphrase-based Perturbation**：使用预训练改写模型对输入问题进行同义改写，并在语义相似度阈值内筛选扰动样本以检验模型对表层表述变化的鲁棒性。  
**Answerable / Unanswerable**：SQuAD 2.0 与 QuAC 中可根据上下文找到答案或无法找到答案的两类样本划分。  
**Fleiss’ Kappa**：用于衡量多名标注者在分类任务上的一致程度，取值越高说明人工标注越可靠。  
**Bonferroni Correction**：针对多重假设检验对显著性阈值进行调整，降低第一类错误风险的方法。  
**Exact Match (EM)**：预测答案与标准答案字符串完全一致的评估指标。

## 可复现要素
- 数据集：SQuAD v2.0 与 QuAC，均为公开数据集（论文提供访问链接）。  
- 代码与实验框架：已开源，地址为 https://github.com/uncertainity-quantification/reliability-estimation-qa1。  
- 关键超参：MC Dropout 样本数 N=50；Dropout 率选取 10%；改写句相似度阈值区间 [0.75, 0.98]；显著性水平 α=0.05，Bonferroni 校正后约 0.0021。  
- 模型权重：使用 HuggingFace 中 BERT-Base、RoBERTa、DistilBERT、ALBERT 及 BART 改写模型、Sentence-BERT `all-MiniLM-L6-v2`。
