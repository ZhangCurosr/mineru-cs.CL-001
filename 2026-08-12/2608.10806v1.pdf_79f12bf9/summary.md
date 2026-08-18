---
title: "Assessing Reliability of BERT-Based Models on Question Answering Tasks"
source: https://arxiv.org/pdf/2608.10806v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:42:04"
field: "自然语言处理-模型可靠性评估"
keywords: ["Reliability Estimation", "Monte Carlo Dropout", "BERT", "Question Answering", "Input Perturbation", "SQuAD", "QuAC"]
innovations: ["提出MCD内部扰动+paraphrasing输入扰动双通道可靠性评估框架", "首次系统论证QA任务中MCD dropout率的选取（10%为最优平衡点）", "揭示准确率与可靠性解耦现象，证明RoBERTa整体最可靠而ALBERT/DistilBERT存在显著不一致"]
benchmarks: ["SQuAD 2.0", "QuAC"]
---

# 论文速读：Assessing Reliability of BERT-Based Models on Question Answering Tasks

## 一句话总结
本文系统评估了四种 BERT 变体（RoBERTa、BERT-Base、DistilBERT、ALBERT）在问答任务中的可靠性，通过 Monte Carlo Dropout 引入内部随机扰动、通过语义改写（paraphrasing）引入输入扰动，以余弦相似度和 $F_1$ 分数为指标，发现 **RoBERTa 整体可靠性最强**，而 **ALBERT 和 DistilBERT 存在显著的不一致性**；同时证明推理阶段启用 Dropout 不会显著破坏模型的预测分布。

## 研究问题与动机
1. **准确率高 ≠ 可靠**：已有研究大量报告 BERT 及其变体在 SQuAD、QuAC 等数据集上的准确率，但模型在内部随机扰动和输入措辞变化下的输出一致性几乎未被系统评估。
2. **现有可靠性评估方法的不足**：Miok 等人（2022）在分类任务中使用 MCD BERT 验证了可靠性，但 QA 任务的结构化 span 提取特性使其可靠性评估更具挑战性，现有工作缺乏针对 QA 的场景化研究。
3. **准确率与可靠性的解耦风险**：若高准确模型在微小扰动下产生不一致答案，将严重影响虚拟助手、自动检索系统等实际部署场景的可信度，而目前缺乏面向 QA 的可靠性评估基准。
4. **缺少 dropout 率的系统论证**：已有工作使用 MCD 时多为经验选取 dropout 率（如 0.15），本文首次通过 sweep（0%–35%）在两个 QA 数据集上系统论证了 10% 作为平衡点的合理性。

## 核心贡献（创新点）
1. **提出双通道可靠性评估框架**：同时从模型内部随机扰动（MCD）和输入语义扰动（paraphrasing）两个正交维度量化 BERT 变体的预测稳定性，与仅关注准确率的已有工作形成本质区别。
2. **系统论证 MCD dropout 率的选取**：通过在 SQuAD 和 QuAC 上 sweep 0%–35% dropout 率，实证发现 ≥15% 会导致信息丢失（大量 blank 回答），5% 无法引入有效随机性，从而为 QA 任务中的 MCD 可靠性评估确立了数据驱动的 10% 操作点。
3. **揭示准确率与可靠性的解耦现象**：发现某些低准确率模型（如 ALBERT 在 QuAC 上 ans 准确率仅 0.07%）在无答案类别上仍保持稳定，而高准确率模型（RoBERTa）在输入扰动下亦会产生语义偏差，挑战"准确率即可靠性"的直觉假设。
4. **建立统计验证与人工评估相结合的可靠性度量体系**：使用 Welch's t-test 和 Bonferroni 校正的 Wilcoxon 符号秩检验量化模型间差异显著性，并以 Fleiss' $\kappa$ 系数（约 0.78）验证人工标注一致性，补充自动指标可能低估语义正确性的不足。

## 方法详解
- **内部扰动（MCD）**：在推理阶段开启 Dropout（经过 sweep 选定 dropout rate = 10%），对每个输入进行 $N=50$ 次前向传播，生成 50 个随机样本；对每个样本计算 cosine similarity 和 $F_1$ score，取均值与标准差作为该输入的可靠性度量；标准差越小表示模型对内部随机性越稳定。
- **输入扰动（Paraphrasing）**：使用预训练 BART 模型（`eugenesiow/bart-paraphrase`）对原始问题进行改写，通过 Sentence-BERT（`all-MiniLM-L6-v2`）计算原问题与改写后问题的余弦相似度，**仅保留相似度落在 0.75–0.98 区间的改写样本**（此区间被实证验证为"语义保持与词汇变化平衡"的最佳范围，见表 2）；用模型处理改写后问题，比较生成答案与 ground truth 的 cosine similarity 和 $F_1$ score。
- **相似度阈值论证**：对 SQuAD 10% 随机样本的验证显示：<0.75 时语义偏离明显（avg Jaccard=0.2932）；0.75–0.98 区间平衡良好（avg cosine=0.9206, avg Jaccard=0.5642）；>0.98 时近重复（avg Jaccard=0.9387）。
- **评估指标**：Cosine Similarity（衡量预测答案与 ground truth 在语义空间的对齐程度）和 $F_1$ Score（Precision-Recall 调和均值）；Exact Match 判定时采用语义阈值 cosine similarity ≥ 0.95 而非严格字符串匹配。
- **统计检验**：Welch's t-test（比较 RoBERTa 与 DistilBERT 在 MCD 扰动下的性能差异，SQuAD 上 p<0.05，QuAC 上 p=0.4547 不显著）；Pairwise Wilcoxon signed-rank test + Bonferroni 校正（$\alpha_{corr} \approx 0.0021$）用于 paraphrasing 结果的多模型比较。
- **人工评估**：对自动判定为 incorrect 的样本进行三人独立标注，分 three classes（correct/partially correct/incorrect），用 Fleiss' $\kappa$ 衡量标注者一致性。

## 实验与结果
**数据集**：
- **SQuAD 2.0**：1204 篇 context，11873 个问题（5928 可答 / 5945 不可答），高准确率基准。
- **QuAC**：1000 篇 context，7354 个问题（5868 可答 / 1486 不可答），对话式 QA，准确率普遍较低。

**MCD 扰动下的主要结果**（Table 3-4，SQuAD total 行）：
| 模型 | Acc. (%) | MCD Acc. (%) | Cosine Sim (avg±std) | $F_1$ (avg±std) |
|---|---|---|---|---|
| **RoBERTa** | 78.57 | 78.34 | 0.8481 ± 0.1543 | 0.8098 ± 0.1772 |
| BERT-Base | 68.66 | 68.25 | 0.7605 ± 0.1992 | 0.6907 ± 0.2200 |
| **DistilBERT** | 78.21 | 78.19 | 0.7912 ± 0.1131 | 0.7584 ± 0.1285 |
| ALBERT | 67.88 | 67.55 | 0.7746 ± 0.2354 | 0.7038 ± 0.2627 |

- **最强结果**：SQuAD 上 DistilBERT 在可答问题（Ans）的 cosine similarity（0.8992）和 $F_1$（0.8496）最高，标准差最低（0.0906/0.1149）；RoBERTa 总体验收一致性最优。
- **QuAC 上 DistilBERT** 无答案类别表现突出（cosine 0.8378，$F_1$ 0.8156），但 ALBERT 在可答问题上出现极低准确率（0.07%）。
- **相关性分析**（Table 5）：未扰动与 MCD 扰动后预测的高相关性（所有模型各数据集 ≥ 0.85），证明 MCD 不破坏推理动力学。
- **Paraphrasing 扰动下的主要结果**（Table 6，SQuAD Total）：RoBERTa 全面领先（cosine 0.4895，$F_1$ 0.3812）；QuAC 上 ALBERT 整体表现最优（cosine 0.3217，$F_1$ 0.2333），RoBERTa 紧随其后（cosine 0.3206，$F_1$ 0.2135）。
- **人工评估**：RoBERTa $\kappa=0.7947$（84.11% 一致），DistilBERT $\kappa=0.7671$（82.24% 一致）；自动指标严重低估语义正确响应比例。

## 相关工作脉络
1. **Miok et al. (2022)**（本文 [17]）：首次在 BERT 分类任务中验证 MCD BERT 可靠性，本文将其框架拓展至 QA 任务，并首次系统论证 dropout 率选取，填补了 MCD 在 QA 场景中缺乏实证依据的空白。
2. **Rawat & Samant (2022)**（[25]）、Ozkurt (2024)（[20]）、Pearce & Zhan (2021)（[21]）：聚焦 BERT 变体在 SQuAD 等数据集上的**准确率**对比；本文与之定位不同——承认上述工作的精度结论，但指出其完全忽略可靠性维度，二者构成互补而非竞争关系。
3. **Van Aken et al. (2019)**（[27]）：层析分析 BERT 在 QA 中的 hidden state 表示；本文关注更高层级的输出稳定性，二者在分析粒度上不同，可结合用于"表示-输出"全链路可靠性评估。
4. **Raj et al. (2022/2023)**（[22][23]）：面向 LLM 的语义一致性可靠性评估；本文聚焦于 encoder-only 的 BERT 变体（更轻量、更适合抽取式 QA），为轻量模型可靠性评估提供了对照基准。
5. **Liu et al. (2019) RoBERTa**（[14]）、**Lan et al. (2020) ALBERT**（[10]）：原始架构改进论文，主要报告准确率提升；本文在此基础上首次系统报告这些模型在扰动下的可靠性差异，给出"选型时应同时考虑稳定性"的实践建议。

## 局限性与未来方向
1. **仅限 BERT 类表示模型**：未全面评估 decoder/generative 架构在 QA 中的可靠性（附录虽测试 Tiny Llama，但性能过低无法有效评估）。
2. **任务单一**：仅覆盖提取式 QA，未扩展至生成式 QA、阅读理解等更复杂 NLP 任务。
3. **数据集有限**：仅使用 SQuAD 2.0 和 QuAC 两个数据集，结论的外部泛化性有待更多基准验证。
4. **扰动方式单一**：内部扰动仅用 MCD，输入扰动仅用 paraphrasing，未探索其他扰动手段（如 adversarial attack、token-level noise）。
5. **未涉及 LLM 系统性评估**：附录仅测试 Tiny Llama 一个模型，未能给出 LLM vs. BERT 可靠性的全面对比。
6. **未来方向**（作者自述）：探索 Bayesian Neural Networks、Ensemble Learning 等更高级的不确定性量化方法；将可靠性评估扩展至更多 NLP 任务；针对特定领域进行 fine-tuning 以提升稳定性。

## 研究启发与可借鉴点
1. **双通道扰动设计可直接迁移**：将"内部随机性 + 输入语义扰动"双通道评估框架应用于本研究团队的其他任务（如文本分类、NER、语义匹配），可快速建立可靠性评估流水线。
2. **Dropout 率 sweep 的实验策略值得复用**：在采用 MCD 进行不确定性估算时，应先对目标数据集 sweep dropout 率（建议 0–35%），以找到既不过度破坏表示又不引入有效随机性的最优操作点，避免经验选取带来的偏差。
3. **语义相似度阈值筛选改写样本的范式**：使用 Sentence-BERT 计算原始-改写文本余弦相似度，结合 Jaccard 词重叠双重验证筛选区间（0.75–0.98），这一策略可推广至其他需要"可控语义扰动"的实验场景。
4. **准确率与可靠性的解耦提醒**：在模型选型时不应仅看准确率排行榜，还需叠加可靠性评估；对于安全关键场景（医疗、法律 QA），优先选择低 std、高相关性的模型变体。
5. **人工评估补自动指标的必要性**：自动 metric（exact match / BLEU）容易低估语义等价但措辞不同的正确回答；引入人工标注并报告 Fleiss' $\kappa$，可使评估结论更为可信。

## 关键术语表
- **Monte Carlo Dropout (MCD)**：在推理阶段保持 Dropout 开启，多次前向传播产生不同输出，用于估计模型预测的不确定性。
- **Cosine Similarity**：衡量两个向量夹角的余弦值，用于评估预测答案与 ground truth 在语义嵌入空间中的对齐程度。
- **Paraphrasing-based Input Perturbation**：通过预训练改写模型对输入问题进行同义改写，引入受控的语言形式变化以评估模型对输入措辞的敏感性。
- **Reliability（可靠性）**：模型在受控内部或输入扰动下维持预测一致性的能力，以输出指标的标准差和均值综合量化。
- **SQuAD 2.0**：斯坦福问答数据集第二版，包含可答与不可答问题，是提取式 QA 的主流 benchmark。
- **QuAC**：Question Answering in Context，对话式问答数据集，问题具有上下文依赖性，挑战更高。
- **Fleiss' Kappa ($\kappa$)**：衡量多位标注者之间分类一致性的统计量，值越接近 1 表示标注者间 agreement 越强。
- **Bonferroni Correction**：多重比较校正方法，将显著性阈值除以比较次数以控制 I 类错误膨胀。

## 可复现要素
- **数据集**：SQuAD 2.0（公开，https://rajpurkar.github.io/SQuAD-explorer/）和 QuAC（公开，https://quac.ai/）。
- **代码/权重**：实验代码、评估流程和复现材料开源在 https://github.com/uncertainity-quantification/reliability-estimation-qa1（匿名仓库）。
- **关键超参**：
  - MCD 随机采样数 $N = 50$
  - 最优 dropout rate = 10%（经 sweep 选定）
  - Paraphrasing 模型：`eugenesiow/bart-paraphrase`
  - 语义过滤阈值：cosine similarity ∈ [0.75, 0.98]
  - Exact Match 语义判定阈值：cosine similarity ≥ 0.95
  - 统计检验显著性水平：$\alpha = 0.05$，Bonferroni 校正后 $\alpha_{corr} \approx 0.0021$
- **环境**：Python，Hugging Face Transformers，Sentence-BERT（`all-MiniLM-L6-v2`）
