---
title: "Are-You-Sure-You-re-Sure-On-the-Impact-of-Instruction-Tuning"
source: https://arxiv.org/pdf/2608.13430v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:37:08"
field: "大语言模型可靠性与不确定性"
keywords: ["instruction tuning", "model calibration", "lexical diversity", "uncertainty estimation", "rationale generation"]
innovations: ["首次配对评估 Base/Instruct 模型在答案不确定性与理由词汇多样性上的联合变化", "提出控制答案选择与生成长度后的多样性对比分析方法", "揭示微调后自信度提升与多样性变化之间存在非均匀解耦现象"]
benchmarks: ["ARC-Easy", "MMLU", "CommonsenseQA"]
---

# 论文速读：Are-You-Sure-You-re-Sure-On-the-Impact-of-Instruction-Tuning

## 一句话总结
本文通过配对评估三组基础与指令微调模型，系统研究了指令微调对多选项问答任务中模型自信度及生成推理依据（rationale）词汇多样性的影响；发现指令微调一致性地提升了模型自信度但未同步改善预测准确率，且词汇多样性的变化呈现高度异质性，与置信度及校准误差并无线性对应关系。

## 研究问题与动机
- **核心问题**：指令微调在提升模型服从指令能力与部分任务表现的同时，是否系统性地改变了模型对自己答案的置信表达，以及这些改变是否伴随生成解释文本（rationale）在词汇层面的多样性变化？
- **现有方法不足**：
    1. 先前研究多关注单一维度的“模型自信度”代理（如基于似然的校准、重采样一致性或言语化概率），缺乏将“答案选择的不确定性”与“支持性理由的词汇多样性”联合考察的系统分析。
    2. 即便有工作指出指令微调后模型易出现“口头过度自信”，也鲜少探讨这种自信变化是否与生成文本的表达丰富度（lexical diversity）存在耦合关系。
    3. 直接比较 Base/Instruct 模型的多样性容易受到“答案切换”和“生成长度差异”的混杂影响，缺少针对相同答案与对齐长度条件的控制实验。
    4. 校准误差（ECE）在基于似然与言语化概率两种估计下常常分化，如何理解其与生成文本多样性变化的关系仍不清楚。

## 核心贡献（创新点）
1. **首次系统配对评估 Base/Instruct 模型的“答案选择不确定性 × 理由词汇多样性”**：在三组主流模型（Qwen2.5-7B、Llama-3.1-8B、Mistral-7B-v0.3）与三个公开基准（ARC-Easy、MMLU、CSQA）上进行对照实验，揭示了自信度与多样性在微调后的解耦现象。
    - **本质区别**：不同于以往仅报告单一置信度指标的工作，本文同时引入 choice entropy 与 verbalized confidence，并将独特 bigram 比例（Unique-2）与 Self-BLEU 交叉验证，形成多维度联合画像。
2. **引入控制变量分析以隔离“答案切换”和“生成长度”的影响**：限定 Base 与 Instruct 选择同一答案，并在 Token 长度上配对截断后再测量多样性变化，确认观察到的多样性差异并非由答案偏移或文本长短差异导致。
    - **本质区别**：与多数直接报告平均指标的研究相比，本文证明即使在同一答案路径上，Instruct 仍系统性降低 cross-rationale 变异性（1-SelfBLEU 下降），同时 Unique-2 的方向因模型/数据集而异。
3. **揭示信心提升并非以多样性下降为代价的统一规律**：数据显示 instruction tuning 后 choice entropy 普遍下降、言语化置信度显著上升，但 lexical diversity 的变化方向和幅度并不一致，且与 ECE 变化也无稳定对应。
    - **本质区别**：反驳了“更自信的模型必然产生更同质化表述”的直觉假设，强调需要多维度指标共同解读微调带来的可靠性和生成风格变化。

## 方法详解
- **模型与任务设定**：选取三对匹配的 Base 与 Instruct 模型（Qwen2.5-7B、Llama-3.1-8B、Mistral-7B-v0.3），在 ARC-Easy（小学科学）、MMLU（多学科知识）、CSQA（常识推理）三个多选项 QA 基准上进行测试。
- **答案选择与置信度度量**：
    - 预测 $\hat{y} = \arg\max_{y \in Y} p_{LM}(y|x)$。
    - **Choice Entropy**：$H_{choice}(x) = -\frac{\sum_{j=1}^{M} p_j \log p_j}{\log M}$，归一化后越高代表对候选答案越不确定。
    - **Verbalized Confidence**：采用两阶段提示协议，第一步用上述最大似然确定答案，第二步让模型输出该答案正确的数值概率（约束为 0–1）。
- **词汇多样性度量**：对每道题采样 $K=5$ 条 CoT rationale（temperature=0.7，nucleus p=1.0，最多 100 tokens），计算：
    - **Unique-2**：distinct bigram 比例，越大说明词汇集合越丰富。
    - **1-SelfBLEU**：各 rationale 两两之间的 Self-BLEU 取反，越大表示跨生成样本间的表面形式越多样。
- **控制实验**：筛选 Base 与 Instruct 选出相同答案的样例，按 token 数配对并截断到相同长度后重新计算 Unique-2 与 1-SelfBLEU，从而排除答案切换与长度差异带来的混杂。
- **校准误差**：按 ECE 公式对不同置信度 bin 内的准确率与平均置信度偏差求加权平均，分别基于 likelihood-based 与 verbalized confidence 计算。

## 实验与结果
- **数据集与基线**：ARC-Easy、MMLU、CSQA；基线为各模型对应的 Base 版本。
- **主要结果（Table 1 摘要）**：
    - **置信度提升一致**：所有模型在所有基准上的 $H_{choice}$ 均下降，且 verbalized confidence 显著上升。例如 Llama 在 ARC-Easy 上从 49.2% 升至 90.4%；Mistral 在 CSQA 上从 47.9% 升至 92.2%。
    - **准确率变化不一致**：ARC-Easy 上 Llama 准确率保持不变（82.2%），但自信度显著提升；Mistral 在 MMLU 上仅微增 0.1pp（59.6→59.7）。
    - **词汇多样性变化异质**：1-SelfBLEU 在所有模型/基准上均下降（cross-rationale 相似度上升），其中 Mistral 在 ARC-Easy 下降最多（0.813→0.626）。Unique-2 则出现分叉：Mistral 在 CSQA 上升（0.719→0.750），而其它组合下降或不变。
- **控制实验结果（Table 3）**：在相同答案、匹配长度条件下，Qwen 的 Unique-2 基本不变（−0.001），Mistral/Llama 的 Unique-2 显著上升（+0.050、+0.053），但 1-SelfBLEU 均下降（−0.036、−0.069、−0.012）。
- **校准与多样性关联**：ECE 变化与多样性变化无稳定方向一致性（Appendix C 详细数据），说明两类指标捕获的是不同的微调效应。

## 相关工作脉络
1. **校准与置信度代理研究**（Jiang 2021; Kadavath 2022; Tian 2023; Xiong 2024）：关注如何度量 LLM 的置信表达；本文在此基础上增加了 lexical diversity 维度并进行配对消融。
2. **指令微调对置信分布的影响**（Kadavath 2022; Zhu 2023; Zhang 2024）：指出微调常导致过自信；本文进一步验证该现象在不同模型族中的一致性，并引入多样性视角。
3. **后训练对语言多样性的影响**（Guo 2025; Yun 2025; Deshpande 2025）：聚焦生成文本的多样性保持；本文与之不同，将多样性与答案选择的不确定性联合建模，而非仅比较生成风格。
4. **基于采样的不确定性估计**（Xiong 2024）：使用重采样一致性评估置信；本文与其互补，不仅看答案层面，还看 rationale 表面的词汇分散程度。
5. **创意与多样性评估**（Tian 2024; Park 2025a,b）：多从人类判断或语义角度评估创意；本文则使用 Unique-2 与 Self-BLEU 等浅层词汇指标，并直接关联到置信度变化。

## 局限性与未来方向
- **局限**：仅评估英文、多选项、选择题型；仅考察表层词汇多样性（bigram 与 n-gram 重叠），未覆盖句法与语义层面的多样性变化。
- **未来方向**：扩展至语义多样性与句法多样性；在更多语言与非选择题生成任务中验证；探讨降低后训练过度自信是否会以进一步压制多样性为代价；研究校准方法与解码干预对 cross-rationale 多样性的影响。

## 研究启发与可借鉴点
1. **联合建模“答案不确定性 × 生成文本多样性”具有揭示隐蔽副作用的价值**：后续研究可在评测套件中同时报告 choice entropy 与 lexical/self-similarity 指标，避免仅凭单一置信度代理判断模型可靠性。
2. **控制实验设计值得复用**：以“相同答案 + 匹配长度”为核心的配对分析，能够有效剥离答案切换与文本长度对多样性指标的干扰，建议在类似对比实验中被广泛采用。
3. **提醒应用端慎用言语化置信度**：当 Instruct 模型的 verbalized confidence 大幅提升但 ECE 并未同步改善时，应结合 likelihood-based 校准与多样性分析，防范高置信度误判导致的盲从风险。
4. **可作为优化目标的新范式**：未来可探索在训练中引入多样性保持项（如约束 Self-BLEU 不低于阈值），以平衡指令遵从与表达冗余压缩之间的矛盾。

## 关键术语表
- **Choice Entropy**：基于模型对各候选答案归一化概率计算的归一化信息熵，越高表示模型对答案选择越不确定。
- **Verbalized Confidence**：通过两步提示让模型显式输出其对选定答案正确概率的数值估计。
- **Unique-2**：生成文本中不同 bigram 占总 bigram 的比例，用于衡量词汇层面的丰富度。
- **Self-BLEU**：同一问题的多次生成结果彼此之间的 BLEU 相似度，越大说明重复生成的表面形式越趋同。
- **Expected Calibration Error (ECE)**：按置信度分箱计算的真实准确率与平均预测置信度之间偏差的加权平均，用于评估校准质量。
- **Instruction Tuning**：基于自然语言指令数据对预训练语言模型进行微调，使其更好地遵循指令并完成各类任务。

## 可复现要素
- **数据集**：ARC-Easy、MMLU、CSQA（公开）
- **模型权重**：Qwen2.5-7B、Llama-3.1-8B、Mistral-7B-v0.3 及其 Instruct 版本均公开（HuggingFace）
- **代码**：论文未明确提供开源仓库链接（仅说明使用 lm-evaluation-harness 和 SacreBLEU）
- **关键超参**：rationale 采样 $K=5$，temperature=0.7，nucleus p=1.0，最大生成 token=100；言语化置信度提示见附录 B。
