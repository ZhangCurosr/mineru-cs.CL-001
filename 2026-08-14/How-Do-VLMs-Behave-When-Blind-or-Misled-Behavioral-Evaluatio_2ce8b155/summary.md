---
title: "How-Do-VLMs-Behave-When-Blind-or-Misled-Behavioral-Evaluatio"
source: https://arxiv.org/pdf/2608.13267v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:00:41"
field: "多模态大模型评估"
keywords: ["VLM", "scientific figure understanding", "behavioral evaluation", "hallucination", "uncertainty acknowledgment", "A-R-I framework", "benchmark"]
innovations: ["提出SCIFIGBENCH基准，首次在三维度（感知/推理/行为可靠性）系统评估VLM科学图表理解", "设计A-R-I框架将行为可靠性分解为承认/抵抗/推断三个正交维度", "构建五类可控探针族（Inexist/Contra/Unanswerable/Caption-bias/Selective-blur）实现压力测试"]
benchmarks: ["SCIFIGBENCH", "ChartQA", "CharXiv", "SciFIBench", "ChartMuseum", "HallusionBench"]
---

# 论文速读：How-Do-VLMs-Behave-When-Blind-or-Misled-Behavioral-Evaluatio

## 一句话总结
本文提出了 **SCIFIGBENCH**，一个面向科学图表理解的行为诊断型VLM基准，首次系统评估模型在视觉证据缺失或被误导时的**行为可靠性**；引入 **A-R-I（Admittance–Resistance–Inductance）框架**，揭示高质量感知/推理评分与行为可靠性之间存在显著分离——GPT-5.2与Gemini 3.1 Pro MQM差距仅1.4分，但面对不可读内容时前者96%编造答案、后者仅29%。

## 研究问题与动机
1. **现有VLM基准缺少行为可靠性维度**：ChartQA、CharXiv、SciFIBench等主流科学图表基准仅关注感知准确率与推理能力，不评估模型在视觉证据不完整或被误导时的行为表现。
2. **高分≠可靠，部署风险被掩盖**：GPT-5.2在描述质量（MQM 91.6）和推理准确率（78.4%）上领先，但在选择性模糊不可恢复元素时，96%的情况仍自信编造答案，对科学工作流的信任构成严重威胁。
3. **科学图表错误后果严重**：图表包含论文核心定量证据，下游摘要、比较和科学推理均依赖其准确性，模型"自信地幻觉"比"承认不确定性"更具破坏性。
4. **不同模型面对不确定性行为差异极大**：Gemini 3.1 Pro在MQM仅差1.4分的情况下，承认不确定率高达71%，抵抗分数0.91，而GPT-5.2主动承认率仅8%，揭示行为维度是独立于感知/推理质量的新能力轴。

## 核心贡献（创新点）
1. **提出SCIFIGBENCH基准**：250张arXiv科学图表，600+小时人工标注，扩展至34,000+评估实例，首次在三维度（感知、推理、行为可靠性）上系统化评测VLM的科学图表理解能力。
2. **设计A-R-I行为框架**：将行为可靠性分解为三个正交维度——承认（Admittance，面对不可恢复信息时是否诚实承认不确定性）、抵抗（Resistance，是否拒绝虚假前提/误导性上下文）、推断（Inductance，能否从部分可见证据中正确推断），区别于以往仅测幻觉率或校准度的工作。
3. **构建五类可控探针族**：Inexist（预设不存在元素）、Contra（嵌入错误数值）、Unanswerable（超出图表范围）、Caption-bias（篡改说明文本含2-3个虚假声明）、Selective-blur（选择性模糊图表元素），形成可复现的压力测试套件。
4. **发现质量-行为分离现象**：证明高感知/推理准确率≠行为可靠性，同一梯队模型（GPT-5.2 vs Gemini 3.1 Pro）行为模式截然相反，挑战了"用准确率代理行为可靠性"的常见做法。
5. **建立MQM适配的科学图表描述评估流水线**：基于清单化适配的MQM框架（Accuracy/Completeness/Clarity三维度惩罚机制），LLM判官与人类标注Krippendorff's α=0.91，验证了自动化评估的可靠性。

## 方法详解
**数据集构建**：
- 来源：187篇arXiv论文（2023–2025）中的250张英文图表，包括条形图（99）、折线图（99）、饼图（52）
- 标注：每位图表由两位受训 annotator 独立撰写 expert description（94%一致率），第三方仲裁分歧；总耗时600+小时（描述240h、推理问题修订200h、模糊目标确认100h、MQM验证90h）
- 推理问题：每图4道（计数/计算/比较/模式分析），共1,000道，由GPT-4o生成后经Mistral Large 3自动校验+人工审核

**感知评估（MQM框架）**：
- 针对每种图表类型手编检查清单（条形图14项、折线图15项、饼图11项），GPT-4o判官评估覆盖度（complete/partial/missing）和正确性（correct/partial/wrong/N/A）
- 惩罚公式：`MQM = max(0, 100 − P × 100 / (N × 5))`，其中P为加权总惩罚（Major=5.0/Minor=2.0），N为检查清单项数；错误类型分为Accuracy（含数值错误、趋势误判、标签绑定错误等）、Completeness、Clarity
- 颜色分类采用11种基础色族（red/orange/yellow/green/blue/purple/pink/brown/gray/black/white），同族不判错

**A-R-I探针设计与评估**：
- **Admittance（承认）**：选择性模糊不可恢复的文字元素（经OCR提取→GPT-4o筛选→三层模糊匹配→手动确认），模糊操作：先blend factor=0.7灰化再kernel=75高斯模糊，确保完全不可读；评估模型是否诚实承认视觉局限
- **Resistance（抵抗）**：三类探针——Inexist（用定冠词预设非存在元素，利用联想先验）、Contra（嵌入偏离真实值20-30%的错误数值作为锚点）、Unanswerable（提出图表外但看似合理的领域分析问题）；评分1.0/0.5/0.0三档；标题偏差抗性：70%正确+30%掺入2-3个虚假声明，用随机A/B设计判断模型跟随图像还是 captions
- **Inductance（推断）**：选择性模糊可通过周围上下文推断恢复的元素（如图例颜色对应、坐标轴规律），评估模型在信息可恢复时是否能正确推断而非编造
- 两种评估模式：**主动**（直接提问）vs **被动**（开放式描述），揭示"必须回答"偏差

**压力测试变换**：
- 感知变换：噪声（σ=25）、低对比度（α=0.3）、旋转（15°）、页内上下文嵌入
- 所有模型temperature=0，GPT-4o作为统一判官

## 实验与结果
**评测模型（8个）**：GPT-5.2、Gemini 3.1 Pro、Phi-4 Multimodal、Llama 4 Maverick、Qwen3-VL-235B/30B/8B、Gemma-3-27B-IT

**感知（MQM）**：
- GPT-5.2：91.6（CI [90.4, 92.8]）排名第一，Gemini 3.1 Pro：90.2（CI [88.9, 91.4]）第二，差距仅1.4分（p<0.01，Cliff's δ=0.09， practically small）
- 中等梯队：Llama 4（81.4）、Qwen-235B（80.8）、Qwen-8B（78.9）；尾部：Phi-4（62.2），完整性惩罚近30分
- **旋转是最破坏性变换**，平均下降19.4 MQM；噪声影响可忽略；低对比度降4-7分；页内嵌入仅降2-5分
- 选择性模糊：不可恢复元素（AdmB）使MQM下降8-10分，可推断元素（IndB）仅降1-3分；GPT-5.2在IndB下88.9、AdmB下81.7

**推理（Capability %）**：
- Gemini 3.1 Pro：81.0%（计数89.2、计算79.4、比较89.6、模式70.0）；GPT-5.2：78.4%（计数76.1、计算82.8、比较77.9、模式72.0）
- Gemini在计数/比较上占优，GPT-5.2在计算/模式分析上略优
- Phi-4 Multimodal仅8.6%，Gemma-3-27B-IT仅27.2%；其余模型均<58%

**行为（A-R-I核心发现）**：
- **Admittance（主动）**：Gemini 3.1 Pro **71%**，Llama 4仅19%，GPT-5.2仅**8%**——"自信编造者"（confident fabricator）；被动模式下GPT-5.2为23%，Gemini为59%
- **Resistance**：Gemini **0.91**（CI [0.89, 0.93]），GPT-5.2为0.81（p<0.001）；Phi-4仅0.21
- 探针难度梯度：Unanswerable（最容易抵抗，>0.92）< Contra < Inexist（最难抵抗，GPT-5.2仅0.77）
- **Caption bias**：Gemini和GPT-5.2均为0.89，Phi-4仅0.05（95%跟随篡改caption）；Qwen系列非单调（参数规模≠caption独立性）
- **Inductance（被动）**：GPT-5.2最高77%，Gemini 73%；说明模型在信息可恢复时能正确推断

**关键结论**：MQM排名与行为可靠性排名在关键场景下完全反转；仅报告MQM的基准会遗漏行为差异；"presupposition embedding"是最强欺骗向量。

## 相关工作脉络
1. **ChartQA (Masry et al., 2022) / CharXiv (Wang et al., 2024) / SciFIBench (Roberts et al., 2024)**：聚焦图表问答准确率，无行为/不确定性评估；SCIFIGBENCH在其基础上引入A-R-I行为框架和选择性模糊探针。
2. **HallusionBench (Guan et al., 2024) / ChartHal (Wang et al., 2025)**：评估VLM幻觉但主要针对通用图像或结构化事实错误；本文聚焦**科学图表**场景中"自信编造不可见内容"的系统性偏差，并区分承认/抵抗/推断三种行为模式。
3. **CHAOS (Moured et al., 2025) / CHART-NOISe (Mahbub et al., 2025)**：研究图表在扰动/噪声下的退化；本文额外引入**误导性上下文**（caption-bias、false-premise）而非仅评估鲁棒性下降幅度。
4. **BeHonest (Chern et al., 2024) / SycEval (Fanous et al., 2025)**：评估LLM诚实性和阿谀行为；本文将其延伸至**多模态图表理解**场景，并细分为视觉证据缺失（Admittance）vs 上下文误导（Resistance）两类行为。
5. **ChartMuseum (Tang et al., 2025) / EncQA (Mukherjee et al., 2025)**：扩展图表推理深度或编码通道维度；本文与之正交，提供**行为可靠性**这一新增评测维度。
6. **Loftus (1975) 目击者记忆研究**：本文引用其"定冠词诱导虚假记忆"发现，论证Inexist探针的认知心理学基础，将社会心理学洞见引入VLM评估设计。

## 局限性与未来方向
1. **图表类型局限**：仅覆盖条形图、折线图、饼图，散点图、热图、网络图、示意图等科学常用类型未纳入。
2. **语言局限**：仅英语图表，跨语言泛化未评估。
3. **归因不精确**：无法区分失败是源于指令遵循压力（RLHF训练导致的"必须回答"偏差）还是视觉编码器能力不足，需结合prompt干预和内部探测实验。
4. **自动评估依赖**：评测完全基于GPT-4o判官，虽有人工验证（α=0.91）和跨判官鲁棒性检验（Mistral Large 3），但仍存在LLM-judge系统性偏差（如低估严重性、高估完整性）。
5. **行为维度非 exhaustive**：A-R-I未涵盖置信度校准（confidence calibration）和多轮视觉对话一致性（multi-turn consistency）等其他行为轴。
6. **未来方向**：扩展到更多图表类型和非英语语料；对开源模型进行内部探测以分离vision encoder vs LLM组件的影响；设计针对"admit-then-infer"协同行为的评估。

## 研究启发与可借鉴点
1. **A-R-I框架的可迁移性**：将行为可靠性分解为正交三维度（承认/抵抗/推断）的思路可迁移至医疗影像、自动驾驶等其他高风险多模态场景，建议探索将其推广为通用VLM行为评估范式。
2. **探针设计模式值得复用**：Inexist（预设不存在元素）利用定冠词诱发虚假记忆的策略，以及Caption-bias（70/30规则保留大部分正确内容）均是对抗测试设计的优秀模板，可应用于其它领域的鲁棒性评测。
3. **主动vs被动双模式评估设计**：揭示GPT-5.2在直接提问时承认率（8%）远低于开放式描述时（23%），验证了RLHF训练下的"必须回答"偏差；该双模式设计可用于诊断模型在对话系统 vs 离线分析场景下的不同风险。
4. **MQM清单化适配评估流水线**：将机器翻译领域的MQM框架适配至科学图表描述评估，结合人工检查清单+LLM判官+规则引擎的方案，为多模态开放生成任务的质量评估提供了可复用的工程模板。
5. **与团队方向结合机会**：若团队关注科研助手或多模态RAG系统，可将Selective-blur探针（模拟PDF渲染退化/OCR失败场景）和Caption-bias探针（模拟文献检索注入错误摘要场景）集成到现有评测流水线，检测系统级幻觉传播风险。

## 关键术语表
**SCIFIGBENCH**：面向科学图表理解的诊断性VLM基准，包含250张arXiv图表和34,000+评估实例，覆盖感知、推理和行为可靠性三维度。

**A-R-I 框架**：Admittance–Resistance–Inductance行为评估框架，将模型在不确定情境下的行为分解为承认不确定性、抵抗误导上下文、从部分证据推断三个正交维度。

**MQM（Multidimensional Quality Metrics）**：源自机器翻译质量评估的框架，本文适配为科学图表描述的三重惩罚体系（Accuracy/Completeness/Clarity），满分100分。

**Selective-blur probe**：通过OCR定位+高斯模糊技术遮挡图表特定元素，区分不可恢复（Admittance）与上下文可推断（Inductance）两种条件。

**Inexist probe**：在问题中预设非存在图表元素（使用定冠词），测试模型是否抗拒隐含虚假前提的诱导性提问。

**Caption-bias probe**：在图表说明中嵌入2-3个看似合理但与图像矛盾的错误声明，测试模型是否盲从文本上下文而非视觉证据。

**Confident fabricator**：指GPT-5.2等模型的行为特征——在感知/推理质量上领先，但在面对不可读内容时以高置信度编造答案而非承认不确定。

**Split-half reliability**：100次随机划分下的Spearman ρ=0.979，验证行为评估结果的稳定性，证明质量-行为分离是模型固有属性而非噪声。

## 可复现要素
- **数据集**：250张arXiv科学图表（187篇论文，2023-2025），英文，开源；项目网站 https://scifigbench.nlp4sci.com/
- **代码/权重**：论文声明"dataset, evaluation scripts, model outputs, and prompts will be released upon publication"；详细prompt库见Appendix F
- **关键超参**：temperature=0（确定性解码），random seed=42；GPT-4o API version=2024-12-01-preview；max tokens=2,048（Gemini用16,000）
- **模型访问**：商业模型（GPT-5.2、Phi-4）走Azure OpenAI，开放模型走OpenRouter；模糊参数σ=25、α=0.3、旋转15°、blend factor=0.7、kernel=75
- **评估验证**：3位annotator独立评分120对（Krippendorff's α=0.91）；跨判官Mistral Large 3验证；探针设计师Ablation（GPT-4o vs Mistral Large 3）
