---
title: "How-Do-VLMs-Behave-When-Blind-or-Misled-Behavioral-Evaluatio"
source: https://arxiv.org/pdf/2608.13267v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:43:13"
field: "多模态大模型评测与可靠性"
keywords: ["VLM 评测", "科学图表理解", "行为可靠性", "幻觉评估", "A-R-I 框架", "SCIFIGBENCH", "选择性模糊", "抗拒探针"]
innovations: ["提出 SCIFIGBENCH 基准，联合评估感知/推理/行为可靠性三个维度", "提出 A-R-I 行为框架（承认-抵抗-推断）量化不确定性下 VLM 行为", "揭示高感知/推理准确率与行为可靠性的解耦现象"]
benchmarks: ["SCIFIGBENCH", "ChartQA", "SciFIBench", "CharXiv", "ChartMuseum", "CHAOS", "CHART-NOISe"]
---

# 论文速读：How-Do-VLMs-Behave-When-Blind-or-Misled-Behavioral-Evaluatio

## 一句话总结
本文提出了 SCIFIGBENCH 基准与 A-R-I 行为框架，系统评估 VLMs 在科学图表理解中面对视觉证据缺失或误导性上下文时的行为可靠性，揭示"高感知/推理准确率 ≠ 高行为可靠性"的关键发现：GPT-5.2 描述质量最高（MQM 91.6），却在 96% 的情况下幻觉不可读内容；Gemini 3.1 Pro 在承认不确定性（71%）和抵抗误导（0.91）上领先。

## 研究问题与动机
1. **现有基准盲区**：主流 VLM 图表基准（如 ChartQA、SciFIBench、CharXiv）主要评估感知与推理准确性，忽视了模型在视觉证据缺失或被误导时的行为可靠性。
2. **科学应用的高风险**：科学图表承载论文核心量化证据，误读会直接影响下游总结、对比与推理，因此行为可靠性是科研部署的关键瓶颈。
3. **能力与行为的脱节**：同等感知/推理能力的模型，在面对不确定情境时可能呈现截然不同的行为特征（如 GPT-5.2 vs. Gemini 3.1 Pro）。
4. **行为维度尚未被系统化度量**：现有工作虽涉及幻觉、诚实性、顺从性等，但未将"承认不确定性、抵抗误导、从部分证据推断"三者统一框架化并应用于科学图表。

## 核心贡献（创新点）
1. **提出 SCIFIGBENCH 基准**：包含 250 个专家标注的科学图表与 34,000+ 评估实例，首次将感知、推理与行为可靠性三者联合评估；与 SciFIBench、ChartMuseum 等基准的本质区别在于显式引入行为维度与对抗探针。
2. **构建 A-R-I 行为框架**：将行为分解为 Admittance（承认不确定）、Resistance（抵抗误导）、Inductance（从部分证据推断），区分于仅测量幻觉率或 abstention 率的既往工作，提供更细粒度的行为画像。
3. **设计多类可控探针套件**：包括抗性探针（inexist/contra/unanswerable）、caption bias 探针、选择性模糊（admittance blur/inductance blur）等，实现对"假设嵌入""虚假锚定值""图注污染"等真实部署风险的精确复现，区别于仅做图像扰动（旋转、噪声）的鲁棒性基准。
4. **实证揭示能力-行为解耦**：首次系统展示 MQM 排名前二的 GPT-5.2 与 Gemini 3.1 Pro 在行为维度上呈反向分布（96% 幻觉 vs. 71% 承认），证明仅凭准确率基准会遗漏关键部署风险。

## 方法详解
- **SCIFIGBENCH 数据集**：从 187 篇 arXiv 论文（2023–2025）抽取 250 张科学图表（99 条形图、99 折线图、52 饼图），由两名专家标注员独立描述（Krippendorff α = 0.91），第三人仲裁分歧；人工标注总时长超 600 小时。
- **感知评估（MQM 适配）**：基于 Multidimensional Quality Metrics 框架，针对每种图表类型手工设计 checklist（条形图 14 项、折线图 15 项、饼图 11 项），每项标记 Major/Minor 严重度；GPT-4o judge 对覆盖率与正确性打分，规则引擎映射至 Accuracy（权重 5.0/2.0）、Completeness（5.0/2.0）、Clarity（2.5/1.0）三类惩罚；最终得分 $MQM = \max(0, 100 - P \times 100 / (N \times 5))$，100 表示零错误。
- **推理评估（Capability Questions）**：每个图表 4 道问答（共 1,000 题），覆盖 Counting（枚举）、Computation（算术）、Comparison（相对判断）、Pattern Analysis（趋势识别）四类；题目由 GPT-4o 生成、Mistral Large 3 校验、人工审核，答案由 GPT-4o + 人工复核。
- **A-R-I 行为评估**：
  - **Admittance（承认）**：选择性模糊无法从上下文恢复的元素，模型应承认"看不到"；分别设计主动提问（active）与被动描述（passive）两种协议。
  - **Resistance（抵抗）**：四类探针——Inexist（假设不存在的元素存在）、Contra（嵌入错误数值锚定，偏差 20–30%）、Unanswerable（需图外信息）、Caption-bias（图注含 2–3 条合理但错误的声明）；评分 1.0（明确抵抗）/0.5（模棱两可）/0.0（接受/幻觉）。
  - **Inductance（推断）**：选择性模糊可从上下文推断的元素（如轴刻度可还原数值、颜色匹配可还原标签），评估模型是否正确推断而非乱猜。
- **图像变换**：高斯噪声（σ=25）、低对比度（α=0.3）、旋转（15°）、in-paper（嵌入原文 PDF 页面）等 1,243 个变换实例。
- **自动化评估 pipeline**：所有 judge 使用 GPT-4o（Azure，API 2024-12-01-preview），temperature=0，随机种子=42；通过人类 120 对标注（Krippendorff α=0.91）与 Mistral Large 3 交叉 judge 双重验证稳定性。

## 实验与结果
- **评测模型**：8 个商用与开源模型——GPT-5.2、Gemini 3.1 Pro、Llama 4 Maverick、Qwen3-VL-235B、Qwen3-VL-30B、Qwen3-VL-8B、Gemma-3-27B-IT、Phi-4 Multimodal。
- **感知（MQM）**：GPT-5.2 以 91.6（CI [90.4, 92.8]）略胜 Gemini 3.1 Pro 的 90.2（CI [88.9, 91.4]）（p<0.01，Cliff's δ=0.09）；Phi-4 最低（62.2），主要由完整性惩罚驱动（比领先者低近 30 分）。旋转是最具破坏性的变换（平均下降 19.4 分），噪声影响可忽略。
- **推理（Capability）**：Gemini 3.1 Pro 综合 81.0% 领先，GPT-5.2 次之 78.4%；两者与其余模型差距显著（无模型单项超过 53.4%）。Gemini 在 Counting（89.2% vs. 76.1%）和 Comparison（89.6% vs. 77.9%）占优，GPT-5.2 在 Computation（82.8% vs. 79.4%）和 Pattern（72.0% vs. 70.0%）微弱领先。Phi-4 全面崩溃（8.6%）。
- **行为（A-R-I）**：
  - **Admittance（主动）**：Gemini 3.1 Pro 71%，遥遥领先；GPT-5.2 仅 8%，却以 96% 概率幻觉不可读内容，呈现"自信型幻觉者"画像。
  - **Resistance**：Gemini 0.91 最优，GPT-5.2 为 0.81（p<0.001），Phi-4 垫底 0.21。最难抵抗的是 Inexist（ presupposition embedding），最易抵抗的是 Unanswerable。
  - **Inductance（主动）**：Gemini 66%，GPT-5.2 为 59%，证明其在上下文可推断时能做出正确推理。
  - **Caption bias**：Gemini 与 GPT-5.2 同为 0.89，Phi-4 仅 0.05（95% 跟随篡改图注）。
- **关键结论**：MQM 排名前二模型在行为维度上呈反向分布——仅依赖准确率基准会掩盖关键部署风险。

## 相关工作脉络
1. **SCIFIGBENCH vs. ChartQA / CharXiv / SciFIBench**：这些基准聚焦感知与推理准确性，缺少行为可靠性、对抗探针与不确定性评估；SCIFIGBENCH 是唯一同时覆盖感知-推理-行为三维度且支持对抗条件的基准（见 Table 1）。
2. **SCIFIGBENCH vs. CHAOS / CHART-NOISe**：后两者评估图像扰动（噪声、遮挡）下的性能衰减，但不检测模型在误导性上下文（虚假前提、篡改图注）中的行为响应。
3. **SCIFIGBENCH vs. CHOCOLATE / ChartHal**：后者专注图表幻觉的定性分类，SCIFIGBENCH 则通过 A-R-I 框架给出可量化的行为分数（0–1 尺度），并区分被动/主动两种评估协议。
4. **A-R-I vs. BeHonest / Selective Prediction 工作**：既往诚实性/拒答基准主要面向纯文本 LLM，SCIFIGBENCH 将其拓展至多模态科学图表，并细化为承认、抵抗、推断三个可分离维度。
5. **MQM 适配 vs. 传统 BLEU/ROUGE 评估**：借鉴机器翻译 MQM 框架，首次针对科学图表描述设计清单式多维惩罚（Accuracy/Completeness/Clarity + 子类型），并通过绑定验证（binding verification）检测属性错配错误。
6. **Selective Blur vs. 普通 Occlusion**：不同于随机遮挡，选择性模糊通过 OCR 定位、两阶段模糊（灰度混合+重度高斯模糊）、人工确认，确保目标不可恢复性或可推断性可控，使 Admittance/Inductance 能被独立测量。

## 局限性与未来方向
1. **图表类型覆盖有限**：仅涵盖条形图、折线图、饼图，尚未包含散点图、热力图、网络图、示意图等常见科学可视化类型。
2. **语言单一**：目前仅限英文 arXiv 论文，扩展到非英语语料是重要方向。
3. **无法完全分离能力与指令压力**：顶级模型 MQM ≥ 90 仍失败于虚假前提探针，可能是训练/对齐偏好所致而非视觉能力缺陷，但 benchmark 本身无法 conclusively 归因；需通过提示干预或开放模型内部探测进一步隔离。
4. **自动化 judge 依赖 GPT-4o**：虽经人类验证（α=0.91）与跨 judge 鲁棒性检验，仍存在系统性低估（mean bias −15.0 MQM）与幻觉内容漏检（recall=0.07）的已知偏差。
5. **未来方向**：扩展至更多图表类型与非英文语料；探索不同架构（MoE vs. dense）与规模对行为维度的影响；设计针对 vision encoder 与 language head 的消融分析；引入 confidence calibration 与多轮对话一致性等新行为轴线。

## 研究启发与可借鉴点
1. **A-R-I 框架可迁移至医疗影像、工业检测等高风险多模态场景**：模型在证据模糊时应能承认局限、抵抗误导上下文、从部分线索谨慎推断，这一三维划分具有通用价值。
2. **MQM 清单式评估设计值得借鉴**：针对不同图表类型定制 checklist + 绑定验证 + 严重度加权惩罚，可推广至任何开放描述型多模态评测任务。
3. **选择性模糊的两阶段生成流程**（OCR 定位 → 模糊候选 → 人工确认）可作为构建可控不确定性感知的标准 pipeline，适用于评测图像字幕、视觉问答等任务中的幻觉抑制。
4. **主动 vs. 被动双协议行为评估**：同一探针分别以直接提问与开放式描述两种模式施测，能捕捉模型在"被追问"时的自信型幻觉倾向（如 GPT-5.2 被动 23% vs. 主动 8% 承认率），建议纳入未来行为基准设计。
5. **Caption bias 探针设计可复用于 RAG/文档理解场景**：通过注入 2–3 条合理但错误的图注声明（70/30 规则），模拟多模态 RAG 中错误检索引起的上下文污染，为评测文档级 VLM 提供现成方案。

## 关键术语表
**SCIFIGBENCH**：论文提出的科学图表 VLM 行为评测基准，覆盖 250 张 expert-annotated 图表与 34,000+ 评估实例，联合衡量感知、推理与行为可靠性三个维度。

**A-R-I 框架**（Admittance–Resistance–Inductance）：行为评估三维框架——Admittance 指模型在证据缺失时承认不确定性；Resistance 指模型拒绝虚假前提与误导性上下文；Inductance 指模型从可推断的部分证据中进行有界推理。

**MQM（Multidimensional Quality Metrics）**：源自机器翻译质量评估的框架，本文适配为科学图表描述的清单式多维评分体系，惩罚维度包括 Accuracy、Completeness、Clarity 及若干子类型。

**Selective Blur**：通过 OCR 定位图表特定元素后施加灰度混合+重度高斯模糊的人造退化操作，用于构造 Admittance（不可恢复）与 Inductance（可推断）两类可控不确定性条件。

**Resistance Probe**：测试模型抵抗误导能力的探针家族，包括 Inexist（假设不存在元素）、Contra（嵌入错误数值锚定）、Unanswerable（需图外信息）三类，分别对应 presupposition embedding、anchoring bias、cooperative pressure 等认知偏差。

**Caption Bias**：在图注中嵌入 2–3 条合理但错误的声明（覆盖 30% 内容），评估模型在图文冲突时更信赖视觉证据还是文本上下文的倾向。

**Active vs. Passive 评估协议**：Active 为针对模糊区域的直接提问，Passive 为开放式描述任务；同一探针通过两种协议施测可揭示模型在"被追问"压力下降低不确定性承认率的"must-answer bias"。

**Confident Fabricator 画像**：指像 GPT-5.2 这样在感知/推理上表现顶尖，却在 96% 的不可读情境下自信幻觉、仅 8% 承认不确定的行为模式，凸显 fluent 输出与 epistemic honesty 的脱节。

## 可复现要素
- **数据集**：250 张科学图表（99 条形图、99 折线图、52 饼图），来源于 187 篇 arXiv 预印本（2023–2025）；论文声明 Dataset、evaluation scripts、model outputs 与 prompts 将在发表后公开，项目网站为 https://scifigbench.nlp4sci.com/。
- **代码/权重**：代码库与提示词文件列于 Appendix F；8 个评测模型均为已发布版本（GPT-5.2、Gemini 3.1 Pro、Llama 4 Maverick、Qwen3-VL-235B/30B/8B、Gemma-3-27B-IT、Phi-4 Multimodal）。
- **关键超参**：所有模型 temperature=0，max tokens=2,048（Gemini 用 16,000）；随机种子=42；judge 使用 GPT-4o（Azure OpenAI，API 2024-12-01-preview）；Bootstrap 置信区间 B=10,000。
- **图像变换参数**：高斯噪声 σ=25，低对比度 α=0.3、β=50，旋转 15°，模糊 kernel=75、blend factor=0.7。
- **人工标注**：5 位研究生，总时长 600+ 小时；MQM 交叉标注 120 对（Krippendorff α=0.91）。
