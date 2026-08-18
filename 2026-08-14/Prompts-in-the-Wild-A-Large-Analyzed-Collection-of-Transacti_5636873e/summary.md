---
title: "Prompts-in-the-Wild-A-Large-Analyzed-Collection-of-Transacti"
source: https://arxiv.org/pdf/2608.12905v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 16:49:12"
field: "提示工程与大语言模型分析"
keywords: ["prompts", "transactional prompts", "prompt ontology", "automated annotation", "multilingual prompts"]
innovations: ["构建首个大规模事务性提示词数据集（57,640条）", "设计多维结构化本体论与自动化标注流水线", "系统评估标注精度并揭示错误模式"]
---

# 论文速读：Prompts-in-the-Wild-A-Large-Analyzed-Collection-of-Transacti

## 一句话总结
本文构建了首个大规模事务性提示词数据集（Prompts‑in‑the‑Wild，共 **57,640** 条），并设计了一套结构化本体论与自动化标注流水线，旨在将非正式的提示词文本转化为可定量分析的结构化语言学对象，从而支持提示工程、可解释性与评测基准构建等下游研究。

## 研究问题与动机
- 现有提示词研究多依赖人工设计的小型数据集或一次性交互提示，缺乏对嵌入真实软件工作流（尤其是 GitHub Python 仓库）的事务性提示词的大规模、系统性分析。
- 提示词缺乏共享的形式化框架与术语体系，阻碍了定量比较、跨语言研究以及自动化处理。
- 现有自动化工具（如正则匹配）难以捕捉提示词的深层语义结构，需要一种基于本体论的多维度标注方案。
- 提示词工程实践高度依赖经验直觉，亟需实证数据以揭示其语言特征、多语言分布、任务/领域构成及指令结构规律。

## 核心贡献（创新点）
1. **首个面向事务性提示词的大规模数据集**：从 GitHub 公开 Python 仓库中自动收集 57,640 条独立提示词（36,916 条来自 `chat.completion.create`，20,724 条来自 LangChain `PromptTemplate`），并记录时间戳。
2. **多维结构化本体论（Prompts Ontology）**：定义正交且可扩展的分析维度（语言、任务/领域、输入/输出特征、指令结构、提示技术、元数据），将非结构化文本转化为富含结构的可计算对象。
3. **基于 LLM 的自动化标注流水线**：设计多个专用子系统（提示技术检测、推理链发现、指令块分解、上下文/问题/方向识别、变异性分析等），实现端到端结构化标注。
4. **系统的错误分析与精度报告**：通过人工抽样验证各字段标注质量，揭示高精度字段（如 Prompt Language 93.0%、Prompting Techniques 98.5%）与低精度字段（如 Output Type 60.4%、Directions Text 69.4%）的差异及错误模式。
5. **丰富的初步分析发现**：揭示数据集语言分布（62 种语言，英语占 84.66%）、模态构成（text→text 占 77.31%）、领域长尾分布（77 个领域，Top 10 覆盖 63.63%）及指令结构特征（平均每条提示词 6.85 个指令块，否定指令约占 31%）。

## 方法详解
### 数据收集
- **来源**：GitHub 公开仓库（仅 Python 文件）。
- **检测规则**：扫描调用 `chat.completion.create` API 或 LangChain `PromptTemplate` 构造函数的文件。
- **解析流程**：AST 静态分析递归追踪变量赋值与函数调用，提取完整提示文本；经启发式过滤（去除空值、纯标点、无动词结构等）后去重。
- **元数据**：每条提示词记录最后一次修改提交日期。

### 本体论维度
- **Languages**：提示词实际使用的语言 + 显式提及的语言。
- **Task & Domain**：粗粒度与细粒度任务、应用领域分类（共 77 个领域）。
- **Input characteristics**：高层指令、具体问题/任务、支持性上下文（标注来源：硬编码 vs. 变量输入）。
- **Output characteristics**：模态、请求结构、语言、答案范式。
- **Prompt structure**：角色消息序列（System/User/Assistant），每消息标注语言、显式提及语言；进一步拆分为指令序列，归类为 **42 种语义类型**。
- **Prompting techniques**：12 种预设提示技术（含证据 span）。
- **Meta‑data**：ID、GitHub URL、最后更新时间、完整提示词文本、非英语提示的英文翻译。

### 自动化标注子系统
1. **提示技术检测**：大段 system prompt + JSON 输出，检测 10+ 项技术（如 persona/role assignment、few‑shot examples、structured output、tool calling 等），每项需提供精确原文 span 作为证据。
2. **Chain‑of‑Thought 发现**：检测 prompt 是否显式要求 `reasoning/explanation/thinking/chain‑of‑thought` 字段，并区分 `reasoning_first = true/false`。
3. **指令块分解器**：将每个 message 拆解为有序指令块序列（~28 类预定义名称），每块包含 `instruction_kind`、`instruction`（原文 exact substring）、`is_central`、`is_negative` 等字段，允许跨块重叠。
4. **Input Context / Question / Directions 识别**：将 prompt 分为三层：Directions（高层动作）、Context（支撑信息）、Question‑like part（具体请求）；每个 span 标注类型（`direct_content` 或 `description`）。
5. **变异性分析**：判定 context 和 question 部分是否包含变量/占位符（`fixed` / `varying` / `none` / `undefined`）。
6. **语言与结构分析**：对 directions/context/question 推断自然语言列表、结构类型（Single/Pair/Tuple/List/Dictionary/`undefined`）及 context 模态。
7. **输出分析**：识别输出单元的模态、描述、来源（extracted/generated）、语言、结构，并区分多个合并值仍按 item 数计为 list/pair/tuple。

### 质量控制
- 基于 LLM 自动标注 + 人工迭代校验（每主要类别采样 ≈100 条人工检查）。
- 错误分析基于额外 100 条随机样本，统计各字段正确率与主要错误类型（幻觉、类别混淆、遗漏等）。

## 实验与结果
- **数据集规模**：57,640 条独立提示词（36,916 来自 `chat.completion.create`，20,724 来自 `PromptTemplate`）。
- **语言覆盖**：62 种语言；英语占 84.66%；超 1% 语言包括中文、韩文、西班牙语、日文、葡萄牙文；多语言提示词仅占 6.3%，其中超 99% 含英语。
- **模态分布**：text→text 占 77.31%；其余为 ungrounded/undefined（18.98%）、image→text（1.98%）、image+text→text（0.66%）、audio→text（0.23%）。
- **领域分布**：77 个不同领域，呈 Zipf 长尾；Top 10 领域覆盖可识别领域提示词的 63.63%，包括 education & instruction、software development、business & commerce 等。
- **指令结构**：总指令块数 394,875，平均每条提示词 6.85 个；高频指令 Top 5 覆盖 81.33%（input context placeholder、constraint/restriction、output content/format requirement、role specification）；核心指令占 18.2%，元/辅助指令占 81.8%；否定指令约占 31%。
- **标注精度**：
  - 高精度字段：Prompt Language **93.0%**、Domain **90.6%**、Context Language/Modality **97%**、Central vs. Meta **96.4%**、Prompting Techniques **98.5%**。
  - 较低精度字段：Output Type **60.4%**、Directions Text **69.4%**、Answer Paradigm **80.2%**、Instruction Kinds **89.8%**、Context Structure **81.9%**。
  - 主要错误类型：幻觉、类别混淆、遗漏、分割与粒度错误；间接推理场景（共指消解、常识推理、领域知识）性能下降最明显。

## 相关工作脉络
- **PromptSource / OpenAI Prompt Hub**：前者为多任务 prompt 集合，但侧重人工设计；本文聚焦真实软件仓库中的事务性提示词，规模更大、来源更自然。
- **IFEval / PROMPTFOOL**：侧重提示词安全与指令遵循评测；本文侧重语言学本体分析与大规模结构刻画，而非评测。
- **Chain‑of‑Thought 检测研究**：多针对特定推理范式；本文提供系统化、可量化的 CoT 字段检测框架，并区分推理顺序。
- **多语言 NLP 数据集**（如 XTREME、FLORES）：聚焦平行语料；本文揭示多语言提示词的真实分布（英语主导、多语言提示稀少且高度英化）。
- **代码生成与提示工程**（如 AutoPrompt、In‑Context Learning 研究）：多关注模型性能提升；本文提供提示词本体论与分析工具，为后续优化提供数据基础。
- **提示词聚类与分类**（如 Prompt Clustering by LLM）：本文的聚类流程在此基础上增加人工校验与错误分析，强调精度评估。

## 局限性与未来方向
- **数据源偏倚**：仅来自 GitHub Python 仓库，可能偏向技术用户与编程场景，未能覆盖企业级、非代码或低资源语言社区。
- **语言代表性不足**：非英语提示词仅占约 8%，多语言提示词更少（6.3%），限制了跨语言比较的可靠性。
- **标注精度不均**：Output Type、Directions Text 等字段准确率低于 70%，主要源于幻觉、类别混淆及复杂推理场景的困难。
- **技术覆盖有限**：仅检测 12 种预设提示技术，未涵盖新兴或复合技术（如 tree‑of‑thought、self‑consistency）。
- **未来方向**：扩展至其他代码库（如 JavaScript、Java）与自然语言应用仓库；引入多模态提示词（图像/音频输入）；改进低精度字段的标注策略（如人工微调、混合方法）；将本体论推广至其他指令系统（如 chatbot、agent workflow）。

## 研究启发与可借鉴点
- **提示词作为独立研究对象**：可将提示词视为语言学/软件工程对象，进行本体论建模与定量分析，而非仅视为模型输入。
- **自动化标注流水线设计**：分模块处理（技术检测、CoT 发现、指令分解、上下文识别）有助于降低单模型复杂度，便于故障定位与精度提升。
- **证据驱动判定原则**：要求所有判定基于原文显式证据（span），禁止无依据猜测，可提升可解释性与可复现性。
- **错误分析与精度分层**：通过人工抽样揭示各字段误差模式，可为后续研究提供质量基准与改进方向。
- **多语言提示词分析框架**：本体论同时捕获“实际使用语言”与“显式提及语言”，有助于研究多语言模型的提示适应行为。
- **向本团队方向结合的机会**：可将本数据集用于跨语言提示工程研究、多模态提示词分析、或作为下游评测基准（如 Instruction Following Benchmark）的数据源。

## 关键术语表
- **事务性提示词（Transactional Prompts）**：嵌入软件工作流、可复现、参数化、鲁棒的提示词，区别于一次性交互提示。
- **Prompts Ontology**：多维度结构化本体论，涵盖语言、任务/领域、输入/输出特征、指令结构、提示技术等正交维度。
- **指令块（Instruction Blocks）**：prompt 中被分解为有序语义单元的最小指令片段，共约 28 类预定义名称。
- **Answer Paradigm**：输出单元与输入的关联方式，如自由生成、精确提取、二元答案、排序等。
- **Chain‑of‑Thought Detection**：检测 prompt 是否显式要求推理字段，并区分推理顺序（先思考后回答 vs. 先回答后解释）。
- **Context Variability**：判定上下文中是否包含变量/占位符，标记为 `fixed`、`varying`、`none` 或 `undefined`。
- **Prompting Techniques**：预设的 12 种提示工程模式，如少样本示例、结构化输出、工具调用、提示链等。
- **Central vs. Meta Instructions**：核心任务指令（直接要求模型执行的任务）与元指令（关于如何执行任务的辅助说明）的区分。

## 可复现要素
- **数据集**：Prompts‑in‑the‑Wild，包含 57,640 条提示词及标注元数据。**公开状态：论文未明确声明开源，但数据收集流程清晰**。
- **代码**：自动化标注流水线各子系统代码部分开源（GitHub 链接见论文附录），包含提示技术检测、CoT 发现、指令分解等模块。
- **关键超参数**：标注 LLM 采用 GPT‑4 / Claude 系列（具体版本论文未详述）；聚类阈值（>100 词项重聚类、<5 词项并入 other）；人工校验采样量每类别约 100 条。
- **环境依赖**：Python 3.8+、AST 解析库、LLM API 调用封装、JSON 输出解析模块。
