---
title: "Prompts-in-the-Wild-A-Large-Analyzed-Collection-of-Transacti"
source: https://arxiv.org/pdf/2608.12905v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 16:48:18"
field: "提示工程与大语言模型应用"
keywords: ["Prompt Engineering", "Transactional Prompts", "Dataset Construction", "Automated Annotation", "Multilingual NLP"]
innovations: ["首个大规模交易性提示数据集（57,640条）", "多模块化LLM自动标注框架（Prompt Ontology）", "正交可扩展的提示分类本体体系"]
benchmarks: ["GitHub Python Prompts", "62-language coverage"]
---

# 论文速读：Prompts-in-the-Wild-A-Large-Analyzed-Collection-of-Transaction-Prompts

## 一句话总结
本文首次从 GitHub 大规模收集并系统分析**交易性提示（Transactional Prompts）**，构建包含 **57,640** 条唯一提示的数据集，并提出一套**多模块化自动标注框架（Prompt Ontology）**，为提示工程研究提供了可复用的实证基础与结构化分析工具。

---

## 研究问题与动机
- **提示应作为独立研究对象**：现有研究将提示视为工具性接口，缺乏将其作为语言对象进行系统科学研究的视角。
- **交易性提示的结构性特征未被充分挖掘**：现有提示多为一次性交互，而嵌入软件工作流的参数化提示具有可复现性与工程化特征，值得专门研究。
- **领域缺乏共同术语与结构框架**：提示研究尚未形成标准化的分类体系与大规模实证基础，阻碍系统性比较与分析。
- **多语言与现实场景覆盖不足**：现有数据集多聚焦英语单模态场景，缺乏对真实生产环境中多语言、多变体提示的刻画。

---

## 核心贡献（创新点）
1. **构建首个大规模交易性提示数据集**：从 GitHub Python 代码中提取 57,640 条唯一提示，覆盖 62 种语言，提供迄今最全面的实证基础。
2. **提出 Prompt Ontology 本体框架**：定义正交可扩展的标注字段体系（语言、任务领域、输入/输出特征、结构、技术类型等），支持系统化分类与分析。
3. **设计多模块化 LLM 自动标注流水线**：通过 9 个专业 system prompt 实现提示的原子化拆解与结构化解析，支撑大规模标注且保持高准确率（核心技术标注准确率 98.5%）。
4. **开源数据集与交互式分析平台**：提供筛选、语义搜索、可视化 span 着色等功能，促进后续研究复用。

---

## 方法详解

### 数据集采集与清洗
- **数据来源**：GitHub Python 文件，匹配模式包括 OpenAI `chat.completions.create` API 调用与 LangChain `PromptTemplate` 构造函数。
- **静态分析**：递归追踪变量赋值，提取完整提示文本。
- **过滤流程**：初始 145,553 对象 → 过滤无效内容后得 85,209 → 去重后最终 **57,640** 条唯一提示（其中 36,916 来自 API，20,724 来自 PromptTemplate）。

### Prompt Ontology 标注体系
定义以下正交字段：
- **Languages**：提示文本语言 + 显式提及的语言（共 151 种语言/方言被提及）。
- **Task & Domain**：77 个领域粗/细粒度分类。
- **Input Characteristics**：指令/问题/上下文的硬编码 vs 变量输入、语言、结构、模态。
- **Output Characteristics**：输出模态、结构、语言、答案范式。
- **Prompt Structure**：角色-消息序列（System/User/Assistant）、每消息语言、指令分解（42 种语义类型）。
- **Prompting Techniques**：12 种技术分类（如 role assignment、structured output、decomposition 等）。
- **Meta-data**：ID、URL、时间戳、英文翻译。

### 自动标注流水线（多模块协作）
1. **text extraction**：从 API messages 字段提取可读文本，处理变量占位符。
2. **language identification & translation**：识别自然语言（严格区分编程语），翻译非英语片段。
3. **instruction blocks decomposition**：将提示拆解为原子指令块（28 类），标记 central/meta、negative 等属性。
4. **input blocks parsing**：识别 Directions、Context、Question 三部分，判断 variability。
5. **output type classification**：判定输出类型（short text/code/table 等），支持复合类型。
6. **answer paradigm identification**：识别答案范式（free_generation、extracted_span、ordering 等）。
7. **task & domain labeling**：映射到标准化 NLP 任务与领域标签。
8. **prompting technique detection**：检测 12 种技术使用情况。

每个模块以 system prompt 形式实现，输出 JSON 格式，支持迭代优化。

### 标注质量控制
- **LLM-based 自动标注**：迭代测试优化各模块提示词。
- **人工在环检查**：每类别采样约 100 条人工核查。
- **错误分析**：额外 100 条随机样本精标评估，核心技术标注准确率达 98.5%。

---

## 实验与结果

### 数据集统计
| 指标 | 数值 |
|------|------|
| 最终唯一提示数 | **57,640** |
| 涉及语言总数 | **62 种** |
| 英语占比 | **84.66%** |
| 多语言提示占比 | **6.3%**（其中 >99% 含英语） |
| 纯非英语提示占比 | **8.19%** |
| 显式语言提及总数 | 15,331 次，覆盖 **151 种** 语言/方言 |

### 结构与内容特征
- **指令块统计**：总 394,875 块（平均每提示 6.85 块）；核心指令占 **18.2%**，支撑/元指令占 **81.8%**。
- **消息结构**：双消息提示占 **64%**（96% 为 system-user），单消息占 **27%**，三消息占 **4%**。
- **输入 variability**：上下文作为变量输入占 **94%**（硬编码仅 6%）；问题/任务作为变量输入占 **70%**（硬编码 30%）。
- **Grounding**：89.25% 的提示包含基于上下文的 grounding。

### 任务与领域分布
- **NLP 任务覆盖**：QA、文本生成、信息抽取、摘要前四项合计 >**48%**。
- **可识别领域**：57.38% 的提示可归类至 77 个领域，前 10 领域覆盖 63.63%。

### 提示技术使用率
| 技术 | 使用率 |
|------|--------|
| Role assignment | **>45%** |
| Structured output | **12.9%** |
| Decomposition via prompt | **10.34%** |
| Sections | **7.83%** |
| Output demonstrations | **3.34%** |
| Decomposition via LLM | **0.56%** |

### 标注质量
| 字段 | 准确率 |
|------|--------|
| Prompting Techniques | **98.5%** |
| Context Language/Modality | **97.0%** |
| Central vs Meta | **96.4%** |
| Prompt Language | **93.0%** |
| Domain | **90.6%** |

---

## 相关工作脉络
- **提示工程研究**：与现有提示调优工作不同，本文聚焦**交易性提示的结构化分析**而非性能优化，提供首个大规模实证数据集。
- **代码生成的 LLM 应用**：本文从代码仓库提取提示，区别于直接在自然语言数据上构建提示数据集的工作。
- **多语言 NLP 数据集**：本文覆盖 62 种语言，填补多语言提示工程研究的空白。
- **提示分类体系**：Prompt Ontology 提供标准化分类框架，区别于零散的提示分类尝试。
- **自动化标注方法**：采用多模块 LLM 提示流水线，与单模型端到端标注形成对比，强调可解释性与模块可复用性。

---

## 局限性与未来方向
- **语言偏差**：英语提示占 84.66%，多语言提示中 >99% 含英语，非英语独立提示比例较低（8.19%）。
- **来源局限**：仅搜索 Python 文件，其他语言生态（如 JavaScript、Java）的提示未被覆盖。
- **领域覆盖不均**：仅 57.38% 提示可归类到已知领域，大量提示缺乏明确领域标签。
- **技术使用不均衡**：高级技术如 decomposition via LLM 使用率极低（0.56%），反映工程实践中的普及度差异。
- **未来方向**：扩展至多语言代码仓库、增加非 Python 生态、深入分析提示性能与结构的关系。

---

## 研究启发与可借鉴点
- **模块化标注架构**：将复杂标注任务拆解为多个专业 system prompt，提升可维护性与准确率，可迁移至其他文本分析任务。
- **变量追踪提取方法**：通过静态分析递归追踪变量赋值提取完整提示，适用于代码驱动的自然语言数据提取场景。
- **提示本体论设计思路**：正交可扩展的字段体系为其他语言对象的分类研究提供范式参考。
- **人机协作标注流程**：LLM 自动标注 + 人工抽样核查 + 错误分析闭环，平衡效率与质量，可复用于大规模数据集构建。
- **交易性提示视角**：将提示视为嵌入工作流的工程对象而非一次性接口，为系统性提示研究开辟新方向。

---

## 关键术语表
- **Transactional Prompts（交易性提示）**：可复现、参数化、嵌入软件工作流的自然语言指令，区别于一次性交互提示。
- **Prompt Ontology（提示本体论）**：正交可扩展的提示标注字段体系，涵盖语言、任务、结构、技术等多维度。
- **Instruction Blocks（指令块）**：提示中被拆解的原子化指令单元，分为核心指令与支撑/元指令。
- **Grounding（上下文锚定）**：提示中基于给定上下文生成输出的机制，89.25% 的提示具备此特征。
- **Answer Paradigm（答案范式）**：输出与输入的关系类型，如自由生成、 span 提取、排序等。
- **Prompting Techniques（提示技术）**：12 种标准化分类的技术类型，包括角色分配、结构化输出、分解等。
- **Variability（可变性）**：提示中输入部分是否为变量或硬编码的属性，影响提示的泛化能力。
- **Central vs Meta Instructions（核心 vs 元指令）**：核心指令直接描述任务目标，元指令提供支撑性指导。

---

## 可复现要素
- **数据集**：57,640 条交易性提示，论文未明确声明开源状态（需查阅原文补充）。
- **代码**：分析框架的系统 prompt 模板已在论文中完整披露，可复现标注流水线。
- **权重**：使用 embeddinggemma-300m-ONNX 进行语义搜索，为开源模型。
- **关键超参**：论文未详细提及训练超参（本工作主要为数据集构建与分析框架，非模型训练）。
