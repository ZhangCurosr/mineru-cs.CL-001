---
title: "REAP: Relation-Aware Elicitation and Parsing for Closed-Book Knowledge Base Construction from LLMs"
source: https://arxiv.org/pdf/2608.10963v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:53:17"
---

# 论文速读：REAP: Relation-Aware Elicitation and Parsing for Closed-Book Knowledge Base Construction from LLMs

## 一句话总结
本文针对AKBC Shared Task 2026的闭卷知识基构建任务，提出REAP两阶段流水线：通过关系特异性CoT推理与空集门控显式 elicitation 参数化事实，再经确定性正则解析高效序列化JSON答案；在≤32B参数且零微调约束下，基于Mistral-Small-24B-Instruct-2501在测试集取得macro-F1 0.62。

## 研究问题与动机
1. **可变基数抽取难题**：LLM参数中虽蕴含大量事实，但将(s, r)映射到知识图谱对象集时，答案可能是空集(∅)、单值或长列表，直接单次生成极易产生幻觉、遗漏或违反JSON Schema。
2. **传统探测方法局限**：LAMA等cloze-style提示与提示集成主要假设单一对象，难以稳定覆盖任意基数场景；而直接生成式方法在格式约束与长列表召回上表现均较弱。
3. **严格任务约束下的效率需求**：AKBC 2026限定闭卷（无检索增强）、≤32B参数、零微调，要求充分挖掘模型内部参数知识，同时控制推理成本，亟需一套轻量、可复用的结构化提取框架。

## 核心贡献（创新点）
1. **关系感知提取（Relation-Aware Elicitation）**：针对6种目标关系设计专属的多步CoT提示策略（如四维地理扫描、时间轴分代查询、容量范围校验），与通用prompt的本质区别在于按关系语义特征解耦查询分解方式。
2. **高效的混合解析机制（Efficient Hybrid Parsing）**：约80%记录通过确定性正则与平衡括号扫描直接完成JSON序列化，仅复杂情况回退至LLM抽取，相较全LLM抽取显著降低计算开销。
3. **基于推理的空集门控（Reasoning-based Empty-Set Gate）**：在CoT中显式插入“属性不存在或证据不足则直接输出`FINAL_ANSWER: []`”的结构化决策节点，区别于依赖概率校准的拒绝机制，直接抑制闭卷设定下的幻觉输出。

## 方法详解
REAP采用两阶段流水线，Stage 1负责事实 elicitation，Stage 2负责答案序列化与格式合规。
- **Stage 1 关系特异性Prompt设计**：遵循三大原则：(i) 查询分解以最大化多值关系召回；(ii) 强制输出格式约束（`BORDERS:` 或 `FINAL_ANSWER: [...]` 标记）；(iii) 逐步推理配合空集门控。Few-shot示例固定为每条关系一正一负，严格取自训练集或手工构造，杜绝数据泄露。
- **各关系推理策略**：
  - `awardWonBy`：时间轴六轮分代查询（创立~1970s、1980s-90s、2000s-10s、2020s+、非西方/次要获奖者），T=0.7采样以多样化长名单回忆。
  - `countryLandBordersCountry`：四向地理扫描（东/西/南/北边界），明确排除海洋边界，岛屿国家返回`BORDERS: NONE`；若响应缺失标记则自动重试更短fallback prompt。
  - `companyTradesAtStockExchange`：四步CoT（识别公司→严格公开性检查/空集门控→检索完整官方交易所名→输出）。
  - `hasArea`：四步CoT（实体消歧→提取面积→单位换算为km²→输出整数/小数）。
  - `hasCapacity`：四步CoT+范围校验（区分场馆类型/同名地点→核对合理区间1k–35k或35k–100k座位）。
  - `personHasCityOfDeath`：五步CoT+空集门控（检查生存状态→名称消歧→检索死亡城市→隔离地名→存活/未知则输出`[]`）。
- **Stage 2 后处理**：直接解析`FINAL_ANSWER:`行或使用平衡括号扫描恢复截断数组；正则提取并归一化数值（去除千位分隔符与单位）；过滤年份前缀、HTML标签、头衔（Dr./Prof./Sir等）；去除尾部括号限定词并进行大小写不敏感去重（保留首现大写形式）。

## 实验与结果
- **数据集与设置**：AKBC Shared Task 2026，含6个关系（地理邻国、死亡城市、场馆容量、奖项获得者、上市公司、面积），验证集与测试集各475条，覆盖∅、单值、多值及定量关系。评估采用组织者提供的脚本计算macro/micro P/R/F1，字符串经最大二分图匹配匹配别名，数值允许5%相对误差。硬件为Kaggle TPU v5e-8，vLLM服务，bfloat16，batch_size=32，全关系一轮验证集推理约2–5分钟。
- **基线模型**：Gemma-2-9B-it、Llama-3.1-8B-Instruct、Mistral-Small-24B-Instruct-2501，及组织者Qwen3.5-9B baseline。
- **主要结果**：零样本提示下三模型macro-F1均为0.38–0.43；应用REAP两阶段流水线后分别提升至0.48/0.51/0.65。最终选用Mistral-24B，在官方测试集取得**macro-F1 0.62**。分项最强为`countryLandBordersCountry` (**F=0.95**)，其次`hasArea` (0.77)与`companyTradesAtStockExchange` (0.73)；最弱为
