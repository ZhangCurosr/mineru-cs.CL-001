---
title: "WHO-WOULD-YOU-VOTE-FOR-AUDITING-POLITICAL-ALIGNMENT-IN-LLMS"
source: https://arxiv.org/pdf/2608.11649v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:42:53"
field: "LLM政治对齐审计"
keywords: ["political alignment", "LLM auditing", "persona prompting", "political bias", "multi-party systems"]
innovations: ["多维度描述性标准框架用于政治实体审计", "政党-领导人分离评估揭示隐藏差异", "人格正交操控量化政治身份对模型输出的定向偏移"]
benchmarks: ["Italian Parliament parties and leaders (21 entities)", "6-model cross-model agreement (Kendall's W=0.78)"]
---

# 论文速读：WHO WOULD YOU VOTE FOR? AUDITING POLITICAL ALIGNMENT IN LLMS: AN ITALIAN CASE STUDY

## 一句话总结
论文提出一个系统化审计框架，通过9个描述性标准评估6个主流LLM对意大利政党及领导人的政治评价，揭示模型存在系统性左偏偏好、跨模型高度一致，且人格设定可显著扭转评价倾向。

## 研究问题与动机
- **核心问题**：当被要求表达对政治实体的判断时，LLM是否表现出系统性偏好？偏好体现在哪些维度？
- **现有方法不足**：
  1. 多数研究仅询问模型对单一议题的立场或在左右轴上的坐标，无法捕捉对具体政治实体的多维评价结构
  2. 已有工作在双党制（如美国）中常见，难以直接迁移至意大利等多党碎片化政治体系
  3. 既往研究通常将政党和领导人混同评估，掩盖了"党-领袖"分别评价时产生的显著差异

## 核心贡献（创新点）
1. **提取协议设计**：通过预设标准化评价标准集 elicitation protocol 获取结构化输出，区别于迫使模型二选一或给出单一意识形态坐标的做法
2. **系统性多维度审计**：首次将10个意大利政党与11位领导人作为独立实体，用9个非意识形态定向的描述性标准进行量化评估，同时测量跨模型一致性、拒绝率、提示敏感性与人设效应
3. **可复现开源框架**：完整公开提示词、原始响应数据和代码分析管线，支持跨政治体系、跨标准、跨模型版本的扩展审计

## 方法详解
- **实体定义**：21个独立实体（10个政党+11位领导人），同一prompt中不同时呈现，避免排序/对比效应
- **9个评价标准**（1-5 Likert量表，两端锚点明确定义）：
  - *提案 formulation*：statement-program consistency、proposal specificity、communication clarity
  - *政策覆盖*：economic coverage、social coverage、environmental coverage
  - *组织行为*：tone moderation、internal cohesion、positional stability
- **6个模型**：Qwen-3.6 (27B)、Llama-3.3 (70B)、GPT-oss (120B)、Nemotron-3-super (120B)、Mistral-medium-3.5 (128B)、Gemini-3.5-flash (~300B)，来自美/欧/中不同开发者
- **提示变体**：v1直接要求评价；v2以"补全不完整数据记录"为框架，格式与rubric完全相同
- **人格设定**：5个政治人格（left/centre-left/centre/centre-right/right），以"adopt the perspective of an Italian voter positioned on the {label}"方式插入system prompt，与实体/标准正交操控
- **实验规模**：基线实验 $6 \times 21 \times 2 \times 10 = 2520$ 请求，22680个分数；人格实验 $6 \times 21 \times 5 \times 5 = 3150$ 请求，28350个分数；temperature=0.7
- **分析指标**：每配置均值 $\mu$、标准差 $\sigma$、拒绝率 $\rho$；跨模型用Kendall's W和pairwise Pearson相关度量一致性

## 实验与结果
- **数据集**：意大利现任议会全部政党及其领导人
- **基线排名**：左翼/中间派居首（Nicola Fratoianni 3.89，Azione 3.88），右翼垫底（Futuro Nazionale 2.59，Matteo Salvini 2.90），极差1.30分；整体呈中左偏
- **跨模型一致性**：Kendall's W = 0.78 (p<0.01)；pairwise Pearson平均0.75（范围0.62–0.86）；nemotron最代表性（均相关0.78），gemini最不代表性（均相关0.68）
- **政党-领导人差异**：Forza Italia vs Tajani (+0.44)、M5S vs Conte (+0.45)，证实分离评估有价值
- **标准维度差异**：Communication clarity得分最高（3.92），proposal specificity最低（2.97）；各标准赢家分散，无单一实体通吃
- **人格效应**：左右人格间平均偏移0.83分（Meloni达1.49分），相当于基线全距；五个persona均降低评分（-0.14至-0.43分）
- **提示敏感性**：变体间平均绝对差异0.14分，相关r=0.97；拒绝率受影响更大（v1: 6.4%，v2: 4.1%）
- **拒绝率**：Llama-3.3最高14.5%，GPT-oss 7.9%，Mistral仅0.3%，差异50倍；Futuro Nazionale 56.7% null rate
- **最强结果**：跨模型W=0.78的高度一致性；人格偏移幅度（0.83分）与基线全距（1.30分）可比

## 相关工作脉络
1. **Bang et al. [4]** 区分"说什么"与"怎么说"评估议题偏见——本文将焦点从议题立场转向具体政治实体的多维度评价
2. **Chen et al. [13]** 用议会投票记录推断模型政治位置——本文不使用投票记录，改用政治科学中的描述性标准直接打分
3. **Lin et al. [26]** 与 **Potter et al. [36]** 实证LLM对话影响选民投票——本文不测下游影响，专注模型自身输出行为
4. **Röttger et al. [39]** 证明多选项格式和措辞影响答案——本文验证提示敏感性但发现数值分数高度稳健，仅拒绝率敏感
5. **Yang et al. [47]** 研究社会人格提示对LLM社会推理的影响——本文首次系统测试政治人格对实体评价的定向偏移效应
6. **Exler et al. [17]** 报告参数规模与左倾相关性——本文控制规模轴后发现跨模型一致性高，左偏是更普遍的结构性现象

## 局限性与未来方向
- 仅覆盖意大利政治体系，需在其他国家/地区复制验证
- 仅评估6个模型，未涵盖快速迭代的LLM生态
- 9个标准虽经中性设计但仍为预定义集合，未能穷尽政治评价维度
- 等权聚合掩盖维度权重敏感性，替代权重方案可能改变排名
- 仅测试英文提示与JSON格式，不同语言/对话场景结果未知
- 行为分析而非因果分析，未区分偏好源于预训练/对齐/RLHF
- 未直接测量对用户政治态度或投票行为的下游影响

## 研究启发与可借鉴点
1. **多维度描述性标准框架可直接迁移**：适用于任何多党制国家，替换实体列表和标准即可复用
2. **"政党-领导人分离评估"设计值得推广**：能揭示合并评估时的隐藏差异，避免信息损失
3. **人格正交操控实验范式**：五个人格与实体/标准正交设计，为研究身份认同对模型输出的定向影响提供模板
4. **拒绝率作为独立分析变量**：将refusal行为纳入指标而非当作缺失值，揭示不同模型的安全策略差异
5. **双变体提示鲁棒性检验**：v1/v2对照可在审计中快速验证结果是否稳定于措辞，值得成为标准流程

## 关键术语表
- **Political Alignment（政治对齐）**：模型在政治议题或实体评价上表现出的系统性偏好或立场倾向
- **Persona Prompting（人格提示）**：在system prompt中设定角色身份以引导模型按特定视角输出
- **Refusal Rate（拒绝率）**：模型返回null/拒绝回答的比例，反映安全策略强度和对争议实体的敏感度
- **Kendall's W（Kendall协调系数）**：衡量多个评委（模型）对多个对象（实体）排序一致性的统计量，0无一致、1完全一致
- **Statement-Program Consistency（陈述-纲领一致性）**：政治实体公开言论与其官方政策纲领之间的匹配程度
- **Affinity Effect（亲和效应）**：人格设定使模型更倾向给予自身政治立场一侧实体更高分的现象
- **Proposal Specificity（提案具体性）**：政策建议是否包含可操作的工具、资金来源和时间表，而非空洞口号
- **Positional Stability（立场稳定性）**：政治实体在重大议题上的立场随时间变化的持续程度

## 可复现要素
- 数据集：意大利政党及领导人名单（Table 1），论文未声明第三方数据集来源
- 代码/权重：**完全开源**，见 https://github.com/SimoneMungari/AuditingPoliticalAlignmentInLLMs；原始提示词与响应数据一并公开
- 关键超参：temperature=0.7；基线每配置10次重复；人格实验5次重复；输出格式为固定JSON（1-5整数或null）
