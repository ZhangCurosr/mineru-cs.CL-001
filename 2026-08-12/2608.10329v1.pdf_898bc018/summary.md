---
title: "Who Gets Heeded? An Obligation-Level Audit of Responsiveness in EPA Rulemaking"
source: https://arxiv.org/pdf/2608.10329v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-16 20:19:38"
---

# 论文速读：Who Gets Heeded? An Obligation-Level Audit of Responsiveness in EPA Rulemaking

## 一句话总结
本文提出“义务级响应审计”（obligation-level responsiveness auditing）框架，通过LLM辅助的结构化提取与分层盲审验证，首次在离散监管义务层面量化美国EPA公众意见与规则文本修订之间的共现关系，揭示出公众参与的形式程序开放性与实质响应能力之间存在上游结构性不平等。

## 核心贡献
1. **概念/测量贡献**：将监管响应的分析单位从“规则文档/听证人组/整条意见”下沉至离散的“监管义务”（regulatory obligation），使测量粒度与公众实际诉求单元及机构修订对象严格对齐。
2. **实证贡献**：基于大规模EPA语料得出三组描述性发现——参与热度与修订呈弱正相关；支持/反对方向对修订率无显著影响；组织主导意见在跨听证人组层面更集中于“编辑性完善”而非“实质性修改”。
3. **方法学贡献**：构建可审计的LLM辅助管道（义务提取+意见-义务匹配+结果分类）并建立透明验证层级；盲审直接暴露文本相似度分类器在关键编辑vs实质边界上的测量失效（κ=0.137），为算法问责提供可复用的审计基础设施与测量有效性教训。

## 方法详解
- **语料与样本构建**：整合2010–2022年EPA联邦公众意见库（786,197条，6,145个docket），采用分层随机+极端案例抽样选取36个anchor rulemakings，最终获得70,075条可分析意见（68,441条内联文本 + 1,634条经pdfplumber/python-docx原生提取与OCR视觉回退恢复的附件文本，样本内恢复率92.4%）。
- **义务提取**（§5）：继承Leahey [20]的结构化道义解析思路，扩展加入GPT-5 LLM验证阶段。管道先基于闭集强情态词词表（must, shall, may not, is/are required to, prohibited from）进行spaCy依存解析，候选通过LLM闭枚举约束验证、复合义务拆分及章节级安全网兜底（回收被动语态与定义内嵌义务）。单开发样本双盲审Cohen’s κ=1.0，precision=0.9545（95% Wilson CI [0.875, 0.984]）。
- **意见-义务匹配**（§6.2）：LLM-judged分类器判断意见是否实质上“address”特定义务并标注立场（support/oppose/modification-suggesting/none）。100对分层盲审双annotator验证显示 addressing 标签 κ=1.000，stance 标签 κ=0.953；各human rater与LLM对齐度对称且稳健（addressing κ=0.820，stance κ≈0.70）。
- **结果状态分类**（§6.3.1/§6.5）：采用sentence-transformer余弦相似度比对同一docket内提出版与最终版义务文本（阈值：≥0.95 → SURVIVED-unchanged；≥0.85 → SURVIVED-edited；0.55–0.85 → MODIFIED；<0.55或无匹配 → DROPPED；最终版独有 → NEW）。为规避EPA常见条款重编号导致的误判，放松了cfr_section共现限制（若施加该限制MODIFIED率将虚高至61.8%，现定为27.4%）。
- **分层验证架构**：明确区分“已审计负载组件”（提取、匹配、结果分类）与“仅描述性使用组件”（23-indicator修辞schema、提交者类型启发式重构），所有审计协议与κ值均透明报告，避免将未验证代理当作因果证据。

## 实验与结果
- **数据集**：EPA公开意见库（2010–2022），36个anchor规则，12,730个可分析义务（12,243个提出版 + 487个最终版独有NEW），398,509条意见-义务配对（其中4.58%获addressed=True标记）。
- **Finding 1（参与与修订弱相关）**：≥5条评论覆盖的义务修订率为69.0%，低参与义务为49.5%（
