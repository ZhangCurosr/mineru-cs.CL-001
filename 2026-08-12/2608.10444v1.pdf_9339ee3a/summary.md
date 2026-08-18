---
title: "From Reasoning Depth to Reasoning Breadth: Evaluating Multi-Point Associative Reasoning in Large Language Models"
source: https://arxiv.org/pdf/2608.10444v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:11:46"
---

# 论文速读：From Reasoning Depth to Reasoning Breadth: Evaluating Multi-Point Associative Reasoning in Large Language Models

## 一句话总结
本文提出双语评测基准 **MPAR-Bench**，通过受桌游 Just One 启发的多线索整合任务，首次系统性测量大模型的“推理广度”（跨语义方向的并联联想与收敛整合能力）；实验表明当前前沿模型在该维度仍明显不足，且开启 extended thinking 虽能提升标准设置下的准确率，却不能一致性地增强扰动鲁棒性，甚至因 **overthinking** 推翻正确的中间结论。

## 研究问题与动机
- 现有 LLM 评测高度聚焦“推理深度”（单条线性链路的逐步演绎与过程监督），对“推理广度”（并行探索多个语义方向并将分散、非冗余线索整合为单一答案）缺乏可靠测量工具。
- 多文档综合、跨域类比、弱证据/干扰环境下推理等实际应用均要求模型同时维持多条部分关联并在抽象层面完成收敛，而现有 benchmark 在此正交轴线上区分度极低。
- 传统联想测验（如 RAT）题目完全公开、线索固定为三个复合词，极易受预训练数据污染与记忆化，无法干净地剥离“知识检索”与“真实整合能力”。
- 需要回答：更长、更深的推理链路是否自动带来更广的整合能力？模型在信息缺失、顺序变化、干扰注入与联想距离拉长等现实扰动下是否依然稳健？

## 核心贡献（创新点）
1. **提出首个面向“推理广度”的开源双语基准 MPAR-Bench**：将多对一的非线性线索整合操作化，区别于 RAT 的三固定提示与现有桌游 benchmark 的固定词池分组/提示生成视角，聚焦开放数量自由文本线索的收敛推理。
2. **多智能体协同线索合成管线**：通过分配差异化联想角度、LLM 判官过滤、Qwen3-Embedding-8B 相似度去重与人工唯一性核查，实现线索语义低冗余与答案可判定性（92.8% 题目经 95% Wilson CI 确认唯一），显著降低记忆化风险。
3. **由粗到细的四维评估协议与四轴扰动套件**：在精确匹配之外引入 ANLS、fastText 词向量相似度与推理轨迹事实/逻辑双维验证；标准化设计线索遮挡、顺序打乱、干扰注入、多步推理四种扰动，系统揭示广度能力的稳健性边界。

## 方法详解
- **任务形式化**：给定线索集 $C = \{c_1, c_2, ..., c_n\}$，目标 $y$ 需满足每条 $c_i$ 提供独立且非冗余的语义关系；广度体现为跨语义方向的并联整合能力。
- **Just One 约束**：禁止线索包含答案本体、直接同义词/译词/谐音/形态变体与近重复，迫使生成端从间接、互补视角出题，接收端必须融合碎片信号而非模式匹配单一强信号。
- **多智能体生成与过滤**：多个 LLM agent 被分配不同联想角度迭代提出候选线索；judge agent 剔除违反 Just One 规则、完全重复/近重复或低质量项；使用 Qwen3-Embedding-8B 对线索-答案、线索-线索相似度打分（阈值 0.3–0.8），优先保留弱相关但信息互补的线索对。
- **答案唯一性与质量控制**：构造阶段判官过滤联合弱决定性组合；在 250 题随机子集上由两名 NLP 硕士独立评估，92.8% 被判定为具备唯一明确答案（95% Wilson CI）。
- **难度与扰动设计（Standard vs. Enhanced）**：
  - **Clue Masking**：随机遮蔽线索，模拟信息缺失；
  - **Order Shuffling**：打乱线索顺序，检验对输入排列的敏感性；
  - **Distractor Injection**：注入语义干扰词，测试抗噪与去虚假关联能力；
  - **Multi-step Inferring**：拉长线索与目标间的联想距离，迫使生成隐式中间概念。
- **多维度评估**：Exact-match 准确率、ANLS（$1 - d_{\mathrm{lev}}(\hat{y}, y) / \max(|\hat{y}|, |y|)$）、fastText 余弦相似度、推理轨迹验证（将解释拆为原子步骤，分别检验逻辑连贯性与事实准确性，人工-LLM 一致性达 94.7% / 98.7%）。

## 实验与结果
- **数据集与基线**：MPAR-Bench 中英各 500 题（共 1,000 题）；评测 GPT-5.2、Gemini-3.1pro/flash、Sonnet-4.5、Qwen3-max、Kimi-k2、Deepseek-v3.2、Seed-2-pro，覆盖 thinking / non-thinking 两种模式。
- **主要结果**：
  - Thinking 模式最佳：**Gemini-3.1pro**（英文 86.8%、中文 72.2%）；non-thinking 最佳为 **
