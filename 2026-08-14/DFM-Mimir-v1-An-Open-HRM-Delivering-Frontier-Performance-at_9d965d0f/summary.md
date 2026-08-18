---
title: "DFM-Mimir-v1-An-Open-HRM-Delivering-Frontier-Performance-at"
source: https://arxiv.org/pdf/2608.13517v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:56:58"
---

# 论文速读：DFM-Mimir-v1-An-Open-HRM-Delivering-Frontier-Performance-at-1B-Parameters-Using-Only-Permissible-Post-Training-Data

## 一句话总结
本文提出 **Mimir v1**，一个仅使用合规（permissible）数据从头训练的 10 亿参数分层推理模型（HRM），在英语与丹麦语任务上达到前沿水平，并为丹麦语多项基准创下新 SOTA，证明了在严格数据合规约束下小参数模型仍可具备强竞争力。

## 研究问题与动机
1. **合规数据壁垒**：当前 LLM 开发高度依赖海量非合规数据（含版权争议或个人隐私内容），对坚持开源与伦理数据的学术/国家项目构成极高门槛。
2. **低资源语言数据匮乏**：以丹麦语为例，高质量公开语料有限，传统万亿级预训练路线难以落地，亟需轻量、合规的基座模型。
3. **架构效率待验证**：HRM-Text 架构虽被提出，但在严格合规数据+小参数规模下的实际表现、以及与主流开源基座的对比尚不明确。
4. **评测对齐偏差**：原 Sapient 数据集合以多选题分类为主，与当前主流 exact-match 评测（GSM8K、MATH、DROP）目标错位，需重构数据配方。

## 核心贡献（创新点）
1. **首个全合规 1B HRM 基座模型**：Mimir v1 仅用可公开/协议授权范围内使用数据从头训练，性能匹敌 Qwen 3.5 4B 与 Gemma 4 E2B 等更大模型；与已有工作本质区别在于首次将 HRM 架构与严格合规数据策略结合，突破“大参数+大语料”的单调假设。
2. **合成移植数据集（Transplant Datasets）策略**：使用 Gemma4 31B 生成并审计替代原 Sapient 集合中的非合规英语指令数据；与已有工作区别在于不依赖数据清洗/过滤，而是主动合成以保持任务分布同时满足版权与隐私要求。
3. **从多选题向自由生成/精确匹配的配方迁移**：将训练数据重心转向需 exact-match 评分的开放式任务；与已有工作区别在于明确针对 GSM8K/MATH/DROP 等评测优化数据构成，而非沿用传统 MMLU 类选择题主导的 SFT 语料。
4. **低资源语言合规基座标杆**：在丹麦语 10 项基准上全面领先，树立小语种合规训练的可复用范式；与已有工作区别在于验证了 HRM 在资源受限语言中的高效适应性。

## 方法详解
- **架构配置**：采用 HRM-Text，隐藏维度 1536，32 层（half layers），每层 12 个注意力头，FFN 扩展因子 4。层级推理设置为 2 个 H-cycle 与 3 个 L-cycle，截断反向传播（BPTT）最大步数为 5，warmup 比例 0.2。
- **位置编码与归一化**：使用 RoPE（$\theta = 10000$），Pre-norm 层归一化（$\epsilon = 10^{-6}$）。
- **分词器**：弃用原 HRM-Text 自定义分词器，改用 **Gemma-4 tokenizer** 从头训练，以更好适配现代对话模板。
- **训练设置**：FSDP 分布式训练，bfloat16 计算 + fp32 聚合精度；AdamW 优化器（峰值 lr=$3\times10^{-4}$，2000 步线性 warmup 后保持恒定），weight decay=0.1，EMA decay=0.9999。全局 batch size=262,144 tokens，梯度累积 2，8×NVIDIA B200 GPU，平均步时 <1.1s，总步数 1.65M（不足 3 周）。
- **数据配方**：共 161 个数据集，每 epoch 约 70.48B tokens。按形态分为 7 类：Reformatted（65.96%）、Curated+reformatted（16.91%）、Synthetic+audited（11.08%）、Tool-call formatted（2.65%）、Translated+audited（2.26%）、Agreement-supplied（0.95%）、Derived task（0.18%）。
- **任务形式重构**：原 Sapient 数据以 Flan/Platypus/Tasksource 的多选题为主，本文通过合成移植将其重构为开放生成/答案生成任务，使模型训练目标与 exact-match 评测对齐。

## 实验与结果
- **评测设置**：20 个基准（英语 7、数学与代码 3、丹麦语 10），temperature=0 greedy 解码，非 MCQ 任务 max_tokens=2048，MCQ 任务 max_tokens=1。
- **英语**：Mimir 1B 平均 **69.0**，略低于 Qwen 3.5 4B（69.3），显著领先 HRM-Text 1B（66.1）与 Qwen 3.5 2B（58.2）。在 BoolQ、Winogrande、DROP 上均获同档最佳。
- **数学与代码**：平均 **64.1**，较 HRM-Text 1B（46.9）提升 **+36.7%**；超越 Qwen 3.5 2B（59.0）。GSM8K 达 89.9（1B 档最佳），HumanEval 达 56.7（HRM-Text 为 0.0）。
- **丹麦语**：平均 **56.8**，创下丹麦语任务新 SOTA，全面碾压同参数基线；DaLA（96.1）、GEC（85.6）、WikiQA（66.8）等任务大幅领先。
- **核心结论**：在严格合规数据约束下，1B 参数 HRM 仍能匹配甚至超越 2–5B 级主流开源模型的多领域基线水平。

## 相关工作脉络
1. **HRM-Text [Wang et al., 2026]**：本文架构基础；原工作使用了部分非 DFM 合规数据，本文聚焦合规适配、丹麦语扩展与合成数据替代。
2. **Sapient 混合集合（Flan/Platypus/Tasksource）**：传统指令微调标配，但包含非合规数据且以多选题为主；本文
