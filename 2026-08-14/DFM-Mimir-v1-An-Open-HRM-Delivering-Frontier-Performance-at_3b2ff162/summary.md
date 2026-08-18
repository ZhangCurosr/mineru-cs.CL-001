---
title: "DFM-Mimir-v1-An-Open-HRM-Delivering-Frontier-Performance-at"
source: https://arxiv.org/pdf/2608.13517v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:56:57"
field: "高效语言模型训练与低资源NLP"
keywords: ["HRM", "Hierarchical Reasoning Model", "permissible data", "low-resource language", "post-training", "synthetic data", "Danish NLP", "open-source LLM"]
innovations: ["仅用可许可后训练数据从零训练的1B HRM模型实现前沿性能", "合成移植数据集替代不可许可数据的方案", "面向自由生成任务的多样化数据配比策略"]
benchmarks: ["GSM8K", "MATH", "HumanEval", "MMLU", "DROP", "BoolQ", "DaLA", "WikiQA"]
---

# 论文速读：DFM-Mimir-v1-An-Open-HRM-Delivering-Frontier-Performance-at

## 一句话总结
本文提出 Mimir v1，一个基于分层推理模型（HRM-Text）架构的 1B 参数开源语言模型，**仅使用可许可（permissible）的后训练数据进行从零训练**，在英语和丹麦语任务上实现了前沿级性能，并在丹麦语 benchmarks 上创下新的 SOTA。

## 研究问题与动机
- **现有 LLM 开发的"单一体配方"问题**：当前大模型训练依赖海量、多阶段流水线和非许可数据（含个人信息、版权侵权），对开源和伦理导向的研究者形成极高门槛。
- **低资源语言的合规困境**：以丹麦语为例，高质量可许可数据池有限，从头预训练可行模型几乎不可能，难以提供完全合规的 base model 作为后训练基础。
- **HRM 架构的潜力尚未被充分利用**：分层推理模型允许将训练重心放在后训练数据上，适用于数据受限但需要高性能的场景。
- **数据合规与性能之间的平衡**：需要证明即使排除不可许可数据，仍能训练出具有竞争力的模型。

## 核心贡献（创新点）
1. **提出 Mimir v1，首个仅用可许可后训练数据训练的 1B HRM 模型**：与传统需海量预训练数据的做法不同，Mimir 从零训练，完全规避个人信息和版权侵权数据。
2. **设计"移植数据集"（transplant datasets）合成方案**：用 Gemma-4 31B 生成并审核的合成数据替代原始 Sapient 集合中不可许可的英文指令数据，实现同等或更优性能。
3. **构建面向生成式任务的多样化数据配比策略**：将训练重心从多项选择题（multiple-choice）转向自由生成（free-form generation），使模型更好地适配 exact-match 评测标准。
4. **实现丹麦语任务的 SOTA 并显著缩小与更大模型的差距**：Mimir 1B 在丹麦语 benchmark 上全面超越所有对比模型，英语平均仅落后 Qwen 3.5 4B 0.3 分，数学代码领域仅落后 SmolLM3 3B 3.8%。

## 方法详解
- **模型架构**：采用 HRM-Text（Hierarchical Reasoning Model Text）架构，hidden size 为 1,536，每层 12 个 attention head，FFN expansion factor 为 4。分层推理配置为 2 个 H-cycles 和 3 个 L-cycles，截断 BPTT 限制为 5 步，warmup ratio 为 0.2。
- **位置编码**：使用 RoPE（Rotary Position Embedding），θ = 10,000。
- **归一化**：采用 pre-norm Layer Normalization，ε = 10⁻⁶。
- **分词器**：使用 Gemma-4 tokenizer（与原始 HRM-Text 的自定义分词器不同）。
- **训练数据**：共 161 个数据集，每 epoch 约 70.5B tokens，涵盖 8 类功能：丹麦语指令&知识（22.07%）、英语指令（19.26%）、Sapient 混合（17.02%）、数学&推理（14.76%）、合成数据（10.00%）、agent&工具使用（9.46%）、机器翻译（4.96%）、科学&摘要（2.47%）。
- **语言分布**：英语 68.62%，丹麦语 24.74%，双语 da+en 6.54%，其他 0.20%。
- **数据处理七种形式**：Reformatted（65.96%）、Curated + reformatted（16.91%）、Synthetic + audited（11.08%）、Tool-call formatted（2.65%）、Translated + audited（2.26%）、Agreement-supplied（0.95%）、Derived task（0.18%）。
- **训练配置**：FSDP + bfloat16 计算 / fp32 汇聚；AdamW 优化器，峰值学习率 3×10⁻⁴，2,000 步线性 warmup，之后恒定；全局 batch size 262,144 tokens，gradient accumulation = 2，8 块 NVIDIA B200 GPU，每卡处理 4 个长度为 4096 的上下文。
- **训练时长**：1.65M 步，约 3 周，平均每步 < 1.1 秒。
- **合成数据生成**：使用 Gemma-4 31B 从 Common Pile（英语）和 Danish Dynaword（丹麦语）生成 span-filling、denoising、reordering、continuation 等任务，并经过人工/自动化审核。

## 实验与结果
- **评测基准**：20 个 benchmarks，分为英语（BoolQ、Winogrande、Hellaswag、MMLU、ARC-C、DROP、GovReport）、数学&代码（GSM8K、MATH、HumanEval）、丹麦语（Angry Tweets、DaLA、GEC、PIQA、Daisy、WikiQA、WMT、N.News、IFEval、Hellaswag-DA）三大类。
- **对比基线**：HRM-Text 1B、Qwen 3.5 0.8B/2B/4B、Gemma 3 1B、OLMo 2 1B、SmolLM3 3B、Gemma 4 E2B（5B 总参数，有效 2.3B）、Munin 系列（8-9B）。
- **主要结果**：
  - **英语平均**：Mimir 1B 得 69.0，超越 HRM-Text 1B（66.1）3 分，仅落后 Qwen 3.5 4B（69.3）0.3 分；在 BoolQ（87.8）、Winogrande（73.5）、DROP（83.1）上领先所有对比模型。
  - **数学&代码平均**：Mimir 1B 得 64.1，较 HRM-Text 1B（46.9）提升 **36.7%**；GSM8K 达 89.9（仅次于 Gemma 4 E2B 的 90.3）；HumanEval 达 56.7，优于 Qwen 3.5 2B（47.6）。
  - **丹麦语平均**：Mimir 1B 得 56.8，远超 HRM-Text 1B（21.7）和所有其他对比模型，创下**新 SOTA**；在 DaLA（96.1）、GEC（85.6）、WikiQA（66.8）等任务上全面领先。
- **评估设置**：temperature=0（greedy decoding），shuffle seed=4242，全部模型使用 vLLM-served（Mimir 用 FlashAttention），非 MCQ 任务 max_tokens=2048。

## 相关工作脉络
- **HRM-Text（Wang et al., 2026）**：本文直接基于其架构，但替换了分词器（Gemma-4）、数据配比（更注重生成式任务）和训练策略（仅用可许可数据），并扩展到丹麦语场景。
- **Sapient 混合数据集**：原始 HRM-Text 使用的 Flan/Platypus/tasksource 以多项选择题为主；本文将其转化为合成移植数据，并大幅降低其占比（从主导变为 17%），同时增加自由生成数据。
- **Qwen 3.5 系列**：作为主流开源前沿模型，本文将其作为 2B-4B 量级的对比基线，证明 1B HRM 可在特定任务上匹敌甚至超越更大传统模型。
- **Gemma 4 E2B**：作为最大对比模型（5B 总参数），本文在其 thinking 模式下展示了数学推理能力的差距，为未来工作指明方向。
- **Danish Foundation Models 项目**：本文是 DFM 项目的核心产出之一，强调在 Danish 低资源场景下构建完全合规 base model 的可行性。
- **OpenMathInstruct-2、AceReason**：作为数学&推理数据的重要来源，体现了本文对精确匹配（exact match）生成任务的重视。

## 局限性与未来方向
- **数学&代码能力仍有提升空间**：Mimir 1B 在 Math & Code 上落后 Gemma 4 E2B（75.4 vs 64.1），差距约 11 个百分点，需进一步优化。
- **助手能力有限**：尽管使用了 Gemma-4 chat template，但作为 assistant 的能力仍落后于 SOTA，缺乏强化学习（RL）优化。
- **HRM 架构的 Scaling 行为未知**：论文指出需要进一步研究 Mimir 这类 HRM 模型的可扩展性。
- **数据完全开放性受限**：部分 agreement-supplied 数据（如 DBC、Lex.dk）因许可限制无法公开分享，尚未实现 100% 开源。
- **未探索 RL 训练**：HRM 架构下的 reinforcement learning 尚属空白，是未来重要方向。

## 研究启发与可借鉴点
- **"Transplant datasets" 思路**：用大模型生成+审核的方式替代不可许可数据，为低资源/合规敏感场景提供了可复用的数据补齐方案。
- **生成式任务优先的数据配比**：将训练重心从多项选择题转向自由生成，可提升模型在 exact-match 评测上的表现，值得在数学推理、代码生成等任务中借鉴。
- **HRM 架构在小参数下的效率优势**：1B 参数即可匹敌 4B 传统模型，证明分层推理机制在计算效率上的独特价值，适合资源受限场景。
- **丹麦语场景的工程实践**：从数据处理（Dynaword、laerebogen）、评估基准到 chat template 适配的完整 pipeline，为其他低资源语言提供了可参考的工作流。
- **合成数据的质控策略**：Gemma-4 31B 生成 + 审核的 pipeline，以及不同类别的 acceptance rate 差异，为合成数据研究提供了实证参考。

## 关键术语表
- **HRM（Hierarchical Reasoning Model）**：分层推理模型，通过 H-cycles 和 L-cycles 的层次化注意力机制实现高效长程推理的模型架构。
- **Permissible Data**：可许可数据，指不含个人信息、无版权侵权、符合欧盟文本数据挖掘例外或 openly licensed 的数据。
- **Transplant Datasets**：移植数据集，用合成方法重新生成的数据，用于替代原始集合中不可许可的样本。
- **Exact Match**：精确匹配，模型输出需与参考答案完全一致才得分的评估方式，常用于数学推理和代码生成任务。
- **Free-form Generation**：自由生成，模型需自主产出完整答案而非从选项中选择，训练目标更贴近实际应用场景。
- **FSDP（Fully Sharded Data Parallelism）**：完全分片数据并行，一种分布式训练策略，将模型参数分片到多卡以节省显存。
- **RoPE（Rotary Position Embedding）**：旋转位置编码，一种将位置信息注入 token 表示的位置编码方法。
- **vLLM**：高性能 LLM 推理服务框架，支持 PagedAttention 等技术实现高效显存管理。

## 可复现要素
- **数据集**：161 个数据集大部分公开于 HuggingFace Hub；部分 agreement-supplied 数据（DBC、Lex.dk）因许可限制不可公开。
- **代码**：训练框架基于 Sapient 开源的 HRM-Text 代码，论文声明为 openly available（脚注 4 指向 GitHub）。
- **模型权重**：已上传至 Hugging Face Hub：https://huggingface.co/danish-foundation-models/DFM-Mimir
- **关键超参**：hidden_size=1536，layers=32，attn_heads=12，expansion=4，H_cycles=2，L_cycles=3，BP_max_steps=5，LR=3×10⁻⁴，global_batch=262144，tokenizer=Gemma-4。
