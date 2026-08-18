---
title: "InSight-doc: Agentic Visual Perception for Long-Document Understanding"
source: https://arxiv.org/pdf/2608.10628v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:32:12"
field: "多模态长文档理解"
keywords: ["long-document VQA", "agentic visual perception", "coarse-to-fine reasoning", "zoom-in tool", "context rot mitigation", "SFT+RL training"]
innovations: ["端到端无检索器的 adaptive zoom-in 智能体框架，将视觉分辨率作为推理时自适应资源", "三阶段 cascade 过滤 + 两智能体轨迹生成构建 17.9K SFT / 19.2K RL 主动感知训练语料", "在 8B 模型上实现超 GPT 级长文档 VQA 准确率同时降低 41%-68% 延迟和 40%+ 幻觉率"]
benchmarks: ["DUDE", "MP-DocVQA", "MMLongBench-Doc", "LongDocURL", "MME-RealWorld-Lite", "O3-Bench"]
---

# 论文速读：InSight-doc: Agentic Visual Perception for Long-Document Understanding

## 一句话总结
InSight-doc 提出了一种端到端、无需外部检索器的智能体视觉感知框架，将图像分辨率作为自适应推理时资源：从低分辨率全局概览出发，通过多轮 zoom-in 工具调用逐步获取高分辨率区域证据，显著提升了长文档 VQA 的准确率并降低了幻觉率、序列长度与推理延迟。

## 研究问题与动机
- **长文档理解面临 context rot 与计算开销双重瓶颈**：Transformer 的注意力机制为 O(N²)，长上下文易导致注意力稀释，且高分辨率全量输入产生大量视觉 token，推理成本高昂。
- **现有端到端方法难以扩展至超长文档**：直接输入全部高分辨率页面保留细节但 token 开销巨大，且大量页面含任务无关内容，无法有效缩放至极端长文档场景。
- **视觉检索增强方法存在解耦缺陷**：Visual RAG 类方法依赖外部检索器选取 top-k 页面，检索与推理松耦合，k 值敏感，且关键证据可能遗漏在检索子集之外。
- **已有粗到细方法仍限于页面级操作**：CogDoc、Doc-V⋆ 等仅在页面级别做检索/选取，仍需将整个页面以高分辨率编码，无法实现亚页面（sub-page）级别的精准证据定位，且 Doc-V⋆ 高度依赖外部检索器。

## 核心贡献（创新点）
1. **提出 InSight-doc 框架：将视觉分辨率视为推理时自适应资源**——与 Doc-V⋆ 等依赖外部检索器做页面级选取的方法不同，InSight-doc 完全端到端、无需任何外部检索器，直接在推理循环中进行 region-level zoom-in，实现亚页面级证据定位。
2. **设计了 zoom-in 工具调用机制并嵌入多轮 reasoning-action 循环**——模型从低分辨率全文出发，在每次推理步自主决定 zoom-in 的目标页面、区域描述和边界框，与 Doc-V⋆ 的页面抓取（page fetching）形成本质区别：InSight-doc 可在同一 multimodal CoT 中完成多跳跨页推理与自我修正。
3. **构建了大规模主动感知训练语料（17.9K SFT + 19.2K RL）**——采用三阶段过滤（prior-only / zoom-free / zoom-in CoT construction）和 InSight-o3 两智能体轨迹生成 pipeline，包含可答与不可答样本，并辅以合成难负例增强；该方法学上的数据构造策略与现有工作（如 VRAG-RL、MM-Doc-R1）的数据构建流程有本质差异。
4. **在多个基准上同时优化准确率-效率 Pareto 前沿**——InSight-doc-8B (SFT+RL) 在 r=0.25 下平均准确率达 66.9%，较 Qwen3-VL-8B 基线提升 16.4 点；在最长文档子集上将推理延迟降低 41%–68% 同时将幻觉率降低超 40%，且保持精度优势。

## 方法详解
- **框架概述**：InSight-doc 以指令微调的 MLLM（Qwen3-VL-8B-Instruct）为底座，初始视觉上下文 $\mathcal{I}_{\text{ctx}}^{(0)}$ 由所有页面经 resize 因子 $r \leq 1$ 下采样后的低分辨率图像构成：$\tilde{I}_k^{(0)} = \text{resize}(I_k, r \cdot \text{size}(I_k))$。
- **Zoom-in 工具调用**：在每一步 $t$，模型通过 CoT 决定是否需要 zoom，若需要则输出工具调用 $\text{zoom\_in}(k, d, b \mid \mathcal{I}_{\text{ctx}}^{(t-1)})$，其中 $k$ 为目标页面索引，$d$ 为自然语言区域描述，$b$ 为边界框；裁剪后图像经可选 resize：$\tilde{I}_{\text{crop}}^{(t)} = \text{resize}(I_{\text{crop}}^{(t)}, c \cdot r \cdot \text{size}(I_{\text{crop}}^{(t)}))$，其中 $c > 1$ 为 zoom 因子；裁剪图追加至视觉上下文 $\mathcal{I}_{\text{ctx}}^{(t)} = \mathcal{I}_{\text{ctx}}^{(t-1)} \cup \{\tilde{I}_{\text{crop}}^{(t)}\}$。
- **递归 zoom 的 resize 因子更新**：$r \leftarrow c \cdot r$，支持多层嵌套 zoom-in。
- **推理终止条件**：模型选择输出 `<answer>` 或达到工具调用上限（10 次）。
- **复杂度分析**：定义了相对序列长度上界 $S_r/S_0 \leq x(r) + \kappa^{-1} y(r)$（Proposition 1）和相对延迟上界 $T_r/T_0 \leq w_p x(r)^2 + w_c x(r)y(r) + w_g y(r)^2$（Proposition 2），在典型参数下（$r=0.35, n=2$）延迟上界约为基线的 32.5%。
- **数据构造 pipeline**：三阶段过滤（① prior-only：用 Qwen3-VL-8B 在 20 DPI 筛选文档无关题；② zoom-free：在 DPI∈{50,70,100} 用 Qwen3-VL-32B 筛选无需 zoom 即可回答的题；③ zoom-in CoT construction：用 InSight-o3 两智能体 pipeline 生成轨迹，正确答案的入 SFT，其余入 RL）；最终语料 37,149 条（SFT 17,913 条，RL 19,236 条），平均文档长度 18.51 页。
- **训练配置**：SFT 使用全参数微调，batch size=32，最大序列 65,536 tokens，lr=$5\times10^{-6}$；RL 使用 GRPO 算法，binary accuracy reward，8 rollouts per prompt，lr=$1\times10^{-6}$，KL 正则系数 0.01，加权随机 refill sampler（可答:不可答=86:14）。

## 实验与结果
- **评测基准**：中等文档 VQA（DUDE、MP-DocVQA，平均 5.7/7.0 页）；长文档 VQA（MMLongBench-Doc、LongDocURL，平均 49.4/85.6 页）；通用高分辨率 VQA（MME-RealWorld-Lite、O3-Bench）。
- **主要结果（Table 2）**：
  - r=0.25（DPI 50）：InSight-doc-8B (SFT+RL) 平均准确率 **66.9%**，较 Qwen3-VL-8B 基线（50.5%）提升 **+16.4 点**；在各基准分别提升 +17.2（DUDE）、+18.3（MP-DVQA）、+17.1（MMLongBench-Doc）、+12.8（LongDocURL）。
  - r=0.5（DPI 100）：平均准确率 **72.6%**，较基线（68.3%）提升 **+4.3 点**；在各基准分别提升 +5.3、+2.7、+7.2、+2.1。
  - 在 r=0.25 时超过所有 GPT 变体；在 r=0.5 时与 GPT/Gemini 竞品相当。
- **通用高分辨率 VQA（Table 3）**：在 MME-RealWorld-Lite 上 r=0.25 达 48.2%（超越 GPT-5.4-nano 的 47.8%），在 O3-Bench 上 r=0.5 达 43.8%。
- **不可答问题（Table 4）**：r=0.25 时 DUDE F1=69.1（较基线 +24.6）、MMLongBench-Doc F1=74.4（+25.9）；幻觉率降低超 40%。
- **轨迹质量（Table 6）**：RL 将证据 box 覆盖率从 SFT 的 68.1% 提升至 82.3%（r=0.25），冗余轨迹从 14.1% 降至 5.8%，卡死轨迹从 9.7% 降至 0.1%。
- **序列长度与延迟**：r=0.35 时以 66% 更少的 token 达到 70.6% 准确率（超过 r=0.7 基线的 69.4%）；最长文档子集延迟降低 71%（11.2s vs 39.3s）。

## 相关工作脉络
- **End-to-end document VQA（Qwen3-VL、InternVL3 系列）**：直接将高分辨率页面输入 MLLM，保留视觉细节但 token 开销大；InSight-doc 与其本质区别在于通过 adaptive zoom 在推理时动态获取证据而非全量处理。
- **Visual RAG（ColPali、VDocRAG、VRAG-RL、M3DocRAG 等）**：使用外部检索器选取 top-k 页面；InSight-doc 不依赖任何外部检索器，而是端到端地在学习到的 policy 中自主决策 zoom-in，避免检索误差的级联传播。
- **Coarse-to-fine 方法（CogDoc、Doc-V⋆）**：均从低分辨率概览起步，但仅定位到页面级别并需全页编码；InSight-doc 进一步细化到 sub-page region level，在同一 multimodal CoT 中完成多跳推理。
- **Doc-V⋆（Zheng et al., 2026）**：与 InSight-doc 最接近，采用 ReAct 框架的 interleaved 推理；但 Doc-V⋆ 94%–99.8% 轨迹调用外部检索器（ColQwen2.5），且证据定位在页面级；InSight-doc 完全 retriever-free 且为 region-level。
- **Visual search / "think with images"（DeepEyes、Pixel-Reasoner、Mini-o3、InSight-o3）**：将 zoom/crop 内化为 MLLM 操作；InSight-doc 将这些技术迁移至多页长文档场景，解决了跨文档、跨页、分散证据的搜索挑战。
- **Video visual search（VideoDeepResearch、LongVideo-R1、ProVCA）**：共享选择性获取稀疏证据的理念，但作用于时间轴；本文聚焦于空间轴上的跨页 sub-region 定位，证据分布在遥远且不连续的页面中。

## 局限性与未来方向
- 仅在 Qwen3-VL-8B-Instruct  backbone 上验证了 SFT+RL 方案，尚未在其他模型架构或更大参数规模（如 72B）上进行实验，泛化性有待验证。
- 未尝试更先进的 RL 算法（如 PPO、GRPO 的变体）或更精细的 reward 设计（当前仅用 binary accuracy reward），存在较大改进空间。
- 数据主要来自 arXiv 论文、DUDE、DocVQA、InfographicVQA 等来源，对财务报表、法律文档等其他视觉丰富型长文档的覆盖不足。
- 训练数据的 SFT 部分仅 17.9K 条，相对于主流 MLLM 训练数据规模较小，可能限制了模型在极端复杂场景下的泛化能力。

## 研究启发与可借鉴点
1. **将分辨率作为推理时资源的范式**：把视觉 token 数量从固定的预处理步骤解放出来，转为推理时的动态可调资源，这一思路可迁移至视频理解、多模态检索等需要权衡 fidelity 与效率的场景。
2. **三阶段 cascade 过滤 + 两智能体轨迹生成 pipeline**：prior-only → zoom-free → zoom-in CoT 的分级过滤策略能有效区分不同难度的样本并匹配最合适的训练信号（SFT vs RL），同时 InSight-o3 的 vReasoner/vSearcher 解耦设计可作为通用模板用于构建其他 agentic vision 任务的训练数据。
3. **不可答样本的合成增强策略**：基于 seed question 的自然扰动（entity/value/date swap）生成 hard negative 并经验证后加入训练，有效增强了模型对证据不足的识别能力，该策略可迁移至其他需要 abstain 行为的视觉问答任务。
4. **Weighted refill sampler 用于 RL 数据配比**：在 RL 阶段通过 YAML 权重表动态平衡 answerable/unanswerable 及不同数据源的比例，避免原始数据分布偏差影响训练，是 RL 阶段数据管理的实用技巧。
5. **理论化的延迟上界分析**（Proposition 1/2）：将序列长度和延迟表示为 resize ratio 和 tool call 次数的函数，为系统设计提供了定量指导，此类分析框架可复用于其他 adaptive-resolution 方法的效率论证。

## 关键术语表
**Context rot**：指 MLLM 性能随 prompt 长度增加而急剧下降的现象，归因于长上下文中注意力被稀释及真正长上下文训练数据的匮乏。
**Coarse-to-fine**：先从低分辨率/概览输入开始，再逐步获取高分辨率细节的视觉理解策略，模拟人类阅读习惯。
**Zoom-in tool call**：InSight-doc 中的核心工具操作，模型输出包含页面索引、区域描述和边界框的调用请求，系统据此从高分辨率源图中裁剪对应区域并追加至视觉上下文。
**Active perception corpus**：本文构建的包含 region-level zoom-in 轨迹的主动感知训练语料，含 17.9K SFT 样本和 19.2K RL 样本。
**InSight-o3**：本文使用的两智能体轨迹生成框架，含 vReasoner（负责推理和发出 zoom 请求）和 vSearcher（负责在指定页面上定位区域并返回边界框）。
**GRPO（Group Relative Policy Optimization）**：本文使用的 RL 算法，通过组内相对优势估计进行策略更新，仅需 advantage 归一化而不需 critic 网络。
**Evidence-box coverage**：衡量 crop 区域与标注证据 box 的重合程度的指标，定义为 $\max_{c,b} \text{area}(c \cap b) / \text{area}(b)$，阈值 τ=0.5。
**Retriever-free**：指完全不依赖外部检索模型（如 embedding-based retriever），模型在端到端推理过程中自主获取视觉证据。

## 可复现要素
- **数据集**：37,149 条训练数据（17,913 SFT + 19,236 RL），论文声明已开源，见 https://github.com/m-Just/InSight-doc
- **代码**：论文声明已开源（SFT 和 RL 代码基于 verl 框架）
- **模型权重**：InSight-doc-8B 模型权重已开源发布
- **关键超参**：SFT lr=$5\times10^{-6}$，batch size=32，max seq=65,536；RL lr=$1\times10^{-6}$，8 rollouts/prompt，KL coeff=0.01，tool-use limit=10；vision encoder frozen；resize ratio 测试点 r∈{0.25, 0.35, 0.5, 0.7}（对应 DPI 50/70/100/140）
