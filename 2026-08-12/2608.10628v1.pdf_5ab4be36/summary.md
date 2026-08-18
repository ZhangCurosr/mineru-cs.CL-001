---
title: "InSight-doc: Agentic Visual Perception for Long-Document Understanding"
source: https://arxiv.org/pdf/2608.10628v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:01:27"
field: "多模态长文档理解与主动视觉感知"
keywords: ["long-document understanding", "document VQA", "agentic visual perception", "coarse-to-fine", "chain-of-thought", "reinforcement learning", "context rot", "zoom-in tool"]
innovations: ["将视觉分辨率作为推理时自适应资源并提出端到端无检索的粗到细区域缩放框架", "构建含 17.9K SFT 与 19.2K RL 的主动感知训练语料并采用双智能体轨迹生成与三阶段过滤", "通过 SFT+RL 显著提升长文档 VQA 准确率并在延迟与幻觉率上获得大幅改善"]
benchmarks: ["DUDE", "MP-DocVQA", "MMLongBench-Doc", "LongDocURL", "MME-RealWorld-Lite", "O3-Bench"]
---

# 论文速读：InSight-doc: Agentic Visual Perception for Long-Document Understanding

## 一句话总结
InSight-doc 是一个端到端、无检索器的代理式视觉感知框架，将视觉分辨率作为推理时的自适应资源：从低分辨率文档概览出发，模型在多轮链式思维中动态选择性地放大高分辨率目标区域以获取证据，从而在长文档 VQA 任务上显著提升准确率的同时减少上下文长度与推理延迟，并降低幻觉率 40% 以上。

## 研究问题与动机
- **长文档理解中的上下文负担与 context rot**：多页文档通常需以高分辨率输入 MLLM 以保留视觉细节，导致 O(N²) 计算开销与 KV cache 压力，且随着 prompt 变长会出现 context rot 现象（模型性能显著下降）。
- **现有方案的局限**：全量高分辨率输入方式对超长文档可扩展性差；基于外部检索器（Visual RAG）的方法存在检索遗漏、与生成过程解耦、检索错误难以恢复等问题；粗到细（coarse-to-fine）方法多停留在页面级别召回，仍会引入大量无关页面的视觉 token。
- **主动视觉感知的缺失**：既有视觉搜索工作主要面向单图或多页中定位单一显著区域，缺乏在严格 token 预算下跨多页、跨远距离非相邻区域联合推理与多轮自校正的能力。
- **训练数据匮乏**：缺乏同时具备多源、多页、多跳、含显式缩放轨迹的文档 QA 数据，难以端到端训练具备主动感知策略的 MLLM。

## 核心贡献（创新点）
1. **提出 InSight-doc 代理式端到端粗到细框架**：与既有方法相比，本文不做外部检索也不依赖固定高分辨率输入，而是让 MLLM 在推理过程中主动裁剪与放大子页面区域，将视觉分辨率作为可自适应分配的资源。
2. **region-level 的交互式图像-文本交织推理机制**：与 Doc-V⋆ 等方法在页面级检索不同，InSight-doc 在同一 multimodal CoT 内完成区域级定位与证据集成，支持多跳查询与多轮自我纠错。
3. **构建 37,149 条主动感知训练语料**：包含 17,913 条含区域缩放轨迹的 SFT 样本与 19,236 条 hard RL 样本，覆盖可答与不可答问题，并通过三阶段过滤与 InSight-o3 双智能体轨迹生成 pipeline 保证数据质量。
4. **SFT+RL 显著提升准确性-效率 Pareto 前沿**：InSight-doc-8B（SFT+RL）在四种文档 VQA 基准上平均较 Qwen3-VL-8B 基线提升 16.4 点（r=0.25）/ 4.3 点（r=0.5），同时在最长文档子集上将延迟降低 41%–68%、幻觉率降低 40% 以上并保持准确率领先。
5. **提供形式化的推理代价分析**：给出相对序列长度与相对推理延迟的上界命题，并在代表性参数下通过数值表验证低分辨率输入配合少量 zoom-in 调用可大幅压缩 token 与时间成本。

## 方法详解
- **初始视觉上下文与缩放定义**：模型以低分辨率页面集 $\{\tilde{I}_k^{(0)}\}$ 为起点，其中 $\tilde{I}_k^{(0)} = \text{resize}(I_k, r \cdot \text{size}(I_k))$，$r \le 1$ 为初始缩放比（如 0.25、0.35、0.5 等）。
- **多轮 zoom-in 工具调用**：在任意推理步 $t$，模型可发出工具调用 $\text{zoomin}(k, d, b | \mathcal{I}_{\text{ctx}}^{(t-1)})$，指定目标图像索引 $k$、自由语言描述 $d$ 与边界框 $b$；系统从对应高分辨率源 $I_{s(k)}$ 裁剪出 $I_{\text{crop}}^{(t)}$，再按 $\tilde{I}_{\text{crop}}^{(t)} = \text{resize}(I_{\text{crop}}^{(t)}, c \cdot r \cdot \text{size}(I_{\text{crop}}^{(t)}))$ 可选地二次缩放，并将裁剪图追加到视觉上下文 $\mathcal{I}_{\text{ctx}}^{(t)}$。
- **交织式 multimodal CoT**：每轮推理产生思维片段 $\langle\text{think}\rangle$ 与工具调用 $\langle\text{tool\_call}\rangle$，裁剪结果以视觉观察形式插入上下文，直至模型决定输出最终答案或达到工具调用上限。
- **损失与训练范式**：先以 SFT 模仿高质量多跳缩放轨迹，再以 GRPO 进行 RL，仅使用二值准确率奖励；SFT 阶段对最终回答进行重写以去除原始推理噪声，RL 阶段加入加权 refill sampler 控制可答/不可答比例（86%/14%）。
- **序列长度与延迟上界**：定义 $x(r) = r^2 + \delta n(r)$、$y(r) = 1 + \lambda n(r)$ 与 $\kappa = P_0/R_0$、$\gamma = \alpha/\beta$，证明相对序列长度上界 $S_r/S_0 \le x(r) + \kappa^{-1}y(r)$，相对延迟上界 $T_r/T_0 \le w_p x(r)^2 + w_c x(r)y(r) + w_g y(r)^2$，其中权重由 $(\gamma\kappa^2, 2\kappa, 1)$ 归一化得到；在代表性参数下，$r=0.35, n=2$ 时延迟上界约为基线的 32.5%。

## 实验与结果
- **数据集与评测基准**：训练数据来自 arXiv、DUDE、DocVQA、InfographicVQA、Paper2Poster、MapTab 六种来源；评测涵盖中等文档 VQA（DUDE、MP-DocVQA）、长文档 VQA（MMLongBench-Doc、LongDocURL）与通用高分辨率 VQA（MME-RealWorld-Lite、O3-Bench），所有 PDF 以 200 DPI 栅格化为地面真值高分辨率图，并在 $r \in \{0.25, 0.35, 0.5, 0.7\}$ 下测试。
- **主要准确率结果**：InSight-doc-8B（SFT+RL）在 $r=0.25$ 时平均准确率达到 66.9%，较 Qwen3-VL-8B 基线（50.5%）提升 +16.4 点；各基准提升分别为 DUDE +17.2、MP-DVQA +18.3、MMLongBench-Doc +17.1、LongDocURL +12.8；在 $r=0.5$ 时平均 72.6%，提升 +4.3 点。SFT+RL 相比 SFT 在 $r=0.25$ 由 56.6% 提升至 66.9%，在 $r=0.5$ 由 64.6% 提升至 72.6%。
- **对比闭源模型**：在 $r=0.25$ 下，InSight-doc-8B 超越所有 GPT 变体；在 $r=0.5$ 下与 GPT 和 Gemini 模型相当。
- **通用高分辨率 VQA 泛化**：在 MME-RealWorld-Lite（$r=0.25$）达 48.2%，超过闭源模型最高 47.8%；在 O3-Bench（$r=0.5$）达 43.8%，超越 GPT-5.4-nano。
- **与已有方法的跨论文对比**：在 MMLongBench-Doc 上达 57.8%，较先前最强结果提升 15.7 点；在 LongDocURL 上达 65.6%，提升 9.3 点；是表中唯一同时具备 retriever-free、coarse-to-fine、iterative、region-level 四个特性的方法。
- **不可答题型表现**：在 DUDE（$r=0.25$）F1 达 69.1（较基线 +24.6），在 MMLongBench-Doc（$r=0.25$）F1 达 74.4（较基线 +25.9）。
- **轨迹质量与效率**：SFT 将证据盒覆盖率从 27.5% 提升至 68.1%（$r=0.25$），RL 进一步至 82.3%；冗余率从 14.1% 降至 5.8%，卡死率从 9.7% 降至 0.1%；在 70 DPI 下比 140 DPI 基线延迟降低约 54%–71%，序列长度降低约 58%–69%。

## 相关工作脉络
- **端到端文档理解**：Qwen3-VL、InternVL3 等直接输入全部高分辨率页面，保留细节但 token 开销与 context rot 严重；本文与其定位差异在于通过低分辨率初始输入与主动区域缩放实现精度-效率帕累托改进。
- **视觉检索增强生成（Visual RAG）**：VDocRAG、VRAG-RL、MoLoRAG 等依赖外部检索器选取 top-k 页面，检索与推理解耦且易受 k 选择与检索错误影响；InSight-doc 在统一 CoT 内无检索地自适应获取区域证据。
- **粗到细方法**：CogDoc、Doc-V⋆ 均采用低分辨率扫描后切换到高分辨率页面检视的策略；本文进一步将证据粒度下移到子页面区域，且 Doc-V⋆ 高度依赖 ColQwen2.5 外部检索器，本文不依赖。
- **视觉搜索与"think with images"**：DeepEyes、Pixel-Reasoner、Mini-o3、InSight-o3 等训练模型在单图内做主动缩放；本文将其扩展到跨多页长文档场景，并处理分散在多页远距离区域的复合证据。
- **视频视觉搜索**：VideoDeepResearch、LongVideo-R1、ProVCA 在时间轴上定位关键帧；本文关注空间轴上的高 DPI 页面内子区域定位，任务形态更贴近实际文档理解。
- **结构化/ agent 式文档推理**：Doc-React、MM-Doc-R1、DocSeeker、DocR1 等提供页面级或结构化推理链路；InSight-doc 的核心区别为 region-level、retriever-free、端到端训练的交替证据获取。

## 局限性与未来方向
- **仅在一个 8B 开源底座上验证**：仅在 Qwen3-VL-8B-Instruct 上完成 SFT+RL 实验，未在其他厂商或更大参数模型上进行系统性评估。
- **RL 算法与奖励设计较为基础**：仅使用二值准确率奖励与 GRPO，未探索更先进的 RL 算法、课程学习或复合奖励设计，存在改进空间。
- **跨论文对比的控制性有限**：与同类方法的比较依赖官方跨论文数字，底座、训练数据、分辨率、页面上限与评估协议均不完全一致，难以做严格 head-to-head 结论。
- **数据与场景覆盖面局限**：当前训练与评测集中于科研/金融/海报/地图等多类文档，但未系统讨论其他领域（如手写字迹、复杂表格跨页、多语言混合排版）的泛化表现。

## 研究启发与可借鉴点
- **将分辨率作为推理时自适应资源**：可将“低分辨率全局输入 + 按需局部放大”的范式迁移到视频理解、高分辨率卫星/医疗图像分析等同样面临 token 预算与 context rot 的任务。
- **双智能体轨迹生成并融合为单智能体 SFT 目标**：用 vReasoner + vSearcher 分工生成高质量多跳轨迹，再合并为扁平化 multimodal CoT，使策略模型同时学习何时、何处与如何整合证据，值得在复杂工具使用训练中复用。
- **三阶段难度过滤机制**：prior-only（去文档无关题）→ zoom-free（去低 DPI 即可解的题）→ zoom-in CoT 构造，可有效筛出真正需要主动视觉感知的样本，提高训练数据的信噪比。
- **不可答硬负样本构造**：采用实体/数值自然扰动生成最小否定样本，并通过独立验证器筛选，可显著提升模型拒绝幻觉能力与 abstain 行为校准。
- **形式化延迟上界分析辅助系统设计**：将输入缩放比、工具调用次数与序列/延迟上界建立解析关系，可为工程侧的分辨率与调用预算权衡提供可解释的设计依据。

## 关键术语表
- **Context rot**：随上下文长度增加，注意力被稀释而导致模型性能显著下降的现象。
- **Coarse-to-fine**：先从低分辨率/概要视角定位相关范围，再在高分辨率局部进行深入检查的策略。
- **Zoom-in tool call**：模型在推理过程中发出的指定目标图像索引、区域描述与边界框的工具调用，用于请求高分辨率裁剪视图。
- **Multimodal CoT**：在链式思维过程中交替包含文本推理片段、工具调用与视觉观察的多模态推理轨迹。
- **Evidence-box coverage**：裁剪区域覆盖标注证据框面积的比例，用于衡量区域定位精度。
- **GRPO（Group Relative Policy Optimization）**：本文用于 RL 训练的 group-relative 策略优化算法。
- **Retriever-free**：不依赖外部文档/页面检索器，完全由模型在统一推理回路中主动获取视觉证据的设置。
- **Weighted refill sampler**：按预设源权重进行无放回采样的 RL 数据流采样器，用于控制每批中可答/不可答与各类来源的分布。

## 可复现要素
- **数据集**：论文已公开，包含 17,913 条 SFT 轨迹与 19,236 条 RL 样本，来源涵盖 arXiv、DUDE、DocVQA、InfographicVQA、Paper2Poster、MapTab。
- **代码与模型**：代码、数据集与模型均已开源，发布于 https://github.com/m-Just/InSight-doc。
- **关键超参**：SFT 使用全参数微调，batch size 32，最大序列长度 65,536 tokens，学习率 $5 \times 10^{-6}$，余弦退火，warmup 0.05；RL 使用 GRPO，batch size 24 prompts、每 prompt 8 rollouts，学习率 $1 \times 10^{-6}$，KL 系数 0.01，工具调用上限 10 次，裁剪最大面积 $1280^2$ 像素，全页最大面积 $3500^2$ 像素。
- **基础模型**：Qwen3-VL-8B-Instruct，视觉编码器冻结。
