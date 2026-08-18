---
title: "InSight-doc: Agentic Visual Perception for Long-Document Understanding"
source: https://arxiv.org/pdf/2608.10628v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:02:39"
---

# 论文速读：InSight-doc: Agentic Visual Perception for Long-Document Understanding

## 一句话总结
论文提出 InSight-doc，一个端到端、检索器无关的智能体视觉感知框架，将视觉分辨率作为推理时的自适应资源：从低分辨率全局视图出发，在多轮 chain-of-thought 中动态发出放大工具调用获取子页面级高分辨率证据，在长文档 VQA 上较基线提升 4.3–16.4 准确率点、减少 40%+ 幻觉与 41%–68% 推理延迟。

## 研究问题与动机
1. 长文档理解中，端到端视觉方法需将所有高分辨率页面编码为视觉 token，导致 O(N²) 注意力开销与高昂推理延迟；同时面临 context rot 问题——模型性能随上下文变长而显著下降。
2. 传统基于解析的 pipeline（布局检测→OCR→阅读顺序重建）在视觉复杂文档上易产生级联错误，泛化性受限。
3. 现有由粗到细方法（如 Doc-V⋆）严重依赖外部检索器（94–99.8% 轨迹调用检索），引入额外索引/检索开销与检索错误传播风险；且多为页面级选择，缺乏子页面区域的细粒度定位。
4. 现有视觉搜索工作主要在单张自然图像上评估，多页文档场景中跨页、分散证据的动态视觉感知与多跳推理仍缺乏充分探索。

## 核心贡献（创新点）
1. **提出端到端检索器无关的智能体视觉感知框架 InSight-doc**：与 Doc-V⋆ 等依赖外部检索器方法不同，InSight-doc 完全内化视觉搜索能力，在多轮推理中自主决定何时放大与放大何处。
2. **设计区域级（region-level）粗到细视觉证据获取机制**：与现有方法仅选择整页不同，InSight-doc 支持子页面 bounding box 裁剪与放大，在统一多模态 chain-of-thought 中实现更精细的视觉定位与 token 效率。
3. **构建大规模主动感知训练语料**：包含 17.9K 高质量 SFT 轨迹（14,216 可问答 + 3,697 不可问答）与 19.2K 硬 RL 示例，覆盖多源、多页、多跳文档 QA 场景，支持模型学习放大策略。
4. **提出 SFT+RL 两阶段训练范式**：先用高质量轨迹做模仿学习建立基础放大行为，再通过 GRPO 强化学习进一步优化轨迹质量（提升证据框命中率、减少冗余/卡死轨迹），显著提升准确率-效率 Pareto 前沿。

## 方法详解
1. **初始化与低分辨率输入**：将原始高分辨率文档页面 $\{I_k\}$ 按缩放因子 $r \leq 1$ 降采样得到初始视觉上下文 $\tilde{I}_k^{(0)} = \text{resize}(I_k, r \cdot \text{size}(I_k))$，大幅压缩初始视觉 token 开销。
2. **多轮交替推理-动作循环**：模型在每一轮生成思维链后，可选择发出 zoom-in 工具调用 $\text{zoom\_in}(k, d, b \mid \mathcal{I}_{ctx}^{(t-1)})$，其中 $k$ 为目标图像索引，$d$ 为自然语言区域描述，$b$ 为边界框；若调用有效，从对应高分辨率源图 $I_{s(k)}$ 裁剪区域并二次缩放得到 $\tilde{I}_{crop}^{(t)}$，追加到视觉上下文 $\mathcal{I}_{ctx}^{(t)}$。
3. **递归放大机制**：支持连续放大操作，每次放大因子 $c > 1$ 相对于当前低分辨率图像累计更新缩放比 $r \leftarrow c \cdot r$，裁剪区域追加规则为 $\mathcal{I}_{ctx}^{(t)} = \mathcal{I}_{ctx}^{(t-1)} \cup \{\tilde{I}_{crop}^{(t)}\}$。
4. **数据构建流程**：三阶段过滤——prior-only 过滤（20 DPI 下 Qwen3-VL-8B 已能正确回答的问题丢弃）→ zoom-free 过滤（50/70/100 DPI 下 Qwen3-VL-32B 无需放大即能正确回答的问题丢弃）→ zoom-in CoT 生成；使用 InSight-o3 双智能体设置（vReasoner 生成推理与放大请求，vSearcher 定位边界框）生成轨迹，合法轨迹归入 SFT，失败轨迹归入 RL。
5. **推理成本分析**：理论推导表明，在典型参数范围（$r \in [0.25, 0.5]$, $n(r) \in [1, 3]$）下，相对序列长度可压缩至基线的 7.8%–45.0%；在代表性参数下，两次工具调用时相对延迟上界约为基线的 32.5%。

## 实验与结果
1. **评测基准**：DUDE（平均 5.7 页）、MP-DocVQA（平均 7.0 页）、MMLongBench-Doc（平均 49.4 页，最长 468 页）、LongDocURL（平均 85.6 页，50–150 页）、MME-RealWorld-Lite、O3-Bench。所有 PDF 以 200 DPI 栅格化，降采样因子 $r \in \{0.25, 0.35, 0.5, 0.7\}$（对应 DPI 50/70/100/140）。
2. **主要结果（Table 2）**：InSight-doc-8B (SFT+RL) 在 $r=0.25$ 下平均准确率达 66.9%，较 Qwen3-VL-8B 基线提升 16.4 点；在 $r=0.5$ 下平均准确率达 72.6%，提升 4.3 点。各基准稳定增益：DUDE +17.2、MP-DVQA +18.3、MMLongBench-Doc +17.1、LongDocURL +12.8。
3. **一般高分辨率 VQA 泛化（Table 3）**：在 MME-RealWorld-Lite 上 $r=0.25$ 达 48.2%，超过 GPT-5.4-nano（40.3%）；在 O3-Bench 上 $r=0.5$ 达 43.8%，超越 GPT-5.4-nano。
4. **对比相关方法（Table 5）**：InSight-doc 是唯一同时支持 retriever-free、coarse-to-fine、iterative multi-turn、region-level 的文档推理方法，在 MMLongBench-Doc 达 57.8%、LongDocURL 达 65.6%，超越此前最强结果 15.7 和 9.3 点。
5. **幻觉抑制（Table 4）**：在不可回答问题上，$r=0.25$ 时 DUDE 的 F1 达 69.1（较基线 +24.6）、MMLongBench-Doc 达 74.4（+25.9）；$r=0.5$ 时分别达 72.4（+15.0）和 75.1（+19.2）。
6. **轨迹质量分析（Table 6）**：RL 后证据框命中率（Box coverage）在 $r=0.25$ 时达 82.3%，冗余轨迹从 14.1% 降至 5.8%，卡死轨迹从 9.7% 降至 0.1%。
7. **效率提升**：在最长文档子集上，70 DPI 下 InSight-doc 用 42.4k token 达 56.2% 准确率，而 140 DPI 基线需 136.8k token 仅达 53.2%，序列长度减少 69%，延迟降低 71%（图 7）。

## 相关工作脉络
1. **与 Doc-V⋆ 的定位差异**：Doc-V⋆ 虽同样采用粗到细交互，但 94–99.8% 轨迹依赖外部检索器 ColQwen2.5；InSight-doc 完全内化视觉搜索，且支持子页面区域级裁剪而非仅页面级检索。
2. **与视觉 RAG 方法（ColPali、VDocRAG、VRAG-RL、MoLoRAG 等）的差异**：视觉 RAG 方法依赖外部检索器进行页面/文档图像选择，检索与推理解耦；InSight-doc 在已有低分辨率全文基础上动态放大区域，检索与推理端到端联合。
3. **与迭代证据获取方法（Doc-React、MM-Doc-R1 等）的差异**：Doc-React 迭代优化检索查询，MM-Doc-R1 使用 Qwen3 + Qwen2.5-VL 混合架构；InSight-doc 是单一端到端智能体，无需多模块协作。
4. **与同类由粗到细方法（CogDoc、DocR1、DocSeeker）的差异**：这些方法主要聚焦页面级证据识别或结构化推理，缺乏多轮区域级动态视觉搜索能力。
5. **与视觉搜索工作（V*、DeepEyes、Pixel-Reasoner、Mini-o3、InSight-o3）的关系**：视觉搜索工作主要在单张自然图像上评估；InSight-doc 将其扩展到多页长文档场景，解决跨页、分散证据的挑战。

## 局限性与未来方向
1. 仅在 Qwen3-VL-8B-Instruct 上验证了 SFT+RL 方案，未测试其他近期模型或不同厂商模型，跨 backbone 泛化性有待进一步验证。
2. 仅使用二元准确率奖励的 GRPO，未尝试更先进的 RL 算法或复杂的 reward 设计，存在优化空间。
3. 数据构建高度依赖 InSight-o3 轨迹生成，质量受限于生成模型的判断能力；未来可探索更强的轨迹生成器或引入人类标注数据。
4. 当前方法针对文档 VQA 场景设计，对视频等多模态长序列场景的扩展未作系统探讨（Related Work 中提及视频理解的并行研究方向可作为未来工作）。

## 研究启发与可借鉴点
1. **分辨率作为推理时资源的思想**：将视觉分辨率视为可自适应分配的推理时间资源而非固定预处理步骤，为遥感图像分析
