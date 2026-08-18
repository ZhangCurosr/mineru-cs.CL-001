---
title: "InSight-doc: Agentic Visual Perception for Long-Document Understanding"
source: https://arxiv.org/pdf/2608.10628v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:01:38"
field: "多模态长文档理解"
keywords: ["long-document VQA", "agentic visual perception", "zoom-in reasoning", "coarse-to-fine", "context rot mitigation", "vision-language model", "active perception", "SFT+RL"]
innovations: ["将视觉分辨率作为推理时动态可调用的自适应资源，实现端到端无检索器的多轮主动感知", "在 region-level（bbox 级）进行子页面裁剪与证据聚合，而非仅停留在整页级", "构造 37K+ 高质量主动感知训练语料并引入加权 refill RL sampler，显著提升轨迹效率与幻觉抑制"]
benchmarks: ["DUDE", "MP-DocVQA", "MMLongBench-Doc", "LongDocURL", "MME-RealWorld-Lite", "O3-Bench"]
---

# 论文速读：InSight-doc: Agentic Visual Perception for Long-Document Understanding

## 一句话总结
本文提出 **InSight-doc**，一种端到端、无检索器的 agentic 视觉感知框架，将视觉分辨率作为推理时可调用的自适应资源：模型从低分辨率全景出发，通过多轮"思考-缩放-观察"循环主动截取高分辨率子区域证据，从而在长文档 VQA 中实现更高准确率的同时显著降低幻觉、序列长度与推理延迟。

## 研究问题与动机
- **长文档 VQA 的上下文膨胀与 context rot 问题**：传统做法将所有高分辨率页面一次性输入 MLLM，导致 O(N²) 注意力计算与大量无关视觉 token；上下文过长还会引发 context rot（注意力稀释、性能骤降）。
- **现有视觉检索方案的脆弱性**：Visual RAG 方法（如 Doc-V\*、VDocRAG）依赖外部检索器挑选 top-k 页面，检索误差无法自纠正，且关键证据可能不在检索子集内。
- **粗到细方法的粒度局限**：既有 C2F 方法（CogDoc、Doc-V\*）仅做到页面级裁剪，仍需将整页高分辨率编码，浪费大量 token 并缺乏子页面级（region-level）定位能力。
- **训练数据缺失**：缺乏覆盖多页、多跳、含显式 zoom-in CoT 轨迹的大规模高质量训练语料，制约了 agentic 主动感知能力的涌现。

## 核心贡献（创新点）
1. **端到端 agentic 主动视觉感知框架**：首次将"分辨率"作为推理时动态可调用的资源，模型可在多轮 reasoning-action 循环中自主决定何时/何地/缩放何处，无需任何外部检索器。
2. **子页面级 region-level 证据获取**：与仅返回整页的 Doc-V\* 等差异显著——InSight-doc 通过 bbox 级裁剪实现 sub-page 级别的聚焦，在同一段 multimodal CoT 内完成多跳聚合。
3. **大规模 active-perception 训练语料（37.1K QA）**：构造含 17.9K SFT 多跳 zoom-in 轨迹 + 19.2K hard RL 示例的混合语料，覆盖可答/不可答、arXiv/多源异构文档，并引入加权 refill sampler 平衡训练流。
4. **推理效率的理论与实验双重验证**：给出序列长度与延迟相对基线的上界命题（Proposition 1/2），实验显示在 70 DPI 下以 41%–68% 延迟与 58%–69% token 数实现同等甚至更优准确率。
5. **显著的幻觉抑制**：在不可答问题上 F1 提升 15–26 个百分点，表明模型能更可靠地识别"证据不足"而非强行编造。

## 方法详解
### 3.1 框架与工具调用
- **初始化**：将 N 页文档的高分辨率图像 {I_k} 统一缩放为低分辨率输入 $\tilde{I}_k^{(0)} = \mathsf{resize}(I_k, r \cdot \mathrm{size}(I_k))$，其中 $r \le 1$ 为初始缩放因子（实验取 0.25/0.35/0.5/0.7，对应 50/70/100/140 DPI，原始为 200 DPI）。
- **Zoom-in 工具调用**：在每一步 t，模型生成 tool call $\mathsf{zoomin}(k, d, b \mid \mathcal{I}_{\mathrm{ctx}}^{(t-1)})$，其中 k 为当前视觉上下文中目标图片索引，d 为自由文本描述感兴趣区域，b 为边界框。
- **裁剪与缩放**：
  $$I_{\mathrm{crop}}^{(t)} = \mathsf{crop}(I_{s(k)}, b, r), \quad \tilde{I}_{\mathrm{crop}}^{(t)} = \mathsf{resize}(I_{\mathrm{crop}}^{(t)}, c \cdot r \cdot \mathrm{size}(I_{\mathrm{crop}}^{(t)}))$$
  其中 c > 1 为 zoom factor（实验 c=2.0），递归时更新 $r \leftarrow c \cdot r$。裁剪图追加到视觉上下文 $\mathcal{I}_{\mathrm{ctx}}^{(t)} = \mathcal{I}_{\mathrm{ctx}}^{(t-1)} \cup \{\tilde{I}_{\mathrm{crop}}^{(t)}\}$。
- **终止条件**：模型输出 `<answer>` 或达到最大 tool-use 次数（10 次）。

### 3.2 推理成本分析
- 假设单轮无缩放基线输入/输出 token 为 $P_0, R_0$，引入 n 次 zoom-in 后总长度：
  $$P_r = x(r)P_0, \quad R_r = y(r)R_0, \quad x(r) = r^2 + \delta n(r), \ y(r) = 1 + \lambda n(r)$$
- **Proposition 1**：相对序列长度 $S_r/S_0 \le x(r) + \kappa^{-1}y(r)$，在典型参数下上界为基线的 7.8%–45.0%。
- **Proposition 2**：相对延迟 $T_r/T_0 \le w_p x(r)^2 + w_c x(r)y(r) + w_g y(r)^2$，在 r=0.35、n=2 时延迟上界约为基线的 32.5%。

### 4 数据构造
- **三阶段过滤**：① prior-only filtering（20 DPI 由 Qwen3-VL-8B 可直接回答的丢弃）→ ② zoom-free filtering（50/70/100 DPI 由 Qwen3-VL-32B 无 zoom 直接回答的丢弃）→ ③ zoom-in CoT construction（用 InSight-o3 两 agent 生成轨迹，成功者入 SFT，其余入 RL）。
- **InSight-o3 两 agent**：vReasoner（GPT-5-mini）生成结构化请求 $\langle \mathrm{PageID}: \mathrm{Desc} \rangle$，vSearcher（Qwen3-VL-8B-Instruct）返回 bbox，裁剪后合并为单条 flat multimodal CoT 作为 SFT target。
- **不可答数据增强**：通过 GPT-5-nano 对可答 seed 做自然扰动（entity/number/date swap）生成 hard negatives，并用 Gemini 3.1 Flash-lite 验证。
- **RL weighted refill sampler**：有效训练流中 86% 可答 / 14% 不可答，按文档来源加权采样避免重复。

### 5 训练配置
- **SFT**：Qwen3-VL-8B-Instruct 全参微调，17.9K 样本，2 epoch，lr=5e-6，max seq 65536，vision encoder frozen。
- **RL**：GRPO 算法，19.2K prompt，8 rollouts/prompt，accuracy-only binary reward，lr=1e-6，KL coeff=0.01，top-k=20，presence penalty=1.5，tool-use limit=10。

## 实验与结果
- **评测基准**：DUDE、MP-DocVQA（medium）、MMLongBench-Doc、LongDocURL（long）；泛化评测 MME-RealWorld-Lite、O3-Bench。
- **主结果（Table 2，r=0.25，DPI 50）**：
  - InSight-doc-8B (SFT+RL) 平均 **66.9%**，相比 Qwen3-VL-8B (E2E, r=0.25) 的 **50.5%** 提升 **+16.4 points**；分别超过 GPT-5.4-nano/mini/mini（51.2/64.3/65.9）。
  - 各榜提升：DUDE +17.2、MP-DocVQA +18.3、MMLongBench-Doc +17.1、LongDocURL +12.8。
- **主结果（r=0.5，DPI 100）**：平均 **72.6%**，对比基线 +4.3 points，接近 GPT/Gemini 顶级闭源模型。
- **泛化（Table 3）**：MME-RealWorld-Lite r=0.25 达 48.2%（超过闭源模型 47.8%）；O3-Bench r=0.5 达 43.8%（超越 GPT-5.4-nano）。
- **与检索方法对比（Table 5/18）**：MMLongBench-Doc 57.8%、LongDocURL 65.6%，分别超出最强 prior +15.7 / +9.3 points；在控制 backbone 后的对照实验中亦优于 ColQwen2.5 + Qwen3-VL-8B 的 Top-K 页面检索方案。
- **序列长度与延迟（Figure 4/7）**：r=0.35 下以 66% token 减少匹配 r=0.7 基线准确率；最长文档子集延迟降至 11.2s vs. 39.3s（**-71%**），准确率还提升 3.0 points。
- **不可答问题（Table 4）**：r=0.25 下 DUDE F1=69.1（+24.6）、MMLongBench-Doc F1=74.4（+25.9），幻觉率降低 **>40%**。
- **Trajectory Quality（Table 6）**：RL 将 evidence-box coverage 提升至 82.3%，冗余率降至 5.8%，stuck 率降至 0.1%（SFT 阶段仍有 5.1%）。

## 相关工作脉络
- **End-to-end 全图输入法**（Qwen3-VL、InternVL3）：一次性编码所有高分辨率页面，token 开销大、难以扩展至超长文档；InSight-doc 改为动态按需获取，仅在必要时放大子区域。
- **Visual RAG/检索方法**（VDocRAG、ColPali、DocSeeker）：依赖外部 embedding 检索器选取 top-k 页面，检索误差不可恢复；InSight-doc 完全 end-to-end，无外部检索器。
- **Coarse-to-fine 方法**（CogDoc、Doc-V\*）：从低分辨率浏览到高分辨率页面，但仅停留在 page-level；InSight-doc 进一步到 region-level（bbox 裁剪），token 利用更高效。
- **"Think with images" / Visual Search**（DeepEyes、Pixel-Reasoner、Mini-o3、InSight-o3）：此前工作多在单张自然图像上验证 zoom/crop 能力；本文将其拓展至多页文档场景，并处理跨页、分散证据的 multi-hop 查询。
- **RL for document VQA**（VRAG-RL、MM-Doc-R1、DocR1）：部分采用多步检索/裁剪动作，但均以页面为粒度；InSight-doc 首次将 region-level zoom-in 与 GRPO RL 结合用于长文档理解。

## 局限性与未来方向
- **仅在一个 backbone 上验证**：只在 Qwen3-VL-8B-Instruct 上跑通 SFT+RL，未在其他开源/闭源模型上验证普适性。
- **RL 较为基础**：仅使用 binary accuracy reward + GRPO，未探索更精细的 reward shaping 或 advanced RL 算法（如 PPO、trajectory-level KL、curiosity 奖励等）。
- **未覆盖端到端全链路评估**：部分对比实验受限于他人未开源代码（如 CogDoc、Doc-V\*），缺少完全 controlled 的 head-to-head。
- **未来方向**：扩展至更大参数模型、引入 richer reward（如证据覆盖率、轨迹效率惩罚）、探索 video/时序场景的类似主动感知范式。

## 研究启发与可借鉴点
1. **"分辨率即推理资源" 的抽象**：将视觉分辨率建模为推理时可调用的自适应资源，这一视角可迁移到视频理解（每帧可调节清晰度）、医疗影像、卫星图像等高分辨率视觉任务。
2. **InSight-o3 两-agent 轨迹合成 → 单 agent flat CoT 的转化策略**：用 vReasoner + vSearcher 解耦推理与定位，再 merge 为单一 agent 的 tool-call 序列作为 SFT target——该模式可复用于训练其他视觉搜索 agent。
3. **三层难度过滤（prior-only → zoom-free → CoT construction）**：通过逐步升级的难度门槛筛出真正需要主动感知的样本，避免简单 OCR/ Lookup 类问题稀释训练信号，该 pipeline 可复用于构建其他 agentic 视觉任务的数据集。
4. **不可答 hard negative 的扰动构造法**：通过对可答 seed 做实体/数字/日期 swap 并经由强 MLLM 验证生成 synthetic unanswerable 数据，可迁移至任何需要抑制幻觉的 VQA 任务。
5. **Weighted Refill Sampler for RL**：在 RL 训练流中按类别/来源加权采样并独立 replenish，避免原始数据分布偏斜，是大规模 RL 训练的有效工程实践。

## 关键术语表
- **Context rot**：随着 prompt 上下文变长，模型注意力被稀释，性能显著下降的现象。
- **Coarse-to-fine（粗到细）**：先从低分辨率全局视图快速定位，再对感兴趣区域进行高分辨率精细阅读的推理范式。
- **Zoom-in tool call**：模型主动生成的调用指令，指定目标图片索引、区域描述与自然语言 bounding box，触发高分辨率裁剪。
- **Region-level grounding**：在子页面级别（bbox）而非整页级别定位证据，实现更精确的视觉聚焦。
- **Active perception**：模型在推理过程中主动选择性地获取新视觉信息的行为，类比人类阅读时的"先看概览、再跳细节"。
- **GRPO（Group Relative Policy Optimization）**：基于组内相对优势的强化学习策略优化算法，本文用于训练 zoom-in 决策策略。
- **SFT + RL pipeline**：先用高质量多跳 zoom-in CoT 做监督微调，再用 GRPO+accuracy reward 做强化学习精调的两阶段训练范式。
- **InSight-o3**：本文前身工作，解耦 vReasoner 与 vSearcher 两 agent 以实现通用视觉搜索，本文以此为基础生成训练轨迹。

## 可复现要素
- **数据集**：37.1K QA 实例（17.9K SFT + 19.2K RL），论文已公开于 https://github.com/m-Just/InSight-doc（含 Hugging Face 示例）。
- **代码/权重**：代码、训练数据与模型 checkpoint 均已开源（同 GitHub 链接）。
- **关键超参**：
  - SFT：lr=5e-6，2 epochs，batch=32，max seq=65536，vision encoder frozen。
  - RL：GRPO，lr=1e-6，800 steps，batch=24 prompts，8 rollouts/prompt，KL coeff=0.01，max prompt=24576 tokens，max response=8192 tokens，tool-use limit=10，zoom factor c=2.0，crop max area=1280² pixels。
  - 推理 resize ratios：r ∈ {0.25, 0.35, 0.5, 0.7}（对应 50/70/100/140 DPI），原始 200 DPI。
- **Judge**：GPT-5-nano 作为 answer correctness judge，经 150 例人工标注校准集校准（legacy-v2 prompt）。
