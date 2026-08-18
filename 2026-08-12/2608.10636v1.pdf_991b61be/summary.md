---
title: "DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation"
source: https://arxiv.org/pdf/2608.10636v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:03:19"
field: "多模态检索与模型压缩"
keywords: ["Visual Document Retrieval", "Model Distillation", "Single-vector Embedding", "Multimodal Retrieval", "Efficient Deep Learning", "ViDoRe Benchmark"]
innovations: ["双边余弦对齐蒸馏实现端到端单向量VDR压缩", "非对称编码器设计匹配图文输入异构性", "动态分块策略平衡局部细节与全局上下文"]
benchmarks: ["ViDoRe v1", "ViDoRe v2", "ViDoRe v3"]
---

# 论文速读：DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation

## 一句话总结
DistilVDR 是一个 524M 参数的端到端单向量视觉文档检索器，通过对称余弦对齐损失从单个 8B 视觉语言教师模型双边蒸馏得到，在 ViDoRe 基准上达到教师性能 86.9%，同时以 15.6 倍更小的索引规模和 10 倍更快的索引速度超越所有已复现的子 1B 多向量基线。

## 研究问题与动机
- **VDR 系统规模过大**：SOTA 检索器参数量达 2-8B，嵌入单页文档需 16GB+ GPU 显存，百万文档索引需数十 GPU 小时。
- **小参数多向量路线代价高昂**：从 scratch 训练的小编码器（如 colSmol、ModernVBERT）虽参数小，但每 token 存储使索引膨胀 10 倍，MaxSim 打分延迟高出 2 个数量级。
- **已有蒸馏工作仅蒸馏查询侧**：NanoVDR 仅将 2B 教师的查询编码器蒸馏为 70M，文档侧仍保留完整教师，部署时仍需加载大模型。
- **单向量压缩导致质量骤降**：将多向量架构改为单向量（如 BiModernVBERT）平均损失约 17-20 NDCG@5 点，说明小编码器用对比学习难以学到足够好的单向量表示。

## 核心贡献（创新点）
- **双边余弦对齐蒸馏框架**：查询和文档两侧学生均独立回归到同一 8B 教师缓存 embeddings，无需相关性标签、负采样或对比项即可实现端到端压缩。
- **非对称编码器设计匹配 VDR 输入异构性**：文档侧使用 454M 参数的视觉-文本双塔（InternViT-300M + ModernBERT-base），查询侧仅 70M DistilBERT，契合"文本查询+图像文档"的输入偏置。
- **动态分块策略解决上下文窗口与分辨率矛盾**：按页面宽高比自适应选择 tile 布局（最多 6 块 448×448），尾部拼接全局缩略图，确保最坏情况 7168 token 不超过 ModernBERT 8192 上下文限制。
- **统一复现基准与深度效率分析**：在相同评估和 profiling 管道下复现 12 个 250M-8.8B 检索器，同时报告检索质量、延迟、显存、索引大小和打分速度，为后续工作提供可对比基线。
- **双部署变体平衡质量与成本**：HiRes（6 tile+缩略图）达 61.74 NDCG@5，Fast（2 tile+缩略图）达 59.98，两者共享编码器和训练，仅视觉 token 预算不同，Fast 视觉 token 预算仅为 HiRes 的 1/3。

## 方法详解
- **问题设定**：给定文本查询 $q$ 和文档图像集合 $\{d_1, \ldots, d_N\}$，查询编码器 $f_q$ 和文档编码器 $f_d$ 分别映射到 $\mathbb{R}^{4096}$ 空间，L2 归一化后通过点积 $s(q, d_i) = \mathbf{q}^\top \mathbf{d}_i$ 计算相关度分数。
- **教师预计算**：冻结的 Qwen3-VL-Embedding-8B 对 1.20M 文档图像和 1.49M 查询各执行一次前向传播，生成 4096 维 float32 目标向量缓存至磁盘，耗时 99.5 H200-GPU 小时，学生训练期间教师不参与计算。
- **文档编码器（454M）**：
  1. **动态分块**：根据页面宽高比在候选网格集合 $\mathcal{G}=\{(p,q): 1\le pq\le 6\}$ 中选择最接近的布局，将图像Resize为 $p^*\times q^*$ 个 448×448 非重叠 tile，额外附加一个同分辨率缩略图提供全局上下文。
  2. **视觉编码**：InternViT-300M-448 将每个 tile 编码为 1024 个 768 维 patch token，全部拼接后序列长度最多 7168。
  3. **双向上下文建模**：通过线性投影将视觉 token 映射到 ModernBERT-base（150M）嵌入空间， ModernBERT 以双向注意力重编码视觉 token（无文本输入），利用其 8192 上下文窗口和旋转位置编码保留长程依赖。
  4. **均值池化+投影**：对所有 token 均值池化后，经 768→4096 线性层投影并 L2 归一化。
- **查询编码器（70M）**：
  1. 查询文本拼接教师训练时使用的指令前缀 $\pi$，使输入分布一致。
  2. DistilBERT-base 编码后对 token 表示均值池化。
  3. 768→4096 线性投影 + L2 归一化。
- **蒸馏损失**：两侧学生独立优化余弦对齐损失，无交叉项：
  $$\mathcal{L}_d = 1 - \langle f_d(d), T(d)\rangle, \quad \mathcal{L}_q = 1 - \langle f_q(\pi\circ q), T(\pi\circ q)\rangle$$
  其中 $T(\cdot)$ 为冻结教师输出，$\langle\cdot,\cdot\rangle$ 为归一化向量的点积（等价于余弦相似度）。
- **训练超参**：文档学生 AdamW，峰值 LR $1\times 10^{-3}$，effective batch 256，3 epochs，2×H200；查询学生 LR $5\times 10^{-4}$，batch 512，15 epochs，2×H200。

## 实验与结果
- **数据集**：ViDoRe 完整套件共 22 个数据集（v1: 10个英语/法语，v2: 4个多语言，v3: 8个专业领域六语言）。
- **基线**：复现 12 个公开检索器（250M-8.8B，涵盖单/多向量），包括 colSmol-500M、SauerkrautLM-ColLFM2、ColModernVBERT、SigLIP2-L、BiModernVBERT、DSE-Qwen2、Qwen3-VL-Embedding-2B/8B、ColPali v1.3、Tomoro-ColQwen3-4B/8B、ColNomic-7B。
- **检索质量**：
  - DistilVDR-HiRes 平均 NDCG@5 = **61.74**（v1: 82.81, v2: 55.34, v3: 47.07），达 8B 教师（71.05）的 **86.9%**。
  - DistilVDR-Fast 平均 NDCG@5 = **59.98**（v1: 81.34, v2: 54.95, v3: 43.66），达教师 **84.4%**。
  - 相对于最强子 1B 基线 colSmol-500M（53.01），HiRes 高出 **8.73 点**，Fast 高出 **6.97 点**。
  - 在 v3（高分辨率敏感 benchmark）上，HiRes 以 47.07 领先次优基线 ColPali v1.3（42.04）达 **13.55 点**。
  - 与 4-8B 多向量模型仍有 6.95-9.80 点差距，主要集中在 v2 和 v3。
- **效率分析**：
  - 索引存储：DistilVDR 百万文档仅需 **16.4 GB**（4096维 float32），而 sub-1B 多向量基线需 **256 GB**，4-8B 多向量需 **819 GB**，前者仅为后者的 **1/15.6**。
  - 索引速度：Fast 达 **99.04 docs/sec**，比所有多向量基线快约 **10 倍**，比 8B 教师快约 **18 倍**。
  - 打分延迟：DistilVDR 打 10K 文档仅需 **9.6 ms**，多向量 MaxSim 需 1.1-3.2 秒，快约 **100 倍**。
  - 查询延迟：DistilBERT 路径仅 **3.4 ms/query**，显著低于多数基线。
  - 峰值显存：Fast 仅 **2.10 GB**，HiRes 为 **3.07 GB**。
- **Gap 分解**：用教师/学生交叉组合测试发现，文档侧替换为学生的平均损失为 6.03 点，查询侧替换为学生的损失为 4.69 点，二者非线性叠加，残差来自两侧学生误差的交互。
- **消融实验**：
  - 无分块（单 tile）平均损失 5.12 点，主要来自 v3。
  - 训练数据从 25% 增至 100%（300K→1.20M）平均提升约 5.5 点，75% 以上开始饱和。
  - 输出维度从 4096 截断至 768（利用 Matryoshka 表示学习）索引缩小 5.3 倍（16.4→3.07 GB/百万文档），但平均损失 3.19 点、v3 损失 4.77 点。
  - 查询编码器改用 ModernBERT-base（149M）平均提升 1.04 点，但参数量 2.1 倍、延迟 5.1 倍，最终选用 DistilBERT-base 作为性价比最优。
  - 对比监督精炼（InfoNCE/KL）在一 epoch 联合训练中无正向收益，验证纯余弦对齐的充分性。

## 相关工作脉络
- **多向量小编码器路线（ColPali 系列）**：colSmol-256M/500M、SauerkrautLM-ColLFM2、ColModernVBERT 等均基于 SmolVLM/Idefics3 + ColPali 架构，参数 250-500M 但依赖 per-token 存储和 MaxSim 打分；本文定位为在相同参数预算下以单向量方式实现更高检索质量，并消除多向量部署成本。
- **单向量小编码器路线**：BiModernVBERT 采用 ModernBERT 双塔结构从 scratch 训练，虽为 encoder-only 设计且在文档检索上优于因果解码器（+10.6 NDCG@5），但单向量表示质量远低于多向量；本文指出直接蒸馏比对比预训练更能有效压缩单向量表示能力。
- **文本检索蒸馏范式**：Hofstätter et al. (2021)、M3-Embedding 等将 cross-encoder 或大 bi-encoder 的知识蒸馏至小 bi-encoder，使用 hard-negative mining 和 topic-aware sampling；本文借鉴其"重型reranker教轻型bi-encoder"的非对称思路，但扩展至跨模态（图文）且目标为几何 embed 而非标量相关性分数。
- **通用多模态 embedder**：VLM2Vec、UniIR 等大模型微调为通用多模态检索器；本文聚焦单一任务（VDR）在 sub-1B 尺度做深度优化，牺牲跨模态泛化换取极致压缩比和部署效率。
- **OCR-free 文档理解**：Donut、Pix2Struct、UDOP、LayoutLMv3 等面向文档生成/抽取任务；本文与其共享"直接处理文档图像避免 OCR 误差级联"的前提，但输出为单密集向量而非生成式表示，适配标准 dense retrieval 基础设施。
- **前序蒸馏工作 NanoVDR**：将 2B 教师蒸馏为 70M 查询编码器，证明了 distillation 在 VDR 上的可行性；本文将其扩展为双边蒸馏，首次实现文档侧也压缩至学生规模，形成真正端到端 compact 系统。

## 局限性与未来方向
- **单一教师限制**：仅使用 Qwen3-VL-Embedding-8B 作为教师，未扫过教师尺度或架构变化，无法量化弱/强教师对最终质量的影响；缓存目标机制使教师尺度 sweep 成本较低，是低成本后续实验。
- **缺乏从 scratch 对比实验**：同等 524M 架构在相同数据上用对比学习+hard-negative mining 训练的基线未执行，无法隔离"蒸馏 vs 对比预训练"的纯粹贡献。
- **学生上限受教师限制**：双边蒸馏只能复现教师 embed 空间，无法超越教师同侧性能（Table 2 验证），若要突破天花板需从更强信号源（如 cross-encoder 相关性分数或多向量教师）蒸馏。
- **查询侧语言覆盖有限**：查询编码器仅用英语及五个拉丁语系语言（经 MarianMT 翻译），不支持中文、日文、阿拉伯文等非拉丁脚本，也未支持图像条件查询。
- **分块策略内容自适应缺失**：当前 $T_{\max}$ 固定按宽高比选择，文本密集页与单图页分配相同视觉 token 预算；可探索基于页面复杂度估计的动态预算分配。
- **未探索量化压缩**：索引以 uncompressed float32 存储（16.4 GB/百万文档），标量量化、乘积量化、二值哈希等正交压缩手段未测试，实际部署足迹有待进一步优化。
- **双边蒸馏解耦的潜在收益**：两侧学生独立训练未利用查询-文档流形间的跨模态耦合信息，联合蒸馏方案可能进一步挖掘交互信号。

## 研究启发与可借鉴点
- **缓存教师 embed 的双边蒸馏范式可迁移至其他多模态检索任务**：只需一次教师前向传播缓存目标，学生训练完全解耦且无需负样本，可推广至图像-文本跨模态检索、表格检索等场景。
- **非对称参数分配策略具有通用性**：根据任务输入偏置（此处为图文不对称）将参数集中在信息瓶颈侧（文档视觉编码），轻量侧保持低延迟，适用于多模态 RAG、多模态问答等系统。
- **Matryoshka 表示学习用于维度压缩的 principled 压缩路径**：利用教师模型的嵌套表示性质直接截断输出维度而非随机投影，可在索引大小和检索质量间提供可解释的 trade-off 曲线，适合资源受限部署。
- **统一 profiling 基准的价值**：在相同硬件、batch size、flash-attention 后端下对比 throughput/VRAM/latency/index-size，避免了不同论文各自报告导致的不可比问题，建议后续工作沿用此协议。
- **余弦对齐损失替代对比损失的可行性**：在教师 embed 质量足够高的前提下，纯点 Wise 蒸馏可消除负采样和对比项的超参敏感性和训练不稳定性，为简化训练流程提供实证依据。
- **动态分块算法的工程复用**：Algorithm 1 的宽高比匹配+缩略图拼接策略兼顾局部细节与全局布局，可直接移植到其他视觉文档处理流水线中。

## 关键术语表
- **Visual Document Retrieval (VDR)**：直接从文档图像出发进行检索的任务设定，避免 OCR/布局提取流水线引入的误差级联，保留表格、图表和版面结构信息。
- **Single-vector vs Multi-vector retrieval**：单向量每文档输出一个密集向量，索引紧凑、打分快；多向量每 token 输出一个向量，通过 Late Interaction（如 MaxSim）打分，质量更高但存储和延迟开销大一个数量级。
- **Cosine Alignment Distillation**：学生网络输出与教师网络输出之间的余弦相似度最大化（等价于最小化 $1-\langle\cdot,\cdot\rangle$），无需负样本或相关性标签的点对点蒸馏目标。
- **Dynamic Tiling**：根据文档图像宽高比自适应选择 tile 网格布局（$p\times q$），将页面切分为多个 448×448 子图，并在末尾附加全局缩略图，平衡局部细节与整体布局。
- **Matryoshka Representation Learning**：高维嵌入的前缀子向量本身也是有效表示，允许在不重新训练的情况下截断维度以压缩存储，本文用于将 4096 维输出压缩至 768 维。
- **Late Interaction / MaxSim**：多向量检索的打分方式，对查询和文档的所有 token 向量对计算相似度后取每查询 token 的最大值再求和，计算复杂度远高于单向量点积。
- **ViDoRe Benchmark**：Visual Document Retrieval 评测套件，包含 v1（10个数据集，英法）、v2（4个，多语言）、v3（8个，六语言专业领域）三阶段基准，以 NDCG@5 为主要指标。
- **Encoder-only 架构**：仅使用双向编码器（如 ModernBERT）而非因果解码器处理视觉 token，利用双向注意力建模全局依赖，在文档检索任务上已被证明优于等价规模的 causal decoder。

## 可复现要素
- **数据集**：训练数据为 1.20M 文档图像（VisRAG-Synthetic/InDomain、ColPali train set、VDR-Multilingual、racineai/VDR_MEGA_2 14个子源去重后 454K、金融补充 31.7K）和 1.49M 查询（NanoVDR 训练集）；评测基准 ViDoRe v1/v2/v3。代码仓库：https://github.com/Ryenhails/NanoVDR（论文声明开源）。
- **代码/权重**：论文声明代码已在 GitHub 开源；教师模型 Qwen3-VL-Embedding-8B 为公开权重；学生模型权重未在正文明确声明开源状态，需查看 GitHub 仓库确认。
- **关键超参**：文档学生 LR $1\times 10^{-3}$、batch 256、3 epochs；查询学生 LR $5\times 10^{-4}$、batch 512、15 epochs；两者均为 AdamW optimizer、one-cycle schedule（3% warmup）、2×H200 GPU；Fast 变体 $T_{\max}=2$，HiRes 变体 $T_{\max}=6$，tile 尺寸 448，ModernBERT 上下文窗口 8192。
