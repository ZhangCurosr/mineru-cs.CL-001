---
title: "DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation"
source: https://arxiv.org/pdf/2608.10636v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:03:22"
field: "视觉文档检索与模型压缩"
keywords: ["Visual Document Retrieval", "Model Distillation", "Single-vector Embedding", "Multi-vector Retrieval", "Efficient Multimodal Retrieval"]
innovations: ["从单一 8B 教师对查询与文档双学生进行独立余弦对齐蒸馏，无需对比/难负样本", "非对称编码器设计：454M 文档塔 + 70M 查询塔，动态瓦片 + 全局缩略图平衡局部细节与上下文窗口", "524M 单向量系统达到 8B 教师 86.9% 质量，索引比最强子 1B 多向量基线小 15.6 倍、入库快 10 倍"]
benchmarks: ["ViDoRe v1", "ViDoRe v2", "ViDoRe v3"]
---

# 论文速读：DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation

## 一句话总结
提出 DistilVDR，一个 524M 参数的端到端单向量视觉文档检索器，通过从单一 8B 视觉语言教师模型双边余弦对齐蒸馏得到；在 ViDoRe v1+v2+v3 上达到 61.74 平均 NDCG@5（教师 86.9%），比最强子 1B 多向量基线高 8.73 点，索引存储小 15.6 倍、入库速度快 10 倍。

## 研究问题与动机
- **VDR 模型过大**：SOTA 检索器横跨 2–8B 参数（如 Qwen3-VL-Embedding-8B、Tomoro-ColQwen3-8B），索引百万文档需数十 GPU 小时、单图推理超 16 GB VRAM，难以部署。
- **现有压缩路线各有缺陷**：① 从头训练小型多向量编码器（如 colSmol-500M、ModernVBERT）虽参数小，但 per-token 存储使索引膨胀 10 倍、MaxSim 评分延迟增加 2 个数量级；② 蒸馏路线（如 NanoVDR）仅蒸馏查询编码器，文档侧仍保留 2B 教师，索引成本未降低。
- **小型单向量模型质量不足**：直接从对比学习训练的小型编码器难以学到可与多向量媲美的单向量表示（BiModernVBERT 较 ColModernVBERT 低 17.6–20.3 NDCG@5）。
- **缺乏端到端紧凑方案**：两种路线均未产出"小参数 + 单向量 + 低部署成本"的端到端系统。

## 核心贡献（创新点）
- **双边余弦对齐蒸馏框架**：查询与文档两个学生各自独立回归到冻结 8B 教师的目标嵌入空间，无需相关性标签、难负样本或对比项，两个蒸馏过程完全解耦可并行。
- **不对称编码器-only 学生架构**：匹配 VDR 的输入非对称性，文档侧 454M（InternViT-300M + ModernBERT-base）承载视觉容量，查询侧仅 70M DistilBERT，总参数 524M。
- **动态瓦片规则与视觉 token 预算控制**：按页面宽高比自适应选择网格布局（最多 6 块 448×448 瓦片 + 1 缩略图），解决低分辨率丢细节与高分辨率超上下文窗口的双重失败模式。
- **统一评测与部署分析**：复现 12 个公开检索器（250M–8.8B），在同一流水线下报告检索质量与 latency / throughput / VRAM / 索引大小，揭示单向量蒸馏在 500M 量级的质量-成本 Pareto 优势。
- **两个可部署变体**：DistilVDR-HiRes（6 瓦片，61.74 avg NDCG@5，v3 领先所有子 1B 基线 13.55 点）与 DistilVDR-Fast（2 瓦片，59.98 avg，视觉 token 预算为 HiRes 的 1/3，吞吐量 99.04 docs/s）。

## 方法详解
- **问题设定**：查询编码器 $f_q$ 与文档编码器 $f_d$ 分别将 $q$ 和 $d_i$ 映射到 $\mathbb{R}^{4096}$ 并 L2 归一化，检索分数 $s(q, d_i) = \mathbf{q}^\top \mathbf{d}_i$，文档向量在索引阶段预计算一次。
- **教师模型**：冻结的 Qwen3-VL-Embedding-8B，一次性对所有训练样本预计算 4096 维目标嵌入并缓存，训练期间不再参与前向。
- **文档编码器 $f_d$（454M）**：
  1. **动态瓦片**：基于 InternVL-V2 规则在 $\mathcal{G}=\{(p,q): n_{\min}\le pq\le n_{\max}\}$ 中选宽高比最匹配网格，切分后拼接 1 张全局缩略图；HiRes 上限 $T_{\max}=6$，Fast 上限 $T_{\max}=2$。
  2. **视觉编码**：InternViT-300M-448 将每瓦片编码为 1024 个 768 维 patch token，最坏 7×1024=7168 token。
  3. **线性投影 + ModernBERT-base（150M）**：将 visual tokens 投影到 768 维后送入双向 ModernBERT，利用其 8192 上下文窗口与 rotary/局部-全局注意力做跨瓦片上下文融合。
  4. **Mean pooling + 输出头**：均值池化后经 768→4096 线性层并 L2 归一化。
- **查询编码器 $f_q$（70M）**： teacher 训练时使用的指令前缀 $\pi$ 拼接后，经 DistilBERT-base 均值池化 + 768→4096 线性投影 + L2 归一化。
- **蒸馏损失**：
  - 文档侧：$\mathcal{L}_d = 1 - \langle f_d(d), T(d)\rangle$
  - 查询侧：$\mathcal{L}_q = 1 - \langle f_q(\pi \circ q), T(\pi \circ q)\rangle$
  - 两学生训练完全解耦，无 contrastive/排名项；教师 embedding 已蕴含相关性信号，故学生端无需额外标签。
- **训练超参**：Doc 端 AdamW，峰值 LR $1\times10^{-3}$，batch=256，3 epochs；Query 端峰值 LR $5\times10^{-4}$，batch=512，15 epochs；one-cycle schedule，3% warmup；2×H200 GPU。

## 实验与结果
- **数据集**：ViDoRe 全套 22 个数据集（v1 英文 10 个、v2 多语言 4 个、v3 专业域 8 个）；训练文档 1.20M（711K base + 454K 多域补充 + 31.7K 金融补充， perceptual hash 去重），训练查询 1.49M。
- **检索质量（NDCG@5，平均）**：
  - DistilVDR-HiRes：**61.74**（v1:82.81, v2:55.34, v3:47.07），达教师 86.9%。
  - DistilVDR-Fast：**59.98**（v1:81.34, v2:54.95, v3:43.66）。
  - 相较最强子 1B 基线 colSmol-500M（53.01），HiRes 高 **8.73** 点，Fast 高 **6.97** 点；在 v3 上 HiRes 领先次优子 1B 模型（SauerkrautLM-ColLFM2 33.19）达 **13.55** 点。
  - 略低于 4–8B 多向量（差距 6.95–9.80 点），但远超同参数单向量（DSE-Qwen2 2.2B 为 60.71）。
- **缺口分解**（Table 2）：T×T=71.05 → T×S=65.02（-6.03）→ S×T=66.36（-4.69 vs T×T）→ S×S=61.74，文档侧误差更大，两边误差非线性叠加。
- **效率（单 H200，B=8，bf16）**：
  - Fast：99.04 docs/s、2.10 GB 峰值 VRAM、查询 3.4 ms、索引 16.4 GB/M docs、10K 评分 9.6 ms。
  - HiRes：36.82 docs/s、3.07 GB、同索引/评分延迟。
  - 相较最强子 1B 多向量 baseline（colSmol-500M 256 GB 索引、1161 ms 评分），存储小 **15.6 倍**、评分快约 **100 倍**、索引吞吐快一个数量级。
- **教师预计算成本**：1.20M 图 + 1.49M 查询共 99.5 H200-GPU-hours，仅跑一次，后续所有实验复用缓存。

## 相关工作脉络
- **密集文本检索蒸馏**（Hofstätter et al., 2021; Chen et al., 2024a）：交叉编码器→双编码器蒸馏范式；本文借鉴但其目标是几何 embedding 而非标量相关性分数，且非对称跨越模态而非模型类别。
- **多向量视觉文档检索**（ColPali / ModernVBERT / colSmol 系列）：以 Per-token 存储换取小型编码器性能；本文放弃 late interaction，以单向量 + 蒸馏在 500M 量级超越同规模多向量方案。
- **OCR-free 文档理解**（Donut / Pix2Struct / LayoutLM）：共享"直接输入图像避免 OCR 级联误差"的前提，但面向生成/抽取任务；本文输出单点稠密向量，兼容标准 dense-retrieval 基础设施。
- **通用多模态嵌入器**（VLM2Vec 等）：追求跨模态广度，参数量仍在十亿级；本文以固定输入形态（text query × image doc）换取 sub-1B 深度优化。
- **NanoVDR**（Liu et al., 2026）：首个 VDR 蒸馏工作，仅蒸馏 70M 查询端，文档侧仍用 2B 教师；本文将其扩展为端到端双边蒸馏。

## 局限性与未来方向
- **缺少 from-scratch 对照**：未训练同架构 524M 模型用对比学习 + 难负样本从零训练，无法严格剥离"蒸馏本身"的贡献。
- **单一教师**：仅用 Qwen3-VL-Embedding-8B，未扫描教师规模/家族对质量的影响（虽缓存策略使扫描成本低）。
- **与 4–8B 多向量仍有 7–10 点差距**：最大缺口在 v3，难以在当前实验设置下解耦 late interaction 与模型容量。
- **学生上界受教师限制**：任何一侧替换为学生不会优于教师侧（Table 2），要突破天花板需蒸馏更强信号（如 cross-encoder 相关性分数或多向量教师）。
- **两学生独立训练**：未利用教师隐式携带的查询-文档跨模态耦合，联合蒸馏是自然延伸。
- **瓦片规则静态**：$T_{\max}$ 在发布时固定，未按页面内容复杂度自适应分配 token 预算。
- **查询塔仅限拉丁脚本**：训练覆盖英语 + 5 种拉丁语系翻译，中文/日文/阿拉伯文及图像条件查询不在范围内。
- **压缩技术未探索**：仅评估 Matryoshka 截断到 768 维（-3.19 avg NDCG@5，索引 3.07 GB/M），标量/乘积量化与二值哈希未测试，当前 16.4 GB/M 是上界。

## 研究启发与可借鉴点
- **缓存教师 embedding 的训练范式**：教师只跑一次，学生训练完全脱离大模型，大幅降低蒸馏阶段硬件门槛，可推广至其他多模态蒸馏场景。
- **非对称编码器设计**：按输入模态难度分配容量（视觉/长上下文侧 454M、纯文本查询侧 70M），在保持检索质量的同时压缩总参数，值得迁移至图文/表格检索。
- **动态瓦片 + 全局缩略图的双层视觉表征**：兼顾局部细节与全局布局，并以固定上下文窗口上限规避截断，可直接复用于任意文档图像编码器。
- **纯点 Wise 余弦对齐损失取代对比/难负样本**：依赖教师 embedding 空间隐含的相关性结构，简化训练管道并保证两端完全解耦并行，适用于任何已有强教师 embedding 的任务。
- **统一多基线 + 多维部署指标评测协议**：在同一流水线下对比质量、throughput、VRAM、索引大小、评分延迟，为后续工作提供可复用的 benchmark 模板。

## 关键术语表
- **Visual Document Retrieval (VDR)**：给定文本查询，直接从文档页面图像库中检索相关页的检索任务，避免 OCR 流水线引入的误差累积。
- **Single-vector vs. Multi-vector retrieval**：前者每页输出一个稠密向量、点积评分；后者每页输出多个 token 级向量、通过 MaxSim 后期交互评分。
- **Dual-student distillation**：查询与文档两个学生编码器分别独立回归同一冻结大教师的 embedding 空间，无需跨端联合训练。
- **Cosine alignment loss**：$\mathcal{L}=1-\langle f(x), T(x)\rangle$，衡量学生输出与教师目标在 L2 归一化后的余弦距离，本工作唯一的学生训练目标。
- **Dynamic tiling**：按页面宽高比在候选网格集合中选最优布局，将图像切分为固定大小瓦片并拼接全局缩略图，以适配视觉编码器的 native 分辨率与文本骨干的上下文窗口。
- **Late interaction (MaxSim)**：多向量检索中查询 token 与文档 token 做逐对点积并取每查询 token 的最大值再求和的评分函数。
- **Matryoshka representation**：教师使用嵌套表示学习，任意前缀子向量本身即为有效 embedding，允许安全截断维度以压缩索引。
- **One-cycle schedule**：学习率先线性上升后下降的单周期调度策略，本工作用于两个学生的训练。

## 可复现要素
- **数据集**：ViDoRe v1/v2/v3 公开评测集；训练文档 1.20M（VisRAG、ColPali train、VDR-Multilingual、Racineai 14 个多域源、Sujet-Finance-Vision-10k 与 FinHNQue 金融源）、训练查询 1.49M（NanoVDR 公开集 + MarianMT 5 语种翻译）；所有来源均为公开许可。
- **代码**：https://github.com/Ryenhails/NanoVDR（公开）。
- **权重**：教师 Qwen3-VL-Embedding-8B 官方发布；学生权重随代码开源（论文声明代码可用，权重路径见仓库）。
- **关键超参**：AdamW；Doc LR peak $1\times10^{-3}$、batch 256、3 epochs；Query LR peak $5\times10^{-4}$、batch 512、15 epochs；one-cycle、3% warmup；HiRes $T_{\max}=6$、Fast $T_{\max}=2$、tile 448×448、1 缩略图；输出维度 4096（默认）/ 768（压缩选项）。
- **硬件**：教师预计算 3× 分片 H200 共 99.5 GPU-hours；学生训练 2×H200。
