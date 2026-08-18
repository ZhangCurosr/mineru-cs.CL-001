---
title: "DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation"
source: https://arxiv.org/pdf/2608.10636v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:02:32"
---

# 论文速读：DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation

## 一句话总结
本文提出 DistilVDR，一个仅 524M 参数的端到端单向量视觉文档检索器，通过冻结的 8B 视觉语言教师模型对查询端与文档端进行独立余弦对齐蒸馏；该系统在 ViDoRe 全套件上达到教师 86.9% 的检索质量，同时将百万文档索引体积压缩至最强 sub-1B 多向量基线的 1/15.6，并实现约 10 倍的索引加速与毫秒级查询延迟。

## 研究问题与动机
- **多亿参数 VDR 部署成本过高**：当前 SOTA 视觉文档检索系统（如 Qwen3-VL-Embedding-8B、Tomoro-ColQwen3-8B）参数量达 2B–8.8B，单图编码需 16 GB+ 显存，百万文档索引耗时数十 GPU 小时，难以在企业级 RAG 或技术文档检索中落地。
- **从零训练的小参数多向量路线代价沉重**：colSmol、SauerkrautLM-ColLFM2、ModernVBERT 等 sub-1B 模型虽参数量小，但依赖 Late Interaction 机制，per-token 存储使索引膨胀约一个数量级，MaxSim 评分延迟较单向量高出两个数量级。
- **仅蒸馏查询侧无法实现端到端紧凑化**：已有工作（如 NanoVDR）仅将 70M 查询编码器蒸馏至 2B 教师空间，文档编码器仍保留大模型，索引百万文档仍需承载教师推理成本。
- **单向量小模型直接训练质量瓶颈明显**：BiModernVBERT 相比多向量版本在 ViDoRe v1/v2 上分别落后 17.6 / 20.3 NDCG@5，说明小参数单向量若缺乏强监督信号，难以逼近多向量检索质量。

## 核心贡献（创新点）
- **双边独立余弦对齐蒸馏框架**：查询塔与文档塔分别独立回归同一个 8B 冻结教师的缓存嵌入空间，学生训练无需相关性标签、硬负样本或对比损失项，两个蒸馏过程完全解耦且可并行化。
- **模态非对称的端到端单向量架构**：针对 VDR 文本短查询、复杂图像文档的输入不对称性，设计 454M 文档塔（InternViT-300M + ModernBERT-base）与 70M 纯文本查询塔（DistilBERT-base），在固定总参数量下优先保障视觉侧容量。
- **统一多维效率-质量评测协议**：在单一评估与性能分析管道下复现并对比了 12 个 250M–8.8B 公开检索器，首次系统性同时报告检索质量、查询延迟、索引吞吐、峰值显存、索引体积与评分延迟，消除单一指标偏差。
- **仅靠视觉瓦片预算即衍生双部署变体**：HiRes（6瓦片+缩略图）与 Fast（2瓦片+缩略图）共享全部编码器与训练流程，仅凭 $T_{\max}$ 差异在质量与吞吐间提供即插即用的部署选项，HiRes 以 61.74 avg NDCG@5 领跑所有 sub-1B 基线，Fast 以 59.98 在 3× 更小额外 token 预算下仍全面超越同规模多向量模型。

## 方法详解
- **整体蒸馏范式**：冻结 8B 教师模型（Qwen3-VL-Embedding-8B）一次性预计算 1.20M 文档图像与 1.49M 查询的 4096 维 L2 归一化嵌入并缓存；两个学生网络在训练阶段完全脱离教师前向传播，仅拟合缓存目标。
- **文档编码器（454M）**：① 动态瓦片划分：依据 InternVL-V2 规则，根据页面宽高比在候选网格 $\mathcal{G}=\{(p,q): n_{\min}\le pq\le n_{\max}\}$ 中选最优布局，切分为 $T_{\max}$ 个 448×448 瓦片并附加 1 张缩略图；② InternViT-300M 将每瓦片编码为 1024 个 768 维 patch token，序列最长 7168 token；③ 线性投影后送入 ModernBERT-base（8192 token 窗口，双向注意力，旋转位置编码），对纯视觉序列进行上下文建模；④ Mean pooling → 768→4096 线性层 → L2 归一化。
- **查询编码器（70M）**：文本追加教师训练所用的指令前缀 $\pi$，经 DistilBERT-base 编码后 mean pooling，再经 768→4096 线性投影并归一化。
- **蒸馏损失函数**：
  $$\mathcal{L}_d = 1 - \langle f_d(d),\, T(d) \rangle, \qquad \mathcal{L}_q = 1 - \langle f_q(\pi\circ q),\, T(\pi\circ q) \rangle$$
  两损失互不耦合，学生端无需对比项；教师空间的语义结构已隐含相关性监督，
