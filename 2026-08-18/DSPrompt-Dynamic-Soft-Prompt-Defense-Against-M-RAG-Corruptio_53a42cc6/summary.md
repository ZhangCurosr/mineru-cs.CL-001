---
title: "DSPrompt-Dynamic-Soft-Prompt-Defense-Against-M-RAG-Corruptio"
source: https://arxiv.org/pdf/2608.16536v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:28:27"
field: "多模态检索增强生成安全"
keywords: ["M-RAG", "adversarial defense", "soft prompt", "knowledge poisoning", "contrastive learning", "vision-language model"]
innovations: ["将 M-RAG 防御从查询时筛选重构为检索器嵌入语义重塑，一次离线部署、零查询时额外开销", "动态 min-max 对抗训练：在线持续生成硬投毒样本使防御泛化到未见攻击", "浅至深几何分配 soft prompt 容量（<1% 参数），精准修正跨模态对齐层"]
benchmarks: ["Places365 + PE-C", "ImageNet + PE-C", "WebQA + GPA", "InfoSeek + Clean-L"]
---

# 论文速读：DSPrompt-Dynamic-Soft-Prompt-Defense-Against-M-RAG-Corruption

## 一句话总结
本文针对多模态检索增强生成（M-RAG）中知识投毒攻击问题，提出了一种轻量级防御框架 DSPrompt：通过在冻结的视觉-文本双编码器各层插入可学习的软提示（soft prompts），并以动态极小极大博弈在线训练，直接重塑检索器的嵌入语义，将投毒文档挤出 top-k 排名，同时保持对干净文档的检索与生成质量。

## 研究问题与动机
1. **核心问题**：M-RAG 系统的多模态知识库易受对抗性投毒攻击——攻击者注入经过精细优化的图像-文本对，使其嵌入与合法条目高度对齐，欺骗检索器并将有害内容推入 top-k，诱导 LVLM 生成错误/有害输出。
2. **现有防御不足**：主流防御（如 RoCLIP、IRAG）在查询时依赖图像-文本一致性检测或重排序，存在三类缺陷：①推理开销随查询量和数据库规模线性增长；②通常针对固定攻击分布校准，泛化性弱；③需要额外模块或多次重排，难以部署。
3. **关键洞察**：投毒样本的高检索分来源于对抗扰动 δ 利用了编码器的局部输入敏感性，而干净样本的高分源于真实语义对齐。若能降低编码器在 δ 利用的非语义方向上的敏感性，即可在保留干净语义的同时使伪造相似度崩塌。
4. **设计动机**：与其在查询时筛选输出，不如直接在检索器层面"少 effort"地重塑嵌入空间——使用少量可学习参数 + 足够强且多样化的在线攻击器训练，可实现轻量、一次离线部署的防御。

## 核心贡献（创新点）
1. **将 M-RAG 防御重构为检索器语义重塑问题**：提出 DSPrompt，一种轻量软提示微调框架，直接修改冻结检索器的嵌入空间以压制投毒文档，同时保留干净检索行为——与已有方法本质区别在于不在查询时引入辅助检测/重排模块，而是"治本"于编码器层面。
2. **动态对抗提示学习机制**：训练过程中在线不断生成硬投毒样本（min-max 博弈），而非依赖固定预生成毒样本集，使防御泛化到未见攻击策略——相比静态对抗训练的过拟合风险，本方法学到的是可迁移的嵌入修正能力。
3. **浅至深 Prompt 分配调度**：按几何级数将更多 prompt 容量分配给深层网络层（跨模态对齐关键层），顶层聚焦对抗耦合抑制，底层保留原始表示——与均匀分配的本质区别在于顺应 ViT 层次化结构（浅层编码纹理边缘、深层负责跨模态匹配）。
4. **无需索引/流水线改动的即插即用部署**：防御后编码器可直接预计算并构建索引，推理时使用标准最近邻搜索，额外参数少于 1%，无查询时额外计算——与 RoCLIP 等需重排序或重新匹配的基线形成成本对比。

## 方法详解
**整体框架（离线训练 → 部署推理）**：
- 冻结主干 CLIP 风格双编码器（$\Phi_{\text{img}}, \Phi_{\text{txt}}$），在各层插入可学习 soft prompt token $P^{(\ell)}$，得到 $\Phi_\theta$；嵌入融合方式不变：$\Phi(d_i) = \frac{\Phi_{\text{img}}(I_i) + \Phi_{\text{txt}}(T_i)}{\|\cdot\|}$。

**三层插入机制（每层）**：
1. **插入阶段**：在 [CLS] 后立即插入 $m_\ell$ 个 prompt token；
2. **交互阶段**：augmented sequence 参与 self-attention，prompt 与原始 token 交互并注入修正信号；
3. **移除阶段**：移除 prompt token，仅保留其注入的修正信息，序列长度与主干不变。

**浅至深长度调度**：$m_\ell = r^{\lceil 3\ell/L \rceil - 1}$，$r=2$，将 L 层分为三组（浅/中/深），prompt token 数按 1:2:4 几何增长，总参数增量 < 1%。

**防御损失函数（三项组合）**：
$$\mathcal{L}(\theta) = \mathcal{L}_{q \to d}(\theta) + \lambda_{\text{sym}} \mathcal{L}_{d \to q}(\theta) + \lambda_{\text{anc}} \mathcal{L}_{\text{anc}}(\theta)$$

- **对比检索损失** $\mathcal{L}_{q \to d}$：InfoNCE，正样本为原始编码器 $\Phi_0$ 的 top-1 干净文档，负样本为其他查询的正样本 + 在线生成的 K 个投毒文档。
- **对称损失** $\mathcal{L}_{d \to q}$：反向 InfoNCE，文档作锚点、查询为正，抑制 hubness。
- **干净锚点正则** $\mathcal{L}_{\text{anc}}$：$\sum \|\Phi_\theta(x) - \Phi_0(x)\|_2^2$，防止 prompt 将干净嵌入偏离原始分布。

**动态 min-max 训练**：
- 内层（攻击者 Max）：对每个 $(q, d^+)$ 对，先在候选文本池 $\mathcal{T}$ 中做无梯度搜索选择最相似恶意文本 $T^-$（含诱导目标答案 + 误导性图像描述），再以固定 $T^-$ 用 PGD（$K_{\text{pgd}}=20$ 步，$\epsilon=0.05$，$\eta=0.005$）优化图像扰动 $\delta$，生成最强投毒样本 $\tilde{d}_\delta(q)$。
- 外层（防御者 Min）：用上述合成的 $\tilde{d}_\delta(q)$ 更新 $\theta$，最小化防御损失。
- 恶意文本构造有两种方式：Way 1（借用数据集负样本描述 + 通用拒绝式诱导语，4 模板）和 Way 2（LLM 离线生成查询相关错误答案 + 支持描述，22 模板），各占 50% 概率。
- 训练数据与测试数据完全隔离，避免数据泄露。

**部署**：训练完成后 $\theta^*$ 冻结，文档嵌入可离线预计算并建立索引；推理时标准 FAISS top-k 搜索，无任何额外开销。

## 实验与结果
**数据集**：Places365、ImageNet-1K（PE-C 攻击）、WebQA（GPA 攻击）、InfoSeek（Clean-L 攻击），投毒候选池来自 OVEN-Wiki（2M 规模）。

**基线**：No Defense（原始 CLIP）和 RoCLIP（鲁棒预训练编码器，可重排序）。

**关键结果（Table 1）**：

| 攻击 / 数据集 | 方法 | PRR@1 | PRR@3 | ASR | SUF@3 | TF |
|---|---|---|---|---|---|---|
| PE-C / Places365 | No Defense | 82.30% | 88.05% | 81.15% | — | 17.70% |
| | RoCLIP | 25.75% | 29.32% | 26.03% | 80.26% | 70.96% |
| | **DSPrompt** | **6.71%** | **6.96%** | **6.66%** | **98.84%** | **93.18%** |
| PE-C / ImageNet | DSPrompt | 4.40% | 5.40% | 4.40% | 99.23% | 95.40% |
| GPA / WebQA | DSPrompt | 0.28% | 0.56% | 0.64% | 69.54% | 80.21% |
| Clean-L / InfoSeek | DSPrompt | 18.00% | 18.00% | 12.00% | 92.88% | 70.00% |

- **最强结果**：PE-C/ImageNet 上 ASR 降至仅 4.40%（No Defense 为 62.67%）；GPA 上 ASR 降至 0.64%（No Defense 为 87%）。
- **泛化性**：GPA 和 Clean-L 在训练中未见，但仍大幅降低 ASR，证明软提示学到的是可迁移的嵌入修正能力而非记忆攻击模板。
- **效率**：DSPrompt 推理开销 1.85×（vs No Defense 1×，RoCLIP 3.08×），参数增量 < 1%。
- **抗密度攻击**：投毒文档数从 1 增至 10，DSPrompt 仍保持 PRR 和 ASR 接近零，TF 稳定。
- **骨干无关性**：在 OpenCLIP ViT-L/14 与 SigLIP 两种编码器、LLaVA-v1.6-Mistral-7B 与 Qwen-VL 两种生成器上均稳定有效。

## 相关工作脉络
1. **RoCLIP**（Yang et al., 2023）：鲁棒预训练编码器，通过图像-文本重新匹配实现重排序防御；本质是查询时辅助模块，开销大且需特定攻击假设，DSPrompt 则在编码器层面一次部署、无额外查询开销。
2. **IRAG**（Luo et al., 2026）：结合图像判别与显式匹配进行危险分离，但依赖每个查询多条冗余参考；DSPrompt 仅需一个编码器即可分离干净/投毒嵌入。
3. **PoisonedEye (PE-C)**（Zhang et al., 2025）：类目标投毒攻击，将注入图像推向目标类嵌入中心；是 DSPrompt 的主要训练攻击来源之一。
4. **MM-PoisonRAG (GPA)**（Ha et al., 2025）：全局投毒攻击，优化跨查询通用的单一毒样本；DSPrompt 在训练中未见此攻击，仍能有效防御，体现泛化优势。
5. **Poisoned-MRAG (Clean-L)**（Liu et al., 2025）：clean-label 攻击，保持图像-文本一致性同时提升可检索性；同样在测试时未见，ASR 降至 12%。
6. **RL-based/文本 RAG 防御**（如 BadRAG、TrustRAG）：文本-only 方法，无法直接推广至连续图像扰动空间；DSPrompt 原生处理多模态联合嵌入。

## 局限性与未来方向
1. **训练依赖在线攻击生成**：需额外计算开销完成内层 min-max 优化（PGD 20 步 × 每样本），训练成本高于纯监督方法；未来可探索更高效的对抗样本生成策略。
2. **仅针对嵌入层防御**：未覆盖生成阶段可能的后门触发或其他注入路径；防御纵深可延伸至 LVLM 输出验证。
3. **SUF@3 在 GPA 上偏低（69.54%）**：全局投毒改变了嵌入空间较大范围，导致干净检索质量有一定下降；如何在全局攻击下更好保持语义保真度有待改进。
4. **评估规模受限**：当前基准为 Places365（65 类）、ImageNet、WebQA、InfoSeek，未在更大规模工业知识库（如 million-scale Wiki）上验证。
5. **未讨论防御对抗自适应攻击**：若攻击者在获知防御机制后重新设计攻击（defense-aware attack），效果需进一步验证。

## 研究启发与可借鉴点
1. **Prompt-tuning 用于安全防御的新范式**：将 prompt tuning 从"性能提升工具"拓展为"鲁棒性修复工具"的思路可迁移至其他 AI 安全场景（如视觉分类器的对抗鲁棒性提升）。
2. **浅至深容量调度策略**：依据网络层级语义深度分配资源的思想，可推广到其他多模态模型的微调/加固任务中（如根据层重要性分配 LoRA rank）。
3. **动态 min-max + 离线部署的设计模式**：训练时"持续生成最难对抗样本"、部署时"零额外开销"的范式，值得借鉴于其他实时性敏感的安全应用。
4. **Clean Anchor 正则化保护语义分布**：在对抗训练中加入嵌入空间锚定损失，以"不偏离原始分布"为约束来保持 utility，是可复用的通用技巧。
5. **与团队方向结合机会**：本方法的动态对抗训练框架可与团队关注的 RAG 可解释性/可信生成结合，例如将 prompt 修正信号可视化为嵌入空间中的语义纠正路径，或扩展至多跳 RAG 场景。

## 关键术语表
**M-RAG（Multimodal Retrieval-Augmented Generation）**：将视觉-文本检索与大型视觉语言模型生成结合的框架，检索结果直接影响生成质量。
**Soft Prompt（软提示）**：可学习的连续向量 token，插入到预训练编码器各层中，以最小改动方式调整模型行为。
**PRR@k（Poisoned Retrieval Rate）**：top-k 检索结果中包含至少一个投毒文档的查询比例，衡量检索侧安全评分，越低越好。
**ASR（Attack Success Rate）**：LVLM 输出匹配攻击者目标响应的查询比例，衡量端到端攻击成功率。
**SUF@k（Semantic Utility Fidelity）**： defended 检索结果与 raw 检索结果在原始编码器空间中的相关度比值，衡量检索质量保留程度。
**TF（Task Fidelity）**：防御后生成答案与干净基准答案一致的比例，衡量生成质量保留程度。
**Min-Max 对抗训练**：内层最大化（攻击者生成最强投毒样本）、外层最小化（防御者更新 prompt 压制投毒）的博弈优化过程。
**Hubness**：高维空间中少数样本成为大量其他样本最近邻的现象，对称 InfoNCE 损失可有效抑制。

## 可复现要素
- **数据集**：Places365、ImageNet-1K、WebQA、InfoSeek、OVEN-Wiki（2M）、M-BEIR（训练查询采样）——原始数据集公开，攻击投毒样本沿用各论文原始协议生成。
- **代码/权重**：论文未提及开源声明。
- **关键超参**：$r=2$，$\epsilon=0.05$，$\eta=0.005$，$K_{\text{pgd}}=20$，$K=2$（每查询投毒数），$\tau=0.07$，$\lambda_{\text{sym}}=0.5$，$\lambda_{\text{anc}}=1.0$，batch size=32，learning rate=$5\times10^{-5}$，weight decay=0.05，cosine schedule，40 epochs；FAISS top-5 搜索。
- **基础模型**：OpenCLIP ViT-L/14（主实验）、SigLIP-SO400M（验证）、LLaVA-v1.6-Mistral-7B / Qwen-VL（生成器泛化）。
