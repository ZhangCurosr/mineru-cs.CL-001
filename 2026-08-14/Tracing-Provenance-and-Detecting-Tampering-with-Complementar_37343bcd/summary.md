---
title: "Tracing-Provenance-and-Detecting-Tampering-with-Complementar"
source: https://arxiv.org/pdf/2608.12713v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:23:28"
field: "LLM 水印与内容溯源"
keywords: ["LLM watermarking", "provenance tracing", "tamper evidence", "piggyback spoofing", "unbiased reweighting", "green-red list"]
innovations: ["在单个文本中共嵌入鲁棒与脆弱双信号以实现溯源与篡改检测的统一框架", "基于归一化文本（非 token ID）的种子设计使篡改检测锚定于读者可见内容", "多轮无偏锦标赛重加权配合周期性轮次分配，支持两信号强度可调共嵌入"]
benchmarks: ["C4 realnewslike", "LFQA"]
---

# 论文速读：Tracing Provenance and Detecting Tampering with Complementary LLM Watermarks

## 一句话总结
论文提出了 Cocktail 水印方案，通过在同一个文本中共同嵌入一个鲁棒信号和一个脆弱信号，同时实现 LLM 生成内容的溯源归因和篡改证据检测，解决了单信号水印无法兼顾"抗编辑溯源"与"暴露篡改"这一根本矛盾（piggyback spoofing 攻击漏洞）。

## 研究问题与动机
- **Piggyback spoofing 攻击威胁**：现有 LLM 水印为抵抗篡改删除攻击，将信号设计得对编辑鲁棒，但这反而允许攻击者在不被发现的情况下篡改关键内容并保留溯源属性，即"piggyback spoofing"。
- **单信号的内在矛盾**：溯源归因需要信号对编辑鲁棒，篡改证据需要信号对编辑敏感，两者目标相反，单一信号无法同时满足。
- **既有方案的缺陷**：最接近的方案 Bileve 对 delivered text 的签名验证在 91.1% 的无修改文本上失败（因 re-tokenization 导致 ID 漂移），且将签名写入后续 token 载体，这些 token 自身不受保护；生成质量急剧下降，perplexity 上升约 6 倍。
- **归因与篡改检测需统一框架**：现有方法通常只解决单一任务，缺乏一个能同时输出"完整 / 篡改 / 无水印"三种判定的统一检测机制。

## 核心贡献（创新点）
- **解耦设计**：在一个文本中共同嵌入鲁棒信号和脆弱信号，分别服务溯源和篡改检测，将二元判定扩展为三元判定。与已有工作（如 Bileve）的本质区别在于，Cocktail 的信号直接锚定在读者收到的归一化文本上，而非 token ID。
- **归一化种子设计**：首次将水印信号种子定义在归一化文本上（而非 token IDs），使篡改检测建立在读者可见的内容层面，而非生成中间产物。
- **无偏共嵌入机制**：通过多轮向量化锦标赛重加权（unbiased tournament reweighting）将两路信号共同嵌入每个 token，同时保持生成分布的无偏性，不损失生成质量。
- **可调节的强度比率**：提出周期性轮次分配模式（round-allocation pattern），以 r:1 比率控制两信号的强度权衡，碰撞熵预算的消耗使该比率为单调调控旋钮。
- **系统性实验验证**：在 Llama-3.2-1B 和 Gemma-3-4B 两个模型、两个数据集上评估，Cocktail 的篡改检测率较最强单信号基线提升超过 66 个百分点（最高达 100% vs 最高基线 23.1%），同时保持归因和 perplexity 竞争力。

## 方法详解

**1. 互补信号设计**
- 每个信号均为词汇表上的绿-红划分（green-red list），由独立密钥 $k_r \neq k_f$ 生成：$g_*^{(t)} = \mathrm{PRNG}(k_*, s_*(t)) \in \{0,1\}^{|\mathcal{V}|}$，$* \in \{r,f\}$。
- **鲁棒信号**：使用短窗口 $h_r = 1$（仅前一个 token + 自身），编辑影响范围小，信号能幸存。
- **脆弱信号**：使用长窗口 $n_f$（完整归一化前缀 + 自身），任何编辑都会污染后续大量 token 的种子，信号易被破坏。

**2. 归一化文本种子**
- 种子计算基于归一化后的文本前缀：$s_r(t) = H(\mathcal{N}(x_{\max(1,t-h_r):t-1}))$，$s_f(t) = H(\mathrm{suffix}_{n_f}(\mathcal{N}(x_{1:t-1})))$。
- 归一化操作包括：移除不可见字符、Unicode NFKC 规范化、同形字折叠（homoglyph folding）、大小写折叠、空白符折叠——所有操作幂等且跨环境可复现。

**3. 无偏共嵌入（Vectorized Tournament Reweighting）**
- 每轮重加权：$\hat{p}^{(i)}(x) = (1 + g^{(i)}[x] - q^{(i)}) \hat{p}^{(i-1)}(x)$，其中 $q^{(i)} = \sum_x g^{(i)}[x]\hat{p}^{(i-1)}(x)$。
- 由于 $g^{(i)}[x] \sim \mathrm{Bernoulli}(0.5)$，每一轮的期望调整因子为 1，因此 $\mathbb{E}[\hat{p}^{(d)}(x)] = p(x)$，无偏性得以保持。
- 周期分配模式：每 $r+1$ 轮中 $r$ 轮分配给鲁棒信号、1 轮分配给脆弱信号，近似按 r:1 比例分享碰撞熵预算。

**4. 二维得分空间与三元判定**
- 鲁棒得分 $z_r$ 衡量溯源，脆弱得分 $z_f$ 衡量完整性：$z_* = (\sum_{t}\sum_{i:\pi_i=*} g^{(i)}[x_t] - \frac{1}{2}Td_*)/\sqrt{Td_*/4}$。
- 判定规则：先判断 $z_r > \tau_r$（是否来自水印模型），再在此前提下判断 $z_f > \tau_f$（是否被篡改）：
  - **Intact**：$z_r > \tau_r$ 且 $z_f > \tau_f$
  - **Tampered**：$z_r > \tau_r$ 但 $z_f \leq \tau_f$
  - **No-Watermark**：$z_r \leq \tau_r$

## 实验与结果

- **数据集**：C4（realnewslike 子集）和 LFQA。
- **模型**：Llama-3.2-1B、Gemma-3-4B。
- **基线**：KGW、Unigram、SynthID、SIR、Bileve。
- **攻击类型**：Paraphrase（Dipper）、Round-trip Translation（opus-mt）、Token Substitution、Sentiment Flip（gpt-oss:20b）、Homoglyph Substitution。
- **主要结果（TPR@1%FPR）**：
  - 归属（Attribution）：Cocktail 在各设置下均达 99.7–100.0%（无攻击），受攻击后仍保持竞争力，4:1 变体在 paraphrase 上达 92.4%。
  - 篡改检测（Tamper Evidence）：Cocktail 各变体在 1% 误报率下检测出 89.5–100.0% 的篡改文本；最强单信号基线最高仅 23.1%（KGW），Cocktail 最低 89.5% 也超出基线超 66 个百分点。
  - Perplexity：Cocktail（7.5–10.3）与基线处于同一量级，无显著质量损失；Bileve 达 62.2（约 6 倍）。
- **Ablation 关键发现**：
  - 归一化必须在种子阶段（Seeding placement）而非仅检测阶段才能同时维持归属（100.0%）和篡改召回（98.3%）。
  - 共嵌入优于逐 token 单一信号分配：后者在 T=50 时归属从 98.3% 降至 86.0%。
  - 脆弱窗口越长，篡改检测越好：$n_f=40$ 时 7.0%，$n_f=200$ 时 88.0%。

## 相关工作脉络

- **KGW [4]**：首个绿色-红色列表生成式水印，以单个 token 为种子，是最直接的强基线，但仅有鲁棒性，无篡改检测能力。
- **Unigram [6]**：固定全局绿红列表以最大化编辑鲁棒性，context-independent 使其对编辑极端鲁棒，但也完全丧失对编辑的敏感性（tamper evidence 极低）。
- **SIR [7]**：基于语义 embedding 的种子，对改写鲁棒，但在半数字设置下调用 semantic invariant 模块后 tamper evidence 为 0.0%，揭示了单信号的内在矛盾。
- **SynthID [5]**：基于 tournament sampling 的无偏水印，Cocktail 直接复用其无偏重加权理论（两样本锦标赛），但 SynthID 仅嵌入单一路信号（鲁棒端）。
- **Bileve [9]**：最接近的双信号方案，采用数字签名 + 统计信号组合，但签名验证因 re-tokenization 在 91.1% 的无修改文本上失败，且 perplexity 升至 62.2（Cocktail 仅 10.3）。
- **Dipper [11]**：基于翻译回译的攻击方法，用于评估水印的改写鲁棒性；本文将其作为归因任务的攻击基线之一。

## 局限性与未来方向

- **尾部编辑的局限**：仅编辑末尾 token 时，仅污染脆弱信号的最后部分种子，目前的检测是对全篇整体打分；论文自述对尾部段落的 $z_f$ 检测是自然的未来方向。
- **鲁棒窗口的 stealing 风险**：短鲁棒窗口与现有鲁棒水印一样，面临通过查询推断绿红列表的 watermark stealing 攻击（引用 [39][40]）；脆弱信号虽因全前缀种子更难伪造，但整体安全性仍需进一步分析。
- **窗口长度与质量的 trade-off**：较长的脆弱窗口虽提升篡改检测率，但也消耗更多碰撞熵，理论上可能轻微影响生成质量，需在具体场景中调参。
- **未扩展到其他模态**：论文主要在文本上验证，提到"recipe 可能扩展到其他需要溯源和篡改证据的生成模态"，但尚未实验验证。

## 研究启发与可借鉴点

- **互补信号设计范式**：将相互冲突的需求（鲁棒 vs 脆弱）解耦为两个独立信号，通过联合得分空间实现三元判定，这一思路可迁移到其他需要多属性验证的场景（如图像水印、视频溯源）。
- **归一化锚定设计**：以读者可见的归一化文本而非内部 token ID 作为种子基础，是一个简洁而有效的工程直觉——使检测对齐用户感知，可推广至防御 homoglyph 等字符级扰动。
- **无偏重加权的可组合性**：tournament reweighting 的无偏性在多轮独立密钥下得以保留，这一性质保证了多信号共嵌入的可行性，可为后续多任务水印设计提供理论基础。
- **周期性轮次分配作为强度旋钮**：用碰撞熵预算的非线性消耗来近似支撑线性比率旋钮，提供了一个可解释且实用的超参调节接口，类似思路可用于其他受限资源下的多目标优化。
- **与团队方向结合机会**：若团队关注 LLM 内容的安全审计或可信 AI，Cocktail 的三元判定框架可直接作为下游管道的检测层；其归一化模块可与现有的文本预处理 pipeline 无缝集成。

## 关键术语表

- **Piggyback Spoofing**：攻击者在保留水印信号强度的同时篡改关键内容，使水印系统错误地将篡改后的内容归因于原模型。
- **Green-Red List**：将词汇表随机划分为绿色和红色两部分，通过提高绿词概率来嵌入水印信号。
- **Unbiased Reweighting**：通过 tournament 或闭式调整使采样分布的期望保持与原始 LLM 分布一致，避免生成质量下降。
- **Collision Entropy（碰撞熵）**：衡量分布集中程度的 Renyi 熵（二阶），在水印中充当信号嵌入的"预算"资源。
- **Z-Score（z 分数）**：将绿词命中计数标准化后得到的检测统计量，用于量化文本与水印信号的匹配程度。
- **Homoglyph Substitution**：用视觉上相似但编码不同的字符替换原文字符（如拉丁 a 替换西里尔 а），属于不可见编辑攻击。
- **Normalization**：对文本进行不可见字符移除、NFKC 规范化、同形字折叠、大小写折叠和空白折叠等操作，使种子在传输前后保持一致。
- **Three-State Decision**：将检测结果分为 Intact（完整）、Tampered（篡改）、No-Watermark（无水印）三种状态，由二维得分空间中的阈值划分。

## 可复现要素

- **数据集**：C4（realnewslike subset，validation split，seed=42，跳过前 5000 篇文档）、LFQA——均来自公开数据集，论文有具体流式读取方式。
- **代码/权重**：论文未提供开源代码链接；基线（KGW、Unigram、SynthID、SIR）使用官方代码运行；Cocktail 的实现细节在附录 B 中给出（Algorithm 1 & 2），但未声明代码仓库。
- **关键超参**：$h_r = 1$，$n_f =$ 完整前缀，$d = 30$ 轮，轮次比率 $d_r:d_f \in \{1:1, 2:1, 4:1\}$，温度 1.0，top-100 logits 采样，检测取前 300 token，阈值 $\tau_r/\tau_f$ 在 1% 尾部分位数校准。
