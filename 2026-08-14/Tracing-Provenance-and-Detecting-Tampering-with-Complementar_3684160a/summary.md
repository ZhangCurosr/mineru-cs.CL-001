---
title: "Tracing-Provenance-and-Detecting-Tampering-with-Complementar"
source: https://arxiv.org/pdf/2608.12713v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:23:08"
field: "大语言模型水印与安全"
keywords: ["LLM watermark", "provenance", "tamper evidence", "piggyback spoofing", "unbiased reweighting", "text authentication", "green-red list", "dual-signal watermark"]
innovations: ["双信号解耦：共嵌入鲁棒与脆弱信号实现溯源与篡改证据统一", "归一化文本锚定种子：使篡改证据针对读者可见内容而非 token ID", "无偏轮次周期分配：以碰撞熵为预算的可调 round ratio 控制双信号强度比"]
benchmarks: ["C4 realnewslike", "LFQA", "Llama-3.2-1B", "Gemma-3-4B", "Paraphrase (Dipper)", "Round-trip Translation (opus-mt)", "Sentiment Flip", "Homoglyph Substitution"]
---

# 论文速读：Tracing-Provenance-and-Detecting-Tampering-with-Complementary-LLM-Watermarks

## 一句话总结
论文提出 **Cocktail**，一种同时在每个生成 token 中共嵌入**鲁棒信号**与**脆弱信号**的 LLM 文本水印，首次实现溯源归因（provenance）与篡改证据（tamper evidence）的双目标统一；在两个模型与两个数据集上，篡改检测率比最强单信号基线高出 **66+ 个百分点**，同时保持相当的溯源稳健性与生成质量（PPL）。

## 研究问题与动机
1. **现有水印的猪背欺骗（piggyback spoofing）漏洞**：为抵御编辑攻击而设计的鲁棒信号，同样允许敌手修改关键内容后仍保留水印归属，导致"经篡改的文本依然被归因于原模型"。
2. **单信号根本性困境**：溯源需要信号对编辑鲁棒，篡改证据需要信号对编辑敏感，二者是相反需求，单信号方案只能取其一。
3. **已有最相关工作（Bileve）的缺陷**：签名锚定在 token ID 而非读者可见内容；重分词导致未编辑文本 91.1% 验证失败；PPL 飙升至 62.2（约 6 倍于 Cocktail），且检测计算开销极高（单次检测 144s）。

## 核心贡献（创新点）
1. **双信号解耦设计**：在同一文本中同时共嵌入鲁棒信号（短窗口、抗编辑）与脆弱信号（长窗口、敏感编辑），将二元判定升级为三元（Intact/Tampered/No-Watermark）。与已有工作的本质区别在于首次显式分离两个相反目标，而非在单一信号上修修补补。
2. **基于归一化文本的篡改证据定义**：将信号种子锚定于**归一化后的读者可见文本**而非 token ID，使篡改证据针对内容本身而非中间表示；这与 Bileve 等基于 token-ID 签名的方案有本质区别。
3. **无偏轮次共嵌入机制**：通过多轮无偏锦标赛重加权（unbiased vectorized tournament reweighting）逐轮交替嵌入两种信号，周期分配模式以可调节的 round ratio 控制两信号强度比，同时保持期望分布不变；与 BiMark/ENS 等组合独立无偏步骤的方案相比，消耗的是可问责的碰撞熵预算。
4. **系统级实验验证**：在 Llama-3.2-1B 和 Gemma-3-4B 上证明 Cocktail 在保持归因能力接近最强单信号基线的同时，篡改检测率实现跨越式提升（89.5–100.0% vs 基线最高 23.1%）。

## 方法详解
- **双信号定义**：两种独立的 green-red 列表 $g_r$（鲁棒）和 $g_f$（脆弱），分别由不同密钥 $k_r \neq k_f$ 生成：
  $$g_{*}^{(t)} = \mathrm{PRNG}(k_{*}, s_{*}(t)) \in \{0,1\}^{|\mathcal{V}|}, \quad * \in \{r,f\}$$
- **窗口设计**：鲁棒信号使用短窗口 $h_r=1$（前 1 个 token），对编辑影响局部；脆弱信号使用全长字符窗口 $n_f$（整个归一化前缀），编辑一次即污染下游大量种子。
- **归一化种子**：种子基于归一化文本计算，$s_r(t) = H(\mathcal{N}(x_{\max(1,t-h_r):t-1}))$，$s_f(t) = H(\mathrm{suffix}_{n_f}(\mathcal{N}(x_{1:t-1})))$；归一化包括零宽字符清除、Unicode NFKC、同形字折叠、大小写折叠和空白折叠，确保读者可见编辑可被检测。
- **无偏共嵌入**：每步进行 $d=30$ 轮向量锦标赛重加权，第 $i$ 轮按周期模式 $\pi$ 分配给鲁棒或脆弱信号，更新分布 $\hat{p}^{(i)}(x)=(1+g^{(i)}[x]-q^{(i)})\hat{p}^{(i-1)}(x)$，期望分布始终保持不变。
- **两轮比控制强度**：周期模式 $\pi_i$ 按 $d_r:d_f \in \{1:1, 2:1, 4:1\}$ 交替，round ratio 单调近似实际信号强度比（受碰撞熵非线性消耗影响）。
- **二维得分空间与三元判定**：检测时计算 $z_r$（溯源得分）和 $z_f$（篡改证据得分）：
  $$z_* = \frac{\sum_{t,i:\pi_i=*} g^{(i)}[x_t] - \frac{1}{2}Td_*}{\sqrt{Td_*/4}}, \quad * \in \{r,f\}$$
  判定规则：$\mathsf{Det}(x)=\mathrm{Intact}$ 当 $z_r>\tau_r \land z_f>\tau_f$；$\mathsf{Det}(x)=\mathrm{Tampered}$ 当 $z_r>\tau_r \land z_f\leq\tau_f$；否则为 No-Watermark。先判溯源再判篡改，避免无水印文本与篡改文本在 $z_f$ 轴上的歧义。

## 实验与结果
- **数据集**：C4 realnewslike（新闻类）和 LFQA（问答类）；模型：Llama-3.2-1B、Gemma-3-4B；每组合 500 篇文本。
- **基线**：KGW、Unigram、SynthID、SIR（均为单信号推断时水印）。
- **攻击**：Paraphrase（Dipper）、Round-trip Translation（opus-mt）、Sentiment Flip（gpt-oss:20b + classifier 验证）、Token Substitution、Homoglyph Substitution（同形字替换 30%）。
- **评估指标**：TPR@1%FPR（溯源任务与篡改检测任务分别报告），以及 PPL（Mistral oracle）。
- **主要结果（Llama-3.2-1B/C4）**：
  - 溯源无攻击：Cocktail 4:1 达 **99.8%**，与 SynthID（100.0%）持平。
  - 溯源+意译：Cocktail 4:1 达 **92.4%**，超越 SynthID（91.2%）。
  - 篡改检测（Sentiment-Flip）：Cocktail 1:1 达 **99.4%**，远超基线最高 23.1%（KGW）；即使最差变体 Cocktail 4:1 也达 **89.5%**，领先基线 **66+ pp**。
  - PPL：Cocktail（8.8–9.3）在各模型上均处于基线范围内，无质量下降。
- **Ablation 关键结论**：
  - 种子处归一化（Seeding）使同形字攻击下篡改召回率保持 **98.3%**；仅检测处归一化则降至 43.5%。
  - 共嵌入 vs 每 token 单信号分配：共嵌入在 50 token 处溯源 98.3% vs 86.0%，round-trip 溯源 89.8% vs 60.2%。
  - 脆弱窗口长度 $n_f$：从 40 到 200，篡改检测从 7.0% 跃升至 **88.0%**，确认长窗口必要性。

## 相关工作脉络
1. **KGW [4]**：首个推理时绿红分割水印，基于前一 token 种子；Cocktail 在共嵌入和无偏框架上扩展，但 KGW 只有单信号。
2. **SynthID [5]**：无偏锦标赛水印，Cocktail 直接复用其 vectorized tournament 作为无偏嵌入基础，但引入双信号与周期分配。
3. **Unigram [6]**：全局固定绿红分割，编辑鲁棒性最强但脆弱信号感知为零；Cocktail 通过双信号同时覆盖鲁棒与敏感两端。
4. **SIR [7]**：基于语义嵌入的鲁棒水印；语义种子虽抗编辑但对读者可见篡改不敏感，Cocktail 的归一化文本种子填补这一空白。
5. **Bileve [9]**：最接近的同类工作，但采用 token-ID 级数字签名；Cocktail 通过内容锚定种子解决重分词失效问题（Bileve 在 Llama-3.2-1B 上 91.1% 误报）。
6. **Lu et al. [27]（图像水印）**：最早提出"鲁棒+脆弱"双信号鸡尾酒水印思想，Cocktail 将其原理首次推广至生成式 LLM 文本领域。

## 局限性与未来方向
1. **尾部编辑的脆弱信号损失**：仅修改末尾 token 时，脆弱种子仅尾部受损，全文整体 $z_f$ 下降有限；作者建议扩展到分段（trailing segment）打分是自然改进方向。
2. **鲁棒信号窗口较短的泄露风险**：$h_r=1$ 的短窗口与现有鲁棒方案共享查询推断（watermark stealing）脆弱性；脆弱信号因长窗口更难被窃取，反而成为更安全的信息锚点。
3. **未涉及其他生成模态**：当前仅针对文本，双信号互补原则有望推广至图像/音频等生成模态，但未在本论文中验证。
4. **round ratio 与强度比的非线性关系**：周期分配是单调但非线性的代理，精确建模需依赖碰撞熵的更高阶分析。

## 研究启发与可借鉴点
1. **双信号解耦范式**：将相互冲突的需求（鲁棒 vs 敏感）分解为两个独立信号并通过联合决策空间统一处理，可迁移至其他安全/取证场景（如深度伪造检测、多模态溯源）。
2. **归一化种子锚定读者可见内容**：用 NFKC+同形字折叠+空白折叠等确定性价操作替代原始 token ID 作为种子源，解决了重分词漂移问题，适用于任何基于内容的数字水印系统。
3. **无偏重加权 + 周期分配控制信号预算**：将碰撞熵视为可分配预算，用周期模式调节两信号强度比，为多目标优化下的嵌入设计提供了一种简洁可调的接口。
4. **三元判定空间的阈值校准方法**：先 $z_r$ 判溯源、再 $z_f$ 判篡改的串行规则有效消解了无水印/篡改文本在单轴上的歧义，此分层决策思路可直接用于其他双指标检测系统。
5. **与团队方向的结合机会**：本团队的 LLM 溯源/抗攻击研究方向可借鉴 Cocktail 的双信号框架，探索多模态（图像/视频）水印中的类似设计，或在对抗训练中加入脆弱信号模块以增强内容完整性验证能力。

## 关键术语表
- **Piggyback Spoofing**：敌手修改水印文本关键内容后仍保留高水印信号强度，使篡改内容继续被归因于原模型的新型攻击。
- **Provenance（溯源）**：判定文本是否源于特定 LLM 的能力，对应水印的绿色 token 统计显著性。
- **Tamper Evidence（篡改证据）**：判定已生成文本是否被读者可见编辑过，要求水印信号对编辑敏感。
- **Green-Red List**：基于密钥和种子生成的词汇表二值划分，绿色 token 被提升概率、红色 token 被抑制。
- **Unbiased Reweighting（无偏重加权）**：锦标赛采样等重加权操作使期望分布保持不变，从而不损害生成质量。
- **Collision Entropy（碰撞熵）**：衡量分布随机性的信息论量，每个嵌入轮次消耗其一部分，是信号强度的预算资源。
- **Homoglyph Substitution（同形字替换）**：用视觉相似的 Unicode 码位（如拉丁 a 与西里尔 а）替换原文字符，对读者不可见但对基于 token ID 的水印构成攻击。
- **2D Score Space**：双信号得分 $(z_r, z_f)$ 构成的二维空间，Intact/Tampered/No-Watermark 三类文本分布在不同区域，支持三元判定。

## 可复现要素
- **数据集**：C4 realnewslike（public，validation split）、LFQA（public）；每模型-数据集组合生成 500 篇文本，每篇最长 512 token。
- **代码/权重**：论文声明配置可复现（generation params.json），Cocktail 未声明独立代码仓库；基线使用官方代码（SynthID、SIR 等）；Bileve 使用官方实现复测。
- **关键超参**：$h_r=1$，$n_f=$ 全前缀，$d=30$，round ratio $\in \{1:1, 2:1, 4:1\}$，温度 1.0，top-100 logits 采样；$\tau_r$ 校准于无水印 $z_r$ 的第 99 百分位，$\tau_f$ 校准于完整文本 $z_f$ 的第 1 百分位（各 1% FPR）。
