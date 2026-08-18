---
title: "MuseCritic-Learning-Multi-Aspect-Song-Rewards-through-Natura"
source: https://arxiv.org/pdf/2608.11755v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 11:52:35"
field: "多模态奖励建模与歌曲生成"
keywords: ["reward modeling", "song generation", "aesthetic evaluation", "critique-based", "semi-scalar reward", "GRPO", "multi-modal RLHF"]
innovations: ["将自然语言评述作为连接歌曲表征与连续奖励分数的显式中间变量", "SFT初始化+自生成评述的两阶段训练缓解critique分布偏移", "critique-conditioned奖励模型成功用于GRPO强化学习优化歌曲生成"]
benchmarks: ["SongEval", "Music Arena"]
---

# 论文速读：MuseCritic: Learning Multi-Aspect Song Rewards through Natural-Language Aesthetic Critiques

## 一句话总结
MuseCritic 是一种面向完整歌曲的半标量奖励模型，通过先生成覆盖五个美学维度的自然语言美学评述（critique），再以此为中间表示预测连续奖励分数，将可解释的语言推理引入歌曲美学评估，有效降低了评分误差并提供了可用于 GRPO 强化学习的优化信号。

## 研究问题与动机
1. **长视频/长歌曲生成模型的审美对齐需求**：Text-to-song 生成模型已从短片段进化到分钟级完整歌曲（如 DifRhym、YuE、LeVo、Muse），可靠的审美奖励模型对模型选择和对齐人类偏好至关重要。
2. **现有评估器无法提供可解释的美学证据**：客观指标（phoneme error rate、MuLan、MuQ-MuLan）只能捕捉歌词清晰度或音频-文本对应关系，无法反映感知和艺术质量；PAM、MusicEval、Audiobox Aesthetics 等扩展到了音频质量和音乐美学，但仍直接映射音频到分数，不提供判断依据。
3. **LLM-as-a-Judge 和评述型奖励模型在文本领域成效显著，但尚未系统扩展到完整歌曲的多维度连续审美评分**：CLoud、Critic-RM、Generative Verifiers 等方法证明了显式语言中间表示可提升奖励判断，但它们主要针对文本响应，音乐领域的多维度连续评估仍处于空白。
4. **自评述（self-generated critique）的训练-推理分布偏移问题**：直接用外部教师模型生成的评述训练奖励模型，会在推理时产生 critique distribution shift，影响评分一致性。

## 核心贡献（创新点）
1. **提出 MuseCritic：首个将自然语言美学评述作为中间变量的半标量歌曲奖励模型**——与已有直接回归分数的判别式奖励模型（如 SongEval/UTMOS）的本质区别在于，评述不是事后附加的解释文本，而是连接歌曲表征与奖励分数的显式中间变量，组织了分布在长歌曲中的听觉证据。
2. **系统性地设计了 SFT-init + self-generated critique 的两阶段训练管线以缓解分布偏移**——与直接使用外部教师评述（offline critique）或在无 SFT 初始化的情况下自生成评述的方法相比，本方法显著改善了绝对分数一致性和排名相关性。
3. **将 MuseCritic 作为 GRPO 的奖励模型应用于 Muse-0.6B 的强化学习训练**——在 SongEval 五个维度和 Audiobox Aesthetics 四个维度共九个指标上全部实现数值提升，证明了 critique-conditioned 奖励可提供有效的优化信号。
4. **在 SongEval 测试集和 Music Arena 外部偏好基准上均达到最优结果**——与 Gemini-3.1-Pro（未适配的 LLM-as-a-Judge）和 SongEval（UTMOS）基线相比，在绝对误差和跨分布偏好泛化上均有显著提升。

## 方法详解
**模型架构**：基于指令调优的音频语言模型 MOSS-Audio-8B-Instruct 作为共享骨干（backbone），包含两个组件：评述生成器 $g_c = h_{LM} \circ f_\theta$（语言建模头，生成自然语言评述）和审美评分器 $g_s = h_{RM} \circ f_\theta$（奖励建模头，输出连续分数），两者共享参数 $\theta$。

**评述生成**：给定歌曲 $x$ 和评估准则 $p$，模型首先生成评述 $\hat{c} = g_c(x, p)$，评述包含五个带标签的分析部分：Overall Coherence（整体连贯性）、Memorability（记忆性）、Naturalness of Vocal Breathing and Phrasing（声乐呼吸与乐句自然性）、Clarity of Song Structure（歌曲结构清晰度）、Overall Musicality（整体音乐性）。

**奖励预测**：评分器联合条件于歌曲、准则和评述，预测五个连续审美分数：$\hat{\mathbf{y}} = g_s(x, p, \hat{c})$。与传统直接预测不同，评述 $\hat{c}$ 是模型在打分前显式组织的证据表示，帮助将长歌曲中分散的听觉证据按维度对齐。

**两阶段训练**：
- **Stage I（SFT 初始化评述生成器）**：使用 Gemini-3-Pro 作为外部教师，观察歌曲 $x_i$、评估准则 $p$ 和专家均值评分 $\mathbf{y}_i^*$，生成与评分极性一致的维度特异性评述，构建离线评述数据集 $\mathcal{D}_{off}$；然后对骨干模型在 $\mathcal{D}_{off}$ 上进行标准 next-to-token cross-entropy 监督微调（学习率 $5 \times 10^{-5}$，1 epoch），得到评述生成器 $g_c'$。生成评述后由8名音乐专家人工验证，确保事实准确性且无幻觉。
- **Stage II（奖励学习，使用自生成评述）**：用 $g_c'$ 为每个训练样本重新生成评述 $c_i^{on} = g_c'(x_i, p)$，替换教师评述构建自生成评述数据集 $\mathcal{D}_{on}$；从 SFT checkpoint 初始化共享 backbone，使用 LoRA（rank 8, scaling 32）更新骨干，全参数更新奖励头（学习率 $2 \times 10^{-4}$），在 $\mathcal{D}_{on}$ 上用 MSE 损失联合训练10个 epoch，权重衰减0.1。
- **推理**：最终模型首先生成自评述，再基于歌曲+准则+自评述联合预测五个分数。

**奖励头实现**：最后一个有效 token 的 hidden state 经 dropout（0.1）→ 线性投影 $\mathbb{R}^d \to \mathbb{R}^5$ → sigmoid 范围变换，输出 $(1,5)^5$ 内的连续分数。使用 DeepSpeed ZeRO-3。

## 实验与结果
**数据集**：SongEval（2,399 首中英文完整歌曲，16位音乐专家标注的五维评分，>140小时音频，9大流派），随机 shuffle seed=42 后取前2,199首训练、剩余200首测试。Music Arena（733 首歌曲偏好对，来自 SongEval 未覆盖的生成系统）。

**基线**：Gemini-3.1-Pro（LLM-as-a-Judge，相同准则生成评述和五维评分）、SongEval（UTMOS，重新在相同2,199/200分割上训练10个epoch）、Audiobox Aesthetics、Qwen3-Omni-30B-A3B-Instruct（LLM-as-a-Judge，解析其五维评分取均值）。

**主要结果**：
- **SongEval 测试集（200首）**：MuseCritic 在所有五个维度上同时达到最低 MSE 和最高 LCC/SRCC/KTAU。宏平均 MSE 从 0.2875（SongEval 基线）降至 0.2316（−19.5%）；宏平均 LCC/SRCC/KTAU 从 0.8793/0.8531/0.6766 提升至 0.9068/0.8838/0.7178。
- **Music Arena（733 偏好对）**：MuseCritic 达到最高准确率 71.35%，比 SongEval（70.80%）、Audiobox Aesthetics（68.49%）、Qwen3-Omni（53.75%）分别提升 0.55/2.86/17.60 个百分点。
- **GRPO 下游强化学习**：用 MuseCritic 作为奖励信号训练 Muse-0.6B 一个 epoch（6×H200 GPU），在所有九个指标上均有提升：Audiobox PC 提升 +0.15、CE 提升 +0.12；SongEval 各维度提升 0.05–0.06。

## 相关工作脉络
1. **RLHF 奖励模型（Ouyang et al., 2022）**：在预训练语言模型上接标量头拟合人类偏好；MuseCritic 的区别是将奖励建模扩展到多维度连续审美评分，并以评述为中间表示而非直接映射。
2. **LLM-as-a-Judge（Zheng et al., 2023）**：用强语言模型进行评估和提供理性；MuseCritic 的区别在于针对完整歌曲音频设计，且评述作为奖励预测的条件输入而非仅输出文本。
3. **CLoud（Ankner et al., 2024）**：先生成自然语言评述再预测奖励；MuseCritic 将其范式从文本扩展到长音频，并引入 self-generated critique 缓解分布偏移。
4. **Critic-RM（Yu et al., 2025）**：联合训练自生成评述和奖励预测；MuseCritic 的两阶段设计（SFT初始化 → 奖励学习）更系统地隔离了评述质量和奖励质量的优化。
5. **SongEval（Yao et al., 2025）**：定义了完整的五维歌曲审美评估准则和数据集；MuseCritic 沿用了 SongEval 准则但增加了评述生成能力，使模型既能描述五维审美质量又能分配连续奖励。
6. **Audiobox Aesthetics（Tjandra et al., 2025）**、**PAM（Deshmukh et al., 2024）**：预测参考无关的音频质量指标；MuseCritic 的差异在于使用 expert-annotated 五维准则，提供更细粒度的审美评估。

## 局限性与未来方向
1. **计算成本较高**：自回归评述生成在奖励预测之前，比直接分数回归更耗时。
2. **训练数据语种和流派受限**：主要在 SongEval 的中英文人声歌曲上训练，缺乏其他语言和多音乐流派的覆盖。
3. **专家评分-评述配对数据稀缺**：现有歌曲数据集很少将专家评分与专家评述配对，依赖 LLM 合成评述引入了潜在的分布偏移风险。
4. **未来方向**：可扩展到更多语言和音乐流派的训练、探索更高效的多模态评述生成方式（如并行生成而非自回归）。

## 研究启发与可借鉴点
1. **SFT 初始化 + self-generated critique 的两阶段设计可迁移到其他多模态奖励建模任务**：如视频审美评估、语音质量评估等需要多维度连续评分的场景，均可借鉴"先训生成器、再训奖励器"的范式。
2. **评述作为中间变量不仅能提升绝对分数准确度，还能保持排名信息**：消融实验显示去掉评述后排序相关系数变化不大但 MSE 翻倍，说明评述对校准分数分布范围（而非仅排序）有不可替代的作用，这对需要 interval-scale 奖励的 RL 训练尤为重要。
3. **外部教师 LLM 辅助构建训练数据 + 人工验证的方案是构建多模态评测数据的可行路径**：8名音乐专家对 Gemini 生成评述的事实准确性和幻觉进行验证，保证了 $\mathcal{D}_{off}$ 的质量，该方法论可复用于其他需要专家标注的音频评测任务。
4. **将 critique-conditioned 奖励模型直接用于 GRPO 强化学习是一个完整的端到端验证闭环**：证明评述型奖励不仅可评估，还能提供有效的策略优化信号，这对"评估即优化"的研究方向有示范意义。

## 关键术语表
**MuseCritic**：本文提出的半标量歌曲奖励模型，先生成自然语言美学评述再预测五维连续分数。
**半标量奖励模型（Semi-scalar reward model）**：结合生成式评述和标量分数的奖励建模方式，评述是中间变量而非仅最终输出。
**Critique distribution shift**：训练时使用的评述分布（外部教师生成）与推理时（模型自身生成）之间的差异，本文通过两阶段训练缓解此问题。
**SongEval**：包含2,399首中英文完整歌曲的五维审美评估数据集，由16位音乐专家标注，定义连贯性、记忆性、声乐自然性、结构清晰度、音乐性五个维度。
**GRPO（Group Relative Policy Optimization）**：Shao et al. (2024) 提出的强化学习算法，以组内相对优势更新策略，本文用于缪斯0.6B的奖励驱动训练。
**MOSS-Audio-8B-Instruct**：本文使用的音频语言模型骨干，具备音频理解与生成能力。
**Music Arena**：Kim et al. (2025) 构建的外部偏好基准，含733对歌曲偏好数据，来自 SongEval 未覆盖的生成系统，用于评估分布外泛化。
**Audiobox Aesthetics**：Tjandra et al. (2025) 的多用途音频美学评估模型，预测内容 enjoyment、content usefulness、production complexity 和 production quality 四个维度。

## 可复现要素
- **数据集**：SongEval（公开，需从原作者处获取）；Music Arena（公开）。训练集：SongEval 2,199 首，测试集：200 首（seed=42 随机分割）。
- **代码/权重**：项目仓库已公开，https://github.com/WuqnEl/MuseCritic。
- **骨干模型**：MOSS-Audio-8B-Instruct。
- **关键超参**：SFT 学习率 $5 \times 10^{-5}$，1 epoch，gradient accumulation 8，global batch size 32，max seq len 10,000；奖励学习 LoRA rank 8, scaling 32，学习率 $2 \times 10^{-4}$，10 epoch，weight decay 0.1；GRPO 训练学习率 $10^{-6}$，8 candidate per prompt，温度 0.9，top-p 0.9，repetition penalty 1.3。
- **硬件**：训练使用 4× NVIDIA H200 GPU，GRPO 使用 6×H200 GPU。
- **教师模型**：Gemini-3-Pro（生成离线评述）。
