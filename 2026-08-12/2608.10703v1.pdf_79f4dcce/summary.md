---
title: "Your LLM, Your Style: Behavioral Mode Axes for LLM Behavioral Control"
source: https://arxiv.org/pdf/2608.10703v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:39:14"
field: "LLM行为对齐与可解释性"
keywords: ["LLM人格", "行为模式轴", "激活空间控制", "行为数据", "特质漂移", "BCL带"]
innovations: ["提出thought-derived与response-derived两种BMA来源并证明前者控制更干净", "发现行为控制集中在earlier-to-middle Behavioral Control Layer带且跨模型泛化"]
benchmarks: ["20个心理测量子域×4种prompt register情境化行为探针"]
---

# 论文速读：Your LLM, Your Style: Behavioral Mode Axes for LLM Behavioral Control

## 一句话总结
本文提出了一种基于情境化行为数据（B-data）的框架来研究和控制大语言模型的"行为人格"，通过从对比行为轨迹中提取**行为模式轴（BMA）**实现激活空间干预；研究发现基于中间推理（thought-derived）的BMA比基于最终回答（response-derived）的BMA能提供更干净的行为控制，且控制效果集中在较早到中层的**行为控制层带（BCL bands）**。

## 研究问题与动机
- **现有问卷式人格测量极不稳定**：直接对LLM施测Big Five、MBTI等人格量表（第一人称自陈），结果对措辞、选项顺序等表面线索高度敏感（mean gap 22.7 percentage points），且自评分数与实际行为选择严重不一致（34.4%的模型-子域对差异≥25分）。
- **自评与行为脱节**：已有证据表明自陈分数不与模型在具体情境中的选择和问题解决策略一致；同一模型在第一人称决策、给出建议、执行任务等不同交互角色中的人格表达是否一致从未被系统检验。
- **现有激活空间控制方法来源可疑**：之前工作从特质描述对比或最终响应token提取控制方向，但无法区分该方向编码的是真正的行为模式机制还是仅编码了输出的表面形式——是否存在"特质漂移"（trait drift）尚未检验。

## 核心贡献（创新点）
- **提出了情境化B-data框架**：将LLM人格从抽象特质标签操作化为具体情境中的选择、建议与任务行为，构建3,200个覆盖20个行为模式×4种prompt register的对比探针。*与已有工作的本质区别：不再使用自陈问卷，而是以可观测的行为选择作为人格测量的基础。*
- **引入了行为模式轴（BMA）及其分层提取方法**：证明行为模式可通过激活空间方向实现因果可控，且有效控制集中在**行为控制层带（BCL bands）**。*与已有激活空间控制工作的本质区别：BMA从结构化行为场景中的中间推理与最终响应两个不同source提取，揭示了thought-derived BMA-T比response-derived BMA-R更干净、更少特质漂移。*
- **发现了跨模型、跨scale的BCL泛化规律并量化了特质漂移现象**：Llama模型BCL峰值在归一化深度0.26–0.28，Qwen随规模加深（0.41→0.51），Gemma在中深度（0.44–0.49）；单一BMA在四个register上均实现强控制（从21.3%基线到82.7%/9.6%两端点）。*与已有工作的本质区别：首次系统比较BMA来源对控制质量的影响，证明仅靠响应激活提取会混入输出风格导致的机制替换。*

## 方法详解
**框架三阶段**：①构建情境化行为探针（situated behavioral probes）；②从模型行为中估计20维行为剖面（behavioral profile）；③沿BMA进行激活干预，检验因果可控性。

**行为模式对比探针设计**：基于BFI-2C、DOSPERT、HEXACO、GCS、UPPS五个经过验证的心理测量子域，构建20个行为模式，每个模式包含两个对立的行为模式（risky pole vs safe pole），横跨四种prompt register：first-person choice、daily advice、task advice、task execution。每种子域-register单元40个探针，共3,200个。关键设计：两极选项均具合理性（plausible & competent），避免社会期许偏差；使用exclusion clauses（"the center is X, not Y"）保持子域边界清晰。

**BMA提取（两种source）**：
- **BMA-R（response-derived）**：从低/高极最终响应token的激活差值提取
$$\tilde{v}_{\ell}^{R} = \frac{1}{N}\sum_{i=1}^{N}\left(\bar{h}_{\ell}^{R}(s_i, y_{i,R}^{\text{low}}; S_i) - \bar{h}_{\ell}^{R}(s_i, y_{i,R}^{\text{high}}; S_i)\right)$$
- **BMA-T（thought-derived）**：从低/高极中间行为推理（rationale）的激活差值提取
$$\tilde{v}_{\ell}^{T} = \frac{1}{N}\sum_{i=1}^{N}\left(\bar{h}_{\ell}^{T}(s_i, y_{i,T}^{\text{low}}; S_i) - \bar{h}_{\ell}^{T}(s_i, y_{i,T}^{\text{high}}; S_i)\right)$$
两轴均归一化：$v_{\ell}^{z} = \tilde{v}_{\ell}^{z} / \|\tilde{v}_{\ell}^{z}\|_2$。

**激活干预**：在推理时对选定层的激活加上缩放后的BMA向量：$h_{\ell}^{(t)} \leftarrow h_{\ell}^{(t)} + c v_{\ell}^{z}$，在prefill和decoding阶段均施加。

**有效控制度量**：定义clean directional range $S_{m,d,\ell}$，要求系数区间内mean和max未知率均<1%，度量目标极选择率的方向性变化范围$A_\ell(c^+) - A_\ell(c^-)$。

**BCL带识别**：搜索连续层区域，在其上所有20个子域的clean directional range均较高；不同模型的BCL位置随规模和架构变化。

## 实验与结果
**模型范围**：Llama-3.1-8B/70B-Instruct、Qwen2.5-7B/14B/32B-Instruct、Gemma-2-2B/9B-it，以及两个Llama-3.1-8B LoRA微调适配器（good/bad medical data）。

**RQ1结果——行为剖面与问卷的巨大差距**：平均gap 22.7 percentage points；34.4%的模型-子域对差异≥25分；Negative Urgency子域gap最大（47.5分）。剖面在register内高度稳定（split-half correlation mean 0.933，Spearman-Brown校正后更高），但跨register仅部分保持形状（cross-register mean r=0.76，range 0.37–0.97）；两advice register最相似（mean r=0.89），first-person与task较不吻合（mean r=0.63）。

**RQ2结果——BMA可因果控制行为**：在Llama-3.1-8B中，有效控制集中在L08–L12（mean clean directional range 0.82–0.89），L14后降至<0.30。跨七模型的BCL峰值：Llama在归一化深度0.26–0.28，Qwen随规模加深（7B:0.41，14B:0.45，32B:0.51），Gemma在0.44–0.49。单一first-person BMA-T跨四个register均实现强控制（Table 2）：baseline 21.3% → 正端点82.7%、负端点9.6%，mean ∆A从79.6（in-register）到67.4（task register）。

**RQ3结果——BMA来源决定控制质量**：BMA-T与BMA-R在BCL处平均cosine仅0.37，编码不同机制。BMA-T对齐行为动机本身（如improvising and adapting），BMA-R混入了输出风格（如dismissive "I don't care" stance），导致特质漂移。 drift在Organization→low-effort dismissal、Recreational Risk→danger romanticization、GCS Yielding→obedience/people-pleasing、HEXACO Sincerity→generic warmth、UPPS Lack of Perseverance→broad laziness等子域中尤为显著。

**BMA几何性质**：20个BMA-T在Llama-3.1-8B BCL处pairwise cosine中位数仅0.13，85%<0.3，证明各模式占据大致独立方向。跨子域迁移实验显示：on-target mean range=77.9，norm-matched random direction mean range=9.1，说明方向估计自行为对比是关键。

## 相关工作脉络
- **LLM人格问卷测量工作**（Serapio-García et al. 2025; Huang et al. 2024; Sorokovikova et al. 2024）：直接对模型施测Big Five/MBTI/HEXACO等问卷，本文指出其结果高度不稳定且与真实行为脱节，主张以B-data替代S-data。
- **Persona Vectors**（Chen et al. 2025）：从响应级激活提取特质方向（谄媚、恶意、幻觉倾向），本文与其关键区别在于：Persona Vectors来源于trait-description prompts或final response tokens，而本文的BMA-T从中间推理提取，且在cross-register transfer和trait-drift诊断下更为干净。
- **Activation Addition / Representation Engineering**（Rimsky et al. 2024; Zou et al. 2025; Turner et al. 2024）：从对比prompt或response提取steering方向干预residual stream，本文将其应用于行为模式维度并揭示source（thought vs response）的质变差异。
- **激活工程用于人格诱导**（Allbert et al. 2025）；**MBTI与安全能力的关联**（Zhang et al. 2024）：本文扩展了"人格信号可被操控的内部方向"这一视角，但首次系统论证了行为模式机制与输出形式的解耦必要性。
- **自陈与行为不一致研究**（Han et al. 2025; Ai et al. 2024）：证明了LLM自陈分数不预测下游行为，本文用更丰富的3,200探针和跨register数据进一步量化了这一差距，并提供了因果控制方案。
- **Psycho-lexical I-data路线**（Müller & Sütterlin 2025; Suh et al. 2024）：从模型log-probability over trait adjectives或访谈式互动提取因子结构，本文指出这是词汇/形容词证据而非行为证据，二者在机制上根本不同。

## 局限性与未来方向
- **行为模式的心理测量学地位未充分确立**：20个子域源自人类心理测量工具的组合，虽经quality control但并未在新样本上做factor analysis以验证其结构性，且人类量表在LLM上不收敛CFA（Peereboom et al. 2025）暗示移植taxonomy本身可能有问题。
- **BMA仅在open-weight模型上验证**：所有实验基于Llama/Qwen/Gemma开源模型，对closed models（如GPT-4系列）的行为控制可行性未知；且仅测试了bfloat16精度下的单卡干预。
- **跨register transfer存在衰减**：第一person BMA在task register上mean ∆A=67.4，低于in-register的79.6，说明register-dependent shift确实存在，通用"一人格轴"的假设受限。
- **unknown rate的1%约束可能过严或过松**：clean directional range的定义依赖这一阈值，不同模型/子域的最优约束可能不同。
- **BCL的机制解释仍黑箱**：BCL集中在earlier-to-middle layers是经验发现，论文未深入解释为何这些层承载行为模式而非其他信息。

## 研究启发与可借鉴点
- **B-data框架可迁移至其他LLM行为研究**：情境化对比探针的设计原则（两极均合理、exclusion clauses避免机制混淆、separate scenario sets防止leakage）可直接用于价值观对齐、偏好工程、任务风格控制等方向。
- **Thought-derived vs Response-derived BMA的区分具有重要方法论价值**：在需要"干净"行为控制的场景（如安全干预、人格一致性维护），应优先从中间推理而非最终响应提取方向；这一区分对任何基于activation steering的工作均有参考价值。
- **BCL band的发现为分层干预提供先验**：在目标模型上做layer sweep前先聚焦earlier-to-middle region可大幅降低搜索成本；不同模型规模的BCL位置随规模加深（Llama 8B: L08, Qwen 32B: L32）的规律值得在更多模型上验证。
- **Cross-register transfer评估应成为BMA方法的标配**：本文证明单一BMA可跨四register工作，但未测试跨model-family transfer；未来可将BMA作为跨模型行为一致性的量化指标。
- **与团队方向的结合机会**：若团队关注LLM价值观对齐或安全微调，可借鉴B-data框架评估微调后行为剖面的register一致性变化（本文已含两个medical LoRA适配器作为proof-of-concept），或用BMA进行post-hoc行为审计。

## 关键术语表
- **Behavioral Mode（行为模式）**：在具体情境中可观察的选择或行动方式，区别于抽象特质标签，以risky/safe pole对立形式实例化。
- **Behavioral Mode Axis / BMA（行为模式轴）**：从对比行为轨迹的激活差值提取的归一化方向向量，用于在推理时施加线性干预以改变模型行为。
- **BMA-T / Thought-derived BMA**：从中间行为推理（rationale）token span提取的BMA，被认为更忠实于目标行为机制。
- **BMA-R / Response-derived BMA**：从最终响应token span提取的BMA，可能混入输出风格导致机制替换。
- **Behavioral Control Layer (BCL) Band（行为控制层带）**：模型中连续多层组成的区域，在该区域内BMA能产生干净的双向行为控制且unknown rate<1%。
- **Trait Drift（特质漂移）**：BMA成功将模型推向目标极选项，但驱动该选择的内部机制偏离了预期行为模式（如Organization→low-effort dismissal）。
- **Interaction Register（交互register）**：四种提示框架——first-person choice、daily advice、task advice、task execution——同一模型在不同register下行为剖面呈现系统性偏移。
- **B-data（行为数据）**：在结构化情境中观察到的模型实际选择/建议/任务行为，区别于自陈S-data和观察报告I-data。

## 可复现要素
- **数据集**：3,200个行为探针场景（含轴提取场景集与探针场景集两套独立生成），代码与数据已在 GitHub: https://github.com/lhz191/LLM-Behavioral-Personality 开源。
- **模型权重**：使用Llama-3.1-8B/70B-Instruct、Qwen2.5-7B/14B/32B-Instruct、Gemma-2-2B/9B-it等公开open-weight模型。
- **关键超参**：steering系数扫描范围因模型而异（Llama-8B: −5~+5 step 1；Qwen-32B: −100~+150 step 5；Gemma: −200~+200 step 20）；SAE训练：131,072 latents, k=64 active, 500M tokens, lr=1e-4, batch=2048, max context=1024；生成使用greedy decoding (do_sample=False), 900-token cap。
- **硬件**：NVIDIA H200 GPU (≈141 GB HBM)，PyTorch 2.9.1, Transformers 4.57.6。
