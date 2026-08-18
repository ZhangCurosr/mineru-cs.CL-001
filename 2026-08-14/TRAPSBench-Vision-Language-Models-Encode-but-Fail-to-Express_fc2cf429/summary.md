---
title: "TRAPSBench-Vision-Language-Models-Encode-but-Fail-to-Express"
source: https://arxiv.org/pdf/2608.13167v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:36:46"
field: "视觉语言模型可靠性与认知校准"
keywords: ["vision-language models", "epistemic calibration", "abstention", "physical reasoning", "activation steering", "probing", "uncertainty"]
innovations: ["TRAPSBench: 首个程序化生成视频物理基准系统评估VLM选择性回避能力", "证明VLM内部编码认知不确定性但输出端被抑制（表征-输出鸿沟）", "PECS: 同时约束正确率与选择性回避的惩罚性认知校准指标"]
benchmarks: ["TRAPSBench"]
---

# 论文速读：TRAPSBench-Vision-Language-Models-Encode-but-Fail-to-Express

## 一句话总结
本文提出了 TRAPSBench——一个基于 MuJoCo 程序生成的 1,404 对匹配物理视频基准，用于系统评估视觉语言模型（VLMs）在证据不足时能否选择性回避回答；同时引入惩罚性认知校准分数（PECS）指标。研究发现：16 个 VLMs 内部确实编码了"不可答"信息（线性探针 AUROC 最高 0.91），但自发表达抑制能力极差（最佳 PECS 仅 0.292），瓶颈在于**表达而非感知**。

## 研究问题与动机
- **核心问题**：VLM 在视觉证据被遮挡、混沌或问题本身无解时，能否识别自身知识边界并选择性 abstain？现有物理推理基准（CLEVRER、IntPhys、PhysBench、Morphheus 等）均不评估此能力。
- **度量缺陷**：已有 abstention 相关指标存在盲区——abstention recall 奖励无差别回避（永远回避得 100% recall），accuracy 忽略认知维度，简单乘积（Acc × AbsRec）不惩罚误回避。
- **表征-输出鸿沟**：VLMs 是否仅在输出阶段丢失了内部的认知不确定信号？这一问题尚无人通过因果干预手段系统验证。
- **模型可靠性需求**：自主智能体在感官信息无法支撑确定性预测时（物体被遮挡、轨迹混沌、查询不合理），正确的行为应是选择性 abstain，这对可靠 AI 部署至关重要。

## 核心贡献（创新点）
1. **TRAPSBench + PECS 指标**：程序化生成 1,404 对匹配可答/不可答物理视频对（三类不确定性分类：遮挡、混沌敏感性、问题表述不当），配合惩罚性认知校准分数（PECS = Acc × max(0, AbsRec − FalseAbs)），对"永远回避"和"从不回避"策略均得分为零。
   - *区别*：现有基准（如 UNK-VQA、VisionTrap）仅在单图上做语义扰动构造不可答题；TRAPSBench 面向视频物理动态，通过 MuJoCo 对比对消除编辑伪影，且 PECS 同时约束正确率与选择性回避。

2. **表征-输出鸿沟的系统性证明**：通过三种互补手段证实——引导提示可激活 latent abstention（中位数 1.9× 提升）、线性探针跨物理域 AUROC 最高 0.91、单层激活 steering 因果控制回避行为，三者均在 Qwen、Gemma、LLaVA 三个无共享训练管线家族中复现。
   - *区别*：此前 probing/steering 工作仅覆盖文本领域的 truth/refusal，本文为首次证明 VLMs 在视觉域编码可转移的认知信号而输出端却被抑制。

3. **失败模式非对称性发现**：模型检测文本不可能性的能力约比视觉信息缺口强 4 倍（混沌分割上最高 197×）；chain-of-thought 推理对校准的影响因家族而异——Gemini Flash Think 通过抑制不确定输出改进了 abstention，Qwen3-VL Think 却压制自身怀疑、过度 confabulate。
   - *区别*：此前研究未系统对比同一模型在不同模态（文本 vs 视觉）不确定性下的检测差异，也未考察 CoT 推理对物理域认知校准的双面效应。

4. **因果机制分析揭示方向几何**：cross-dataset steering 表明遮挡类 void 方向编码域通用的"证据缺失"信号（cos ≈ 0.19–0.25 聚类），而混沌方向域特异且近正交（cos ≤ 0.08）；跨模态方向（语言→视觉）同样有效。
   - *区别*：将 Steering 从文本事实性/情感/拒绝控制，拓展到视觉认知不确定性的因果干预，首次建立表征与行为间的因果链路。

## 方法详解
- **最小视频对范式**：用 MuJoCo 物理模拟器为每个场景生成一对视频——control（信息完整、输出确定）和 void（单一扰动使结果从视觉上不可计算），包括：① 遮挡（N=202）——不透明遮挡物阻挡关键信息；② 混沌敏感性（N=500）——对初值极端敏感的确定性系统，在结果解析前截断视频；③ 不当问题（Ill-Posed，每类各 N=202/500）——复用 control 视频但构造仅凭文本即可判定的无解问题。
- **PECS 公式**：$$\mathrm{PECS} = \mathrm{Acc} \times \max(0,\ \mathrm{AbsRec} - \mathrm{FalseAbs})$$其中 AbsRec − FalseAbs 即 Youden's J 统计量，对无差别回避（J=0）和从不安避（J=0）均惩罚为零，六类退化策略（永远回避、从不回避、随机、完美回答者等）均得 PECS=0。
- **评估管线**：输入（视频, 问题）→ 模型自由文本输出 → 三模型 judge panel（Gemini 3 Flash + Qwen3-VL Instruct + Claude 4.6 Opus，多数投票）评分——仅基于文本与 MuJoCo ground truth 比对正确性、或检测 nuanced 回避表达，judge 不看视频以避免视觉感知混淆。
- **线性探针**：冻结 Qwen3-VL-8B 全部 37 层隐藏状态，在 L2-regularized LR 上训练"void vs control"分类器，报告最佳层 AUROC（阈值无关，对校准偏移鲁棒），训练集不含目标数据集标签。
- **激活 Steering**：在 layer 20 提取 void 方向 $\mathbf{v}_\ell = (\bar{\mathbf{h}}_\ell^{\mathrm{void}} - \bar{\mathbf{h}}_\ell^{\mathrm{control}})/\|\cdot\|$，在自回归每一步叠加：$\mathbf{h}_\ell^{(t,i)} \gets \mathbf{h}_\ell^{(t,i)} + \alpha_{\mathrm{eff}} \cdot \mathbf{v}_\ell$，控制样本加 +α 诱导回避，void 样本加 −α 强制 confabulate，$\alpha \in \{0,2,5,10\}$ 扫描。

## 实验与结果
- **基准规模**：1,404 对 matched 视频对（202 遮挡 + 500 混沌 + 各 202/500 不当问题变体），跨 4 个子任务。
- **16 个 VLMs，5 个家族**：Gemini（6）、Qwen3-VL（3）、GPT-5（5）、Gemma 4 E4B、LLaVA-Video-7B。
- **主结果（Table 1）**：
  - 标准提示下最佳：**Gemini 2.5 Flash**，PECS = **0.292**（Acc 63%，AbsRec 49%，FalseAbs 1.5%）。
  - 引导提示下最佳：**Gemini 3.1 Pro R-Low**，PECS = **0.568**（AbsRec 86%，FalseAbs 4.9%）。
  - 引导提示中位数提升：AbsRec 1.9×（1.4–2.8×），Acc 变化不大。
  - 开放权重复现家族（Gemma 0.215、LLaVA 0.101）仍远未饱和，证明低抑制是共性而非单架构 artifact。
- **文本 vs 视觉非对称性（Table 2）**：所有 16 模型均高于对角线；GPT-5 Medium 在混沌分割上视觉 AbsRec 仅 0.1% vs 文本 14.6%（197× gap）；总体中位 gap ≈ 4×。
- **CoT 推理的双刃效应（Figure 4）**：Gemini Flash Think 改进 AbsRec（+4–13pp）；Qwen3-VL Think 反而降低（Chaotic Ill-Posed：Instruct 82.3% vs Think 71.7%），尽管 Think 有最高 doubt 率（24%），仍压制自身怀疑。
- **探针跨域转移（Table 3）**：
  - 跨模态（同物理，视觉↔文本）：AUROC 最高 1.000（LLaVA ch→ch.ip）。
  - 跨域（同 void 类型，遮挡↔混沌）：AUROC 最高 0.913（Qwen oc→ch.ip）。
  - 全样本 & Cf→Cf 限制（仅 confabulated 样本）差异 ≤ 0.03，证明信号在模型失败样本上依然存在。
- **Steering 因果验证（Table 4, Figure 5）**：
  - Qwen3-VL-8B：标准 prompt 下 +α 诱导回避几乎失效（0→1%），但在 guided 推理下生效（11→54%），表明**单向输出门控**。
  - 遮挡方向跨域强转移（75%），混沌方向弱（15%）；cos 矩阵显示 oc 类方向成团（0.19–0.25），ch 近正交（≤ 0.08）。
  - 跨模态：oc_ip→ch 达 100% 控制回避，强于同模 oc→ch（90%）。

## 相关工作脉络
1. **物理推理基准**（CLEVRER、IntPhys、PhysBench、Morphheus）：评估模型预测什么，而非模型是否能认识到"无法预测"——TRAPSBench 衡量后者，且通过匹配对比对消除捷径推理。
2. **Uncertainty & Abstention**（Kendall & Gal 2017、Geifman & El-Yaniv 2017/2019）：依赖模型内部（logits/dropout）度量；TRAPSBench 的 PECS 是纯文本判定的黑盒友好度量。
3. **NLP 不可答问题**（SQuAD 2.0、AbstentionBench）：形式化了文本不可答；TRAPSBench 将其扩展到视觉物理域，并通过 procedural 生成避免污染。
4. **LLM 元认知**（Kadavath 2022、Xiong 2024、Tian 2023）：显示 LLM 有有限 self-knowledge 且提示可改善置信度表达；本文在视频域验证内部表征是否存在但输出被抑制。
5. **Probing & Steering**（Burns 2023、Marks & Tegmark 2024、Arditi 2024、Rimsky 2024）：揭示 LLM 内部 truth/refusal 结构并可通过方向向量因果干预；本文为首次将此方法体系应用于**视觉认知不确定性**，且提供跨域/跨模态转移证据。
6. **不可答视觉问题**（UNK-VQA、VisionTrap、TUBench、CertainlyUncertain）：均在单图上通过语义扰动构造，可能引入编辑伪影；TRAPSBench 基于 MuJoCo 对比对无伪影，且 PECS 兼容黑盒 API。

## 局限性与未来方向
- **物理范围局限**：仅 rigid-body 动力学，尚未覆盖 soft-body、流体、真实世界视频。
- **机制分析仅限开放权重**：闭源模型无法 probing，但行为签名在全部 16 模型中一致。
- **Prompt gating 和方向几何因家族而异**：单向门控仅在 Qwen 显著，Gemma 部分，LLaVA 无；方向几何在三家族间不一致，属模型依赖发现。
- **Sim-to-real gap**：程序化场景是简化测试用例，需与真实世界基准互补验证。
- **未来方向**：扩展至新物理域、soft bodies、真实视频；针对 void 视频的微调训练；生成式 rollout 评估；更细粒度的输出阶段干预设计。

## 研究启发与可借鉴点
1. **最小对比对范式**（minimal video pair）可移植到其他领域的"能力有无分离"评估——构造控制/扰动对，仅改变一个因素使结果不可计算，精准隔离认知边界识别能力。
2. **PECS 度量设计思想**（乘积 + Youden's J 统计量 + 零惩罚退化策略）可推广至其他需要同时评估"正确率"和"选择性回避"的场景（如安全敏感型 AI、医学诊断辅助）。
3. **Cf→Cf 探针限制**（仅用模型已 confabulate 的样本训练/测试）是排除行为混淆的强力技巧——证明内部信号独立于已有输出行为，值得在类似研究中采用。
4. **Steering 方向几何分析**（cosine 聚类 + 跨域/跨模态转移）可复用于诊断其他能力（如 truthfulness、refusal）的表征可转移性。
5. **CoT 对校准的双面效应发现**提示：团队在引入 reasoning 能力时，应同步评估其对不确定性表达的破坏风险，避免"更聪明地胡说"。

## 关键术语表
- **TRAPSBench**：Testing Restraint in Ambiguous Physical Scenarios，基于 MuJoCo 的程序化视频基准，含 1,404 对匹配的可答/不可答物理场景。
- **PECS**（Penalized Epistemic Calibration Score）：惩罚性认知校准分数，PECS = Acc × max(0, AbsRec − FalseAbs)，对永远回避/从不回避均给零分。
- **Epistemic Restraint**（认知抑制）：在证据不足以支持确定预测时主动选择 abstain 的能力。
- **Minimal Video Pair Paradigm**（最小视频对范式）：control 与 void 视频仅一个因素影响输出确定性，用于精准隔离认知边界识别。
- **Void Direction**（void 方向）：从冻结模型隐藏状态中提取的 void−control 均值差向量，叠加到生成过程中可因果诱导/抑制 abstention。
- **Confabulation**（虚构）：模型在无足够证据时自信编造答案的行为，87–99% 包含 hallucinated premise（视觉事实捏造）。
- **Prompt Gating**（提示门控）：标准提示对模型输出形成单向抑制（阻止 abstention 表达），引导提示可打开此门。
- **Cf→Cf 限制**（Confabulation-filtered probe）：探针仅在模型已 confabulate 的 void 样本上训练/测试，排除行为混淆。

## 可复现要素
- **数据集**：TRAPSBench 已公开，GitHub：https://github.com/facebookresearch/TRAPS-Benchmark（CC BY-NC 4.0）。
- **代码/权重**：评估提示词已随论文附录公开；部分模型本地运行（Qwen3-VL-8B、Gemma 4 E4B、LLaVA-NeXT-Video-7B from Hugging Face）；闭源 API 模型通过官方接口评估。
- **关键超参**：API 模型 temperature=0.8，top-p=0.95，max_completion_tokens=8,000，5 FPS/最多 25 帧；Qwen3-VL-8B 本地 bf16，16 uniform frames，greedy；Gemma/LLaVA 本地 bf16，32 frames @2 FPS，max_new_tokens=512；Steering α∈{0,2,5,10}，layer 20（Qwen）/35（Gemma）/28（LLaVA）。
