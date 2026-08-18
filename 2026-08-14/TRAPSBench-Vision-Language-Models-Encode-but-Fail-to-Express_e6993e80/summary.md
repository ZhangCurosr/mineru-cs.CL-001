---
title: "TRAPSBench-Vision-Language-Models-Encode-but-Fail-to-Express"
source: https://arxiv.org/pdf/2608.13167v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:36:50"
field: "多模态大模型可靠性与可解释性"
keywords: ["Vision-Language Models", "Epistemic Calibration", "Abstention", "TRAPSBench", "Activation Steering", "Physical Reasoning", "Mechanistic Interpretability"]
innovations: ["提出TRAPSBench与PECS联合度量VLM在物理视频证据缺失时的选择性拒答能力", "通过线性探针与激活引导证明VLM内部编码认识论不确定性但输出层存在表达瓶颈", "揭示文本不可答检测显著优于视觉信息缺失，且CoT推理可能覆盖内部疑虑加剧虚构"]
benchmarks: ["TRAPSBench"]
---

# 论文速读：TRAPSBench: Vision-Language Models Encode but Fail to Express Epistemic Restraint

## 一句话总结
论文提出程序化视频基准 TRAPSBench 与新指标 PECS，系统检验 VLM 在物理证据不足时“主动拒答”的能力；发现模型内部确实编码了认识论不确定性，但自回归输出路径存在强抑制，导致自发克制表现极差（最高 PECS 仅 0.292），瓶颈在表达而非感知。

## 研究问题与动机
- **核心问题**：在视觉证据被遮挡、混沌截断或问题本身不可答时，VLM 能否正确识别并选择性拒答（epistemic restraint）？
- **现有基准不足**：CLEVRER、IntPhys、PhysBench 等物理推理基准仅评估预测正确率，未设计证据缺失场景，无法衡量模型的拒答能力。
- **指标缺陷**：传统 abstention recall 会奖励无条件拒答，accuracy 完全忽略认识论维度，二者均无法刻画“该答则答、不该答则拒”的校准能力。
- **机制未明**：模型拒答失败是因为真正缺乏证据感知，还是内部有信号却无法正常输出？现有工作多停留在文本或单图，缺乏视频物理场景下的因果机制验证。

## 核心贡献（创新点）
- **TRAPSBench 与 PECS 联合发布**：构建 1,404 对匹配的物理控制/虚空视频对，配套 PECS 指标（准确率 × Youden's J 统计量），同时惩罚“永远拒答”与“永远乱答”的退化策略。
- **揭示表征-输出鸿沟（Representation-Output Gap）**：通过线性探针与激活引导双重验证，证明模型内部能解码可答/不可答区分（AUROC 最高 0.91），但自回归输出层将其抑制。
- **发现两类失败模式不对称性**：文本层面不可答比视觉信息缺失更易被检测（中位数差距约 4×）；Chain-of-Thought 推理非但不提升校准，反而可能覆盖内部疑虑、加剧虚构（confabulation）。
- **因果机制解剖与跨架构复现**：首次对视觉认识论不确定性做激活引导干预；oduction-family 方向具有跨域通用性，chaotic 方向近正交且域特定，且结果在 Qwen、Gemma、LLaVA 三大家族中一致复现。

## 方法详解
- **最小视频对范式**：基于 MuJoCo 刚性体物理引擎程序化生成 matched control/void 对，控制对含完整信息，虚空对仅引入单一修改（遮挡、截断或问题改写）使答案不可推导。
- **三类不确定性分类法**：
  1. **Occlusion (N=202)**：不透明遮挡物阻断关键视觉信息；
  2. **Chaotic Sensitivity (N=500)**：对初值极度敏感的系统（弹珠、骰子、跷跷板等），视频在结果解析前截断；
  3. **Ill-Posed Questions (文本虚空)**：复用控制视频，但问题含虚假前提或不可观测细节，用于对比文本/视觉不可答的检测差异。
- **PECS 指标**：$$\mathrm{PECS} = \mathrm{Acc} \times \max\left(0,\ \mathrm{AbsRec} - \mathrm{FalseAbs}\right)$$，其中 AbsRec−FalseAbs 即 Youden's J 统计量，要求模型在可答样本上准确、在不可答样本上选择性拒答，且 false abstention 会直接拉低得分。
- **评测流程**：模型输出自由文本 → 3 模型 Judge 面板（Gemini 3 Flash、Qwen3-VL Instruct、Claude 4.6 Opus）多数投票 → 严格基于文本判读拒答与正确答案（不访问视频，避免 judge 视觉能力干扰）。
- **线性探针（Linear Probe）**：冻结模型各层隐藏状态，训练 L2 正则 LR 分类器预测 void/control，报告最佳层 AUROC；跨数据集/跨模态/跨领域迁移验证信号通用性。
- **激活引导（Activation Steering）**：提取单层虚空方向 $\mathbf{v}_\ell = (\bar{\mathbf{h}}_\ell^{\mathrm{void}} - \bar{\mathbf{h}}_\ell^{\mathrm{control}})/\|\cdot\|$，在每一步自回归生成时叠加/减去 $\alpha \mathbf{v}_\ell$，因果性地诱导或压制拒答行为。

## 实验与结果
- **评测规模**：16 个 VLM、5 个家族、3 种提示 regime（Standard / Guided / JSON）、4 个数据集，共约 19.4 万评测对、38.7 万次模型调用，3 次独立重运行取均值。
- **主要结果（标准 regime）**：最佳 PECS 为 Gemini 2.5 Flash（0.292），其余模型普遍低于 0.2；**引导 regime** 下最佳为 Gemini 3.1 Pro R-Low（0.568），但仍未达饱和。
- **提示干预效应**：Guided 提示使 14 个主流视频原生模型的 AbsRec 中位数提升 1.9×（最高 2.8×），准确率基本持平（±2pp），证明 latent 能力广泛存在。
- **文本 vs 视觉不对称**：所有 16 个模型在文本不可答上的 AbsRec 均高于视觉缺失（混沌集差距 3–25×，图像模型最高 197×，中位数约 4×）。
- **推理链（CoT）双刃剑**：Gemini Flash 启用思考后 AbsRec 提升 4–13pp；Qwen3-VL Think 思考 trace 中 doubt 率最高（24%），但最终输出仍覆盖疑虑并生成更多虚构答案，标准 regime 下 AbsRec 反而下降。
- **探针跨集迁移**：Qwen3-VL-8B 跨数据集 AUROC 最高 0.91；限定在模型已**错误虚构**的样本上（Cf→Cf）仍保持 0.57–1.00 的高迁移率，证明内部信号独立于行为输出。
- **激活引导因果验证**：occlusion-family 方向跨域迁移强（α=10 时控制拒答达 75%），chaotic 方向近正交（cos ≤ 0.08）且域特定；跨模态也可迁移（文本 oc\_ip 方向对视觉 ch 场景可达 100% 控制拒答）。
- **开放权重复现**：Gemma 4 E4B 与 LLaVA-Video-7B 在标准/引导 regime 下 PECS 极低（0.160 / 0.215 与 0.079 / 0.101），但探针与引导结果与 Qwen 一致，排除单一架构假象。

## 相关工作脉络
- **物理推理基准**（CLEVRER、IntPhys、PhysBench、Morpheus）：聚焦“能否预测”，未设计证据缺失对照组；本文通过 matched pairs 直接隔离认识论判断能力。
- **不可答问题基准**（UNK-VQA、VisionTrap、TUBench、CertainlyCertain）：多基于单图语义扰动，且 CWA 需 logit 访问；本文扩展到视频物理动态，采用纯文本黑盒评测，且无生成伪影。
- **NLP 拒答评测**（SQuAD 2.0、AbstentionBench）：仅覆盖文本不确定性与语言级元认知；本文首次将拒答评估推向视觉-物理多模态。
- **机制可解释性工作**（truth/probe、activation steering for refusal）：主要在文本域验证事实性与拒绝行为；本文将其拓展至视觉认识论不确定性，并给出跨域/跨模态的因果证据。
- **本文定位差异**：不止于测评分数，而是通过“探针+引导+退化策略检验”三重证据链，锁定瓶颈位置在输出通路而非表征层，为后续输出阶段干预提供明确靶点。

## 局限性与未来方向
- **合成环境偏简单**：MuJoCo 刚性体物理为理想化设定，复杂度低于真实世界视频，结论外推需谨慎。
- **闭源模型机制受限**：无法对闭源 API 模型进行探针与引导，机制结论依赖开放权重家族，可能存在架构特异性。
- **Prompt gating 与方向几何具家族差异**：单向门控在 Qwen 中完全、Gemma 中部分、LLaVA 中不存在；虚空方向几何在不同模型中结构不一致，不宜泛化为通用规律。
- **未来方向**：扩展至软体/流体物理、真实世界视频；针对 void 样本进行微调；探索生成式 rollout 下的认识论校准；研究输出端干预（如解码时干预、奖励塑形）以闭合表征-输出鸿沟。

## 研究启发与可借鉴点
- **Benchmark 设计范式**：采用“最小扰动 matched pair”隔离单一变量（信息缺失类型），是刻画模型某项隐性能力（如拒答、校准、反事实推理）的高效范式。
- **指标构造技巧**：PECS 利用 Youden's J 统计量将“选择性”与“准确性”耦合，并显式钳制 J≤0 的情况，彻底堵死永远拒答/永远回答的作弊路径，可直接移植到 calibratation 类评测。
- **机制验证流水线**：“线性探针（相关性）→ 限制 confabulated 样本（排除行为混淆）→ 激活引导（因果性）”构成一套可复用的黑盒/白盒 bridging 协议，适合用于验证其他模型的 latent capability。
- **推理时的廉价干预**：Guided prompting 与 JSON 结构化输出均可作为轻量级“输出门控”，在不调参情况下显著提升拒答率；对安全关键部署，可在推理阶段通过提示/格式约束弥补训练不足。
- **跨家族对比价值**：同时评测闭源 API 与开源多架构模型，并用统一 protocol 交叉验证，能有效区分“数据/算力优势”与“机制共性”，对团队选型与基线建设极具参考意义。

## 关键术语表
- **TRAPSBench**：Testing Restraint in Ambiguous Physical Scenarios，基于 MuJoCo 程序化生成的物理视频评测基准。
- **PECS**（Penalized Epistemic Calibration Score）：惩罚性认识论校准得分，= 准确率 × max(0, 拒答召回 − 假拒答率)，联合评估能力与选择性。
- **Void / Control Pair**：虚空-控制视频对，控制对信息完整可答，虚空对仅含一处信息缺失或逻辑悖论。
- **Epistemic Restraint**：认识论克制，指模型在证据不足时主动识别并选择不回答的能力。
- **Linear Probe**：在冻结隐藏状态上训练的轻量分类器，用于解码模型内部是否编码了特定隐信息。
- **Activation Steering**：向自回归生成步的隐藏状态叠加/减去 Learned Direction Vector，以因果方式干预模型行为。
- **Confabulation**：虚构，指模型在证据不足时自信生成未观察到或未推导出的物理事实。
- **Youden's J Statistic**：约登指数，衡量二分类器真正例率与假正例率之差，此处量化拒答的选择性判别力。

## 可复现要素
- **数据集**：TRAPSBench（1,404 对 control/void 视频与问题），已公开于 https://github.com/facebookresearch/TRAPS-Benchmark（CC BY-NC 4.0）。
- **代码/权重**：评测代码与系统提示词开源；模型通过公共 API 或 HuggingFace 公开权重评估（Qwen3-VL-8B、Gemma 4 E4B、LLaVA-NeXT-Video-7B）。
- **关键超参**：API 模型 temperature=0.8、top-p=0.95、max_completion_tokens=8000、5 FPS 输入（≤25帧）；本地 8B bf16 贪心解码（16 均匀帧）；重放模型 bf16、32 帧@2 FPS、max_tokens=512；Judge 面板使用 Gemini 3 Flash/Qwen3-VL Instruct/Claude 4.6 Opus 多数投票，temperature=0.8、max_tokens=512。
