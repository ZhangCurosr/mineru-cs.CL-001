---
title: "Matched-Outcomes-Divergent-Gaze-How-Foveated-MLLMs-Search-Co"
source: https://arxiv.org/pdf/2608.16514v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:41:26"
field: "多模态大模型视觉认知评估"
keywords: ["Visual Search", "Foveated Vision", "Multimodal LLMs", "Eye Movements", "COCO-Search18", "Human-Model Alignment", "Scanpath Prediction"]
innovations: ["将类人搜索解耦为决策/找到/注视三维并发现模型在决策和找到轴匹配人类但注视轴显著偏离", "发现三个跨架构MLLM共享低熵大振幅高自洽非人类注视签名，证明单次通过非串行架构属性", "系统性退化sweep证明无运行点可同时达到类人首次命中率与类人最终成功率"]
benchmarks: ["COCO-Search18"]
---

# 论文速读：Matched-Outcomes-Divergent-Gaze-How-Foveated-MLLMs-Search-Co

## 一句话总结
该研究将三个通用多模态大语言模型（MLLMs）与人类眼动扫描路径在目标导向搜索任务上进行逐注视对比，发现模型在决策和找到目标方面匹配或超越人类，但其注视过程（低熵、大振幅、高度自洽的扫描路径）显著不同于人类，表明零样本MLLMs是单次通过的非串行架构，而非受限于视觉锐度。

## 研究问题与动机
- **核心问题**：当MLLMs被赋予与人类相同的焦中心（foveated）视觉输入时，其搜索行为是否与人类一致？
- **动机一**：当前研究常将MLLMs用作人类观察者的替代品，或将注意力对齐分数（模型注意力与人类注视的重叠度）视为模型"像人类一样看"的证据，这种假设依赖于行为或空间一致性来推断过程一致性。
- **动机二**：人类目标导向视觉搜索是串行的（serial process），视网膜中央凹只能解析~1-2°范围内的细节，目标必须在外周被粗略检测到后才能通过注视确认；而MLLMs是否具有类似的串行搜索瓶颈尚不清楚。
- **动机三**：答案对齐（answer-alignment）和显著图（saliency）指标仅计算在结果或时间坍缩的空间图上，可能无法捕捉搜索过程中的时序差异，因此这些指标不能认证模型具有类人视觉过程。

## 核心贡献（创新点）
- **提出三维解耦评估框架**：将"类人搜索"分解为决策轴（目标存在/缺席判断）、找到轴（注视是否到达目标及效率）和注视轴（眼动动力学特征），并在相同焦中心输入条件下定量比较三者，发现决策与找到轴上模型匹配或超越人类，但注视轴上显著偏离人类。
- **发现跨模型共享的非人类注视签名**：三个来自不同架构家族的MLLMs在人类匹配条件下共享一个低熵、大振幅、高度自洽（cross-seed ScanMatch 0.84/0.91/0.71，远超人类间一致性上限0.53）的扫描路径特征，表明这是单次通过非串行架构的属性而非个别模型的偏差。
- **证明无退化 regime 能恢复类人搜索**：系统性地 sweeping 合成模糊程度（gist-k 从8到128），发现当首次扫视命中率达到人类水平（TFP@1≈0.49）时，最终成功率已远低于人类（TFP-end≈0.93），表明不存在同时满足类人效率和类人成功率的运行点。
- **揭示现有评估指标的盲区**：论证答案对齐和显著图指标因仅计算在结果或时间坍缩空间图上，对时序过程差异不敏感，因此高分数是类人视觉的必要但非充分条件；零样本MLLMs适用于 outcome 和 spatial 问题但不适用于 process 和 temporal 问题。
- **提出非串行搜索器作为基线模型**：一个在无需串行瓶颈的情况下达到类人甚至优于人类结果的搜索系统，可作为隔离串行性相关行为的 null model。

## 方法详解
- **数据与场景**：使用 COCO-Search18 验证集，包含141个目标存在（target-present）和144个目标不存在（target-absent）场景，每场景有10条人类扫描路径；场景分辨率1680×1050 px，视角约54°×35°，角分辨率约30 px/deg。
- **焦中心渲染器（Foveation Bracket）**：采用确定性渲染器，在四个条件族（共9个条件）下施加焦中心约束：
  - **Sharp**：无模糊，全清晰
  - **Geisler-Perry (GP)**：基于 Geisler & Perry [15] 的锐度衰减函数 $f_c(e) = \frac{e_2 \ln(1/CT_0)}{\alpha(e + e_2)}$，是唯一天生校准到人眼锐度的条件
  - **Gaussian gist-k**：相同衰减形式但外围截止需求缩放 k∈{8,16,24,32,48,128} 倍，是人工合成退化
  - **Crop**：仅保留~2.5°半径的中央凹圆盘
- **搜索循环（Search Loop）**：每个 episode 从强制中央注视开始；每步模型接收当前注视点的渲染视图（保留之前 glimpse 在上下文中），返回单一指令：LOOK(x,y) 继续搜索、FOUND(x,y) 终止并判定存在、ABSENT 终止并判定缺席；无搜索策略施加，50次 glimpse 上限从不强制终止。
- **指令读取（Directive Readout）**：每次生成以单一指令行结尾，坐标归一化到 [0,1] 单位正方形，原点左上角，y 向下增长；仅消费最终指令行，前置推理文本不影响读取。
- **评估模型**：Qwen3.5-35B-A3B（35B MoE，3B 活跃，reasoning-tuned）、GLM-4.6V-Flash（≈9B，reasoning-tuned）、Gemma-4-E4B（≈4B，instruction-tuned，thinking disabled）。
- **决策轴度量**：一次 existence 准确性（present: 0.99/0.99/0.97，absent: 0.81/0.81/0.80）、信号检测 d'（人类 2.91，模型 3.84/4.23/3.14）、虚报率（0.19/0.19/0.20）。
- **找到轴度量**：目标注视概率 TFP by saccade，命中定义为注视点位于目标框 1° 内；报告 TFP@1（首次扫视命中率）和 TFP-end（最终成功率）；中位注视数 NFix。
- **注视轴度量**：7个 per-scanpath 统计量（saccade amplitude、entropy、refixation、length、center bias、turning angle、coverage）以 Cliff's δ 与人类比较；ScanMatch 序列相似度（agent↔human、agent↔agent cross-seed、human↔human ceiling 0.53）；5-statistic PCA（PC1 45% 方差：scanpath length 和 gaze entropy；PC2 30% 方差：saccade amplitude vs refixation）。
- **统计方法**：线性混合效应模型（交叉 scene 和 rater 随机截距），Cliff's δ 效应量，95% CI 排除零视为显著。

## 实验与结果
- **总实验规模**：3 模型 × 285 场景 × 5 seeds × 9 条件 = 38,475 episodes，全部完成无排除。
- **决策轴结果（GP 条件）**：
  - 一次 existence 准确性：target-present 0.99/0.99/0.97，target-absent 0.81/0.81/0.80
  - d' 灵敏度：Qwen 3.84、GLM 4.23、Gemma 3.14，人类 2.91
  - 虚报率（yes-bias）：0.19/0.19/0.20
- **找到轴结果（GP 条件）**：
  - TFP@1：Qwen 0.97、GLM 0.97、Gemma 0.80，人类仅 0.49
  - TFP-end：0.98/0.98/0.93，人类 0.93
  - NFix-TP 中位数：2/2/3，人类 3
  - NFix-TA（缺席搜索长度）：1/4/5，人类 5
- **注视轴结果（GP 条件，Cliff's δ vs 人类）**：
  - Gaze entropy：-0.67/-0.65/-0.27（模型更集中）
  - Saccade amplitude：+0.50/+0.61/+0.23（模型跳跃更大）
  - Center bias：-0.18/-0.31/+0.06（接近人类）
  - Cross-seed self-consistency (ScanMatch)：0.84/0.91/0.71，远超人类↔人类上限 0.53
- **退化 sweep 结果（gist-k）**：
  - 随着 k 增加（模糊加重），TFP@1 和 TFP-end 同步下降，无 regime 能同时达到类人首次命中率（0.49）和类人最终成功率（0.93）
  - 在 k=32（Qwen TFP@1≈0.42 接近人类）时 TFP-end 已降至 0.71
  - 重度退化下 refixation 上升，随后模型默认判定 absent
- **最强结果**：GLM-4.6V-Flash 在 TFP@1 上达 0.97（人类 0.49 的约 2 倍），且在 gaze entropy（δ=-0.65）和 saccade amplitude（δ=+0.61）上偏离最大；Gemma-4-E4B 最接近人类（entropy δ=-0.27，NFix 与人类无显著差异）。
- **统计稳健性**：mixed-effects 95% CI 排除零；TFP@1 对 hit tolerance ±0.5/1/1.5° 变化最大偏移 ≤0.095；temperature sweep 至 1.0 时 cross-seed determinism 仍远高于人类上限。

## 相关工作脉络
- **COCO-Search18 基准与人类搜索建模**：COCO-Search18 [7,8] 提供实验室人类注视数据，支撑 scanpath 预测器、逆强化学习、transformer 模型 [37,52,53]、对抗训练变体 [26,44]、domain-adapted 生成器 [25,27] 及理想观察者/Bayesian 搜索者 [5,45]；本文将其扩展到 MLLM 过程比较。
- **焦中心视觉作为建模约束**：Geisler-Perry 锐度衰减 [15] 是本研究的人类匹配渲染器基础；焦中心架构被研究为人类表征模型 [11]、场景识别中的中央-外围分工 [47]、可变分辨率下类人表征 [18,19]、直接搜索的焦中心检测器 [38]、焦中心 transformer [22]、场景理解时间预测 [48]、双 what/where 扫视选择 [10]、语义引导焦中心模型预测 COCO-Search18 扫描路径 [35,36]、以及部分可解析几何下的观看架构 [24]；Biologically constrained networks viewing scenes foveally 无需训练即可产生类人扫描路径 [40,55]——本文对此发现构成对照。
- **MLLM 与人类视觉比较**：认知范式研究 [6,12,14,34] 发现模型"see but do not perceive"；眼动注意力比较 [17,43,42,29] 显示注意力相似性与任务表现解耦 [43]；语言先验依赖批判区分空间与语义引导 [23,50]；VLMs 的串行缺陷从反应时间推断 [4]、人类易搜索任务上 VLMs  barely 优于 chance [2]、正确区域错误答案 [30]——本文的 right-answer/non-human-process 发现与之镜像但聚焦 fixation-by-fixation 过程。
- **Agentic 搜索训练 MLLMs**：添加主动感知以提升准确性的并行工作包括 LLM-guided search [51]、tree-based zoom [41]、reinforcement-learned focusing [1,32,33]、embodied 变体 [13,54]、chain-of-thought 可能退化 embodied searcher [56]；本文不提出 scanpath 预测器也不竞争准确率，而是 characterization general-purpose model 在固定焦中心约束下的探索行为。
- **计算资源分配而非锐度**：通过减少或合并视觉 token 以提升效率的 budget-allocation 机制 [3,57] 属于资源分配而非外周视觉建模，与本文焦点不同。

## 局限性与未来方向
- **未评估搜索训练 agentic 模型**：search-trained agentic、pointing-native 和 frontier closed-source 模型是最相关的下一步比较对象，但未纳入本研究。
- **跨模型趋势混杂**：三个模型在 architecture、training recipe、MoE sparsity 上均不同，交叉变量无法分离，因此跨模型趋势仅描述性报告。
- **Prompt 可能影响 refixation**：gaze signature 在单一 prompt 下测量，其中 glimpse memory 子句可能部分塑造 refixation 水平。
- **注视时长仅人类可用**：fixation durations 无法从模型获取，仅人类有。
- **人类↔人类跨 seed 基线缺失**：cross-seed 和 human inter-observer agreement 不是同一构念（COCO-Search18 无 matched human intra-observer baseline），determinism gap 基于 temperature-1.0 持久性作为佐证。
- **未来方向**：直接操控架构以验证 serial sampling 是缺失组件；比较 trained agentic models；开发 process-level 评估指标；构建引入串行采样的 causal test。

## 研究启发与可借鉴点
- **三维解耦评估框架可直接迁移**：将模型-人类比较从单一 accuracy 指标解耦为 decision/finding/gaze 三轴，适用于任何需要对齐人类行为的 AI 系统评估（如 robotics、embodied AI），尤其适合识别"正确结果但错误过程"的场景。
- **焦中心渲染器作为标准化 benchmark 工具**：Geisler-Perry 衰减函数的 byte-identical harness 设计（harness/prompt/renderer 跨模型完全一致）确保行为差异仅归因于模型本身，可作为 MLLM 视觉能力比较的通用实验范式。
- **Cross-seed self-consistency 作为模型确定性度量**：利用 multiple seeds 计算 agent↔agent ScanMatch 可量化模型的内在 determinism，与 human↔human ceiling 对比可揭示非人类特性，该方法可推广到任何需要评估模型稳定性的任务。
- **Degradation sweep 揭示性能-过程 tradeoff**：系统性遍历退化程度（gist-k）发现 TFP@1 与 TFP-end 同步下降、无 regime 能同时达到类人效率与成功率，此方法可用于诊断模型在何种输入质量下开始"失败"而非"搜索"。
- **Null model 概念启发新评估范式**：非串行搜索器作为基线可隔离串行性相关行为，启示团队在视觉搜索研究中构建 parallel single-pass baseline 以量化 serial bottleneck 的行为代价。

## 关键术语表
- **Foveated vision / 焦中心视觉**：模拟人眼视网膜中央凹高锐度、外周低锐度的视觉特性，通过锐度衰减函数在注视点附近保持清晰、远离注视点快速退化。
- **Scanpath / 扫描路径**：目标导向视觉搜索中按时间顺序排列的注视点序列，反映搜索过程的时序结构。
- **TFP@1 / 首次扫视目标命中率**：强制中央注视后第一次扫视即命中目标的比例，衡量搜索效率。
- **Cliff's δ / Cliff 优势统计量**：非参数效应量指标，计算 A>B 的对数减去 A<B 的对数再除以总对数，范围 [-1,1]，|δ|>0.33 视为非平凡效应。
- **ScanMatch / 扫描路径匹配**：基于 Needleman-Wunsch 全局对齐的扫描路径相似度度量，考虑注视顺序和空间接近性，归一化到 [0,1]。
- **Gaze entropy / 注视熵**：将显示区域划分为 14×9 网格，计算各网格注视比例的信息熵，衡量注视的空间分散程度。
- **Cross-seed self-consistency / 跨种子自洽性**：同一模型在不同 random seed 下产生的扫描路径之间的相似度，反映模型的内在确定性。
- **Single-pass non-serial architecture / 单次通过非串行架构**：并行视觉编码器一次性解析清晰帧并直接跳向目标的架构，与人类的串行逐注视检查形成对比。

## 可复现要素
- **数据集**：COCO-Search18 [7]（公开，链接：https://coco-search.github.io/），使用验证集冻结类别分层子集（141 target-present + 144 target-absent）
- **代码/权重**：论文未明确声明开源仓库，但 prompt、renderer 和 harness 细节在 supplementary material Sec.S1-S12 完整给出；三模型（Qwen3.5-35B-A3B、GLM-4.6V-Flash、Gemma-4-E4B）均为公开模型
- **关键超参**：sampling temperature 0.6、5 seeds per condition、50-glimpse cap、hit tolerance 1°（默认）、gist-k ∈ {8,16,24,32,48,128}、rendering resolution 1680×1050 downscale to 1024px max side、grid 14×9 for entropy
