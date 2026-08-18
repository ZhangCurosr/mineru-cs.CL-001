---
title: "Matched-Outcomes-Divergent-Gaze-How-Foveated-MLLMs-Search-Co"
source: https://arxiv.org/pdf/2608.16514v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:42:14"
---

# 论文速读：Matched-Outcomes-Divergent-Gaze-How-Foveated-MLLMs-Search-Co

## 一句话总结
论文在 COCO-Search18 数据集上，通过人类匹配的 foveated 输入驱动三个通用 MLLM 逐次注视搜索，发现模型在决策正确性和目标获取效率上匹配或超越人类，但其注视时序动态呈现低熵、大幅值、高自洽性的非类人签名；匹配的视网膜输入复现了人类"看向哪里"，却未能复现"如何看"，证明现有答案对齐与显著性指标无法认证类人视觉过程。

## 研究问题与动机
1. **核心问题**：给定与人类完全相同的 foveated（中央凹）视网膜输入，通用 MLLM 是否以与人类相似的串行方式展开目标导向视觉搜索？
2. **假设负载**：广泛实践假定使用 MLLM 作为人类观察者替代，或将注意力对齐分数（模型注意力与人类注视的重叠）解读为"模型像人类一样看"的证据，两者均将行为/空间吻合视为过程层面的许可条件，缺乏实证检验。
3. **现有指标盲区**：answer-alignment 和 saliency metrics 仅衡量空间分布或最终结果匹配，对搜索的时序动态轴（scanpath order、amplitude、determinism）盲视，可能高估 MLLM 的类人视觉程度。
4. **动机**：将"类人搜索"解耦为 decision / finding / gaze 三个正交轴进行系统性 falsifiable 检验，为 MLLM 作为人类视觉模型的适用边界提供定量依据。

## 核心贡献（创新点）
1. **三轴解耦评估框架**：将类人搜索拆解为决策（present/absent判断）、发现（目标获取效率）和注视（眼动时序动态）三个独立轴，揭示 MLLM 在这些轴上的 dissociation。与已有工作相比，先前研究多依赖单一准确率或静态显著图，本文首次在 fixated-by-fixated 层面分离行为输出与过程动态。
2. **字节级一致的 foveated MLLM 测试平台**：使用 Geisler-Perry 精度衰减函数构建与人类视觉校准的渲染器，harness/prompt/renderer 对所有模型 byte-identical，隔离出架构差异带来的搜索策略分歧。与已有工作相比，不同以往仅比较最终 attention map，本文确保输入层面的严格对齐。
3. **跨架构共享的非类人注视签名**：在 GP 条件下，三个模型共享低熵（δ = −0.67/−0.65/−0.27）、大幅值（δ = +0.50/+0.61/+0.23）、高 cross-seed 自洽性（ScanMatch 0.84/0.91/0.71 vs 人类间天花板 0.53）的扫描路径特征。与已有工作相比，通过三家族独立模型的一致性，将观察归因于 single-pass non-serial 架构共性而非个别模型缺陷。
4. **揭示现有对齐指标的方法论盲区**：证明 answer-alignment 和 saliency scores 计算的 outcome/time-collapsed map 对 process axis 盲视，high score 仅是类人视觉的必要而非充分条件。与已有工作相比，直接挑战了将注意力重叠度解读为"类人感知"的普遍做法。

## 方法详解
1. **数据集与人类参考**：COCO-Search18 验证集，141 个 target-present + 144 个 target-absent 场景，每场景 10 名人类被试 scanpath；场景 1680×1050 px，视角 ~54°×35°（~30 px/deg）。
2. **Foveation 条件族（9 个条件）**：Sharp（全清晰）、Geisler-Perry GP（Eq.S1 人类校准，唯一 human-matched）、Gaussian gist-k（k∈{8,16,24,32,48,128} 合成退化梯度）、Crop（仅保留 ~2.5° 中心圆盘）。
3. **搜索循环（Fig.3）**：每 episode 从强制中央注视开始；每步模型观察当前 gaze point 处的 foveated glimpse（历史 glimpse 保留在 context）；返回单一指令 LOOK/FUNCTION/ABSENT；不施加搜索策略，50-glimpse 上限从不主动终止。
4. **指令解析**：坐标归一化到 [0,1] 单位正方形（原点左上，y 向下）；仅消费末尾 directive line，preceding reasoning text 不影响 readout。
5. **Decision 轴度量**：One-shot existence accuracy、信号检测 d′（Hit Rate/False Alarm Rate + loglinear 校正）。
6. **Finding 轴度量**：TFP 曲线（Eq.S3，按 saccade 序号 n）、TFP@1（首次眼跳命中目标，hit 定义：注视落在目标框 ±1° 内）、TFP-end、中位注视数 NFix。
7. **Gaze 轴度量**：Gaze entropy（14×9 网格香农熵，Eq.S4）、saccade amplitude、refixation rate、scanpath length、center bias、turning angle、convex-hull coverage；效应量用 Clif's δ（Eq.S5，|δ|>0.33 视为非平凡）。
8. **Scanpath 相似度**：ScanMatch [9]，Needleman-Wunsch 全局对齐，14×9 网格 quantization，替代得分随 cell 间欧氏距离线性衰减（阈值 3.5，gap penalty 0）。
9. **统计方法**：线性混合效应模型（Eq.S6），跨 scene 和 rater 随机截距；PCA 对五个 gaze 统计量的标准化向量降维。
10. **模型**：Qwen3.5-35B-A3B（35B MoE，3B active，reasoning-tuned）、GLM-4.6V-Flash（≈9B，reasoning-tuned）、Gemma-4-E4B（≈4B，instruction-tuned，thinking disabled）。

## 实验与结果
1. **实验规模**：3 模型 × 285
