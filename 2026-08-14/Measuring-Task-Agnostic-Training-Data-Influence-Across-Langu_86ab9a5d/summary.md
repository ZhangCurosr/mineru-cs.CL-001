---
title: "Measuring-Task-Agnostic-Training-Data-Influence-Across-Langu"
source: https://arxiv.org/pdf/2608.13515v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:45:36"
---

# 论文速读：Measuring-Task-Agnostic-Training-Data-Influence-Across-Langu

## 一句话总结
本文提出了一种任务无关的训练数据影响力度量方法，将单个样本的贡献定义为“其梯度更新使模型参数向最终预训练参数靠拢的 squared L2 距离减少量”。通过在已有 checkpoint 上进行事后近似估计，无需重新训练即可全局分析，揭示了大语言模型预训练中影响力数据的系统性阶段迁移规律（早期文学类数据更重要，后期 STEM 类数据贡献显著上升）。

## 研究问题与动机
- **核心问题**：如何在大规模语言模型预训练过程中一致、可比较地衡量训练数据的影响力，且不受限于特定下游任务或验证集的选择。
- **现有方法不足**：
  1. 主流影响力方法（Influence Functions、Data Shapley、TracIn 等）均将样本效果绑定于特定目标行为（测试点损失、预测结果或下游任务指标），难以覆盖预训练所追求的“通用能力”。
  2. 预训练周期跨度大，不同阶段模型具备的下游能力差异显著（早期可能具备相关能力但尚未完全掌握），导致基于任务性能的中间 checkpoint 影响力难以跨阶段公平对比。
  3. 现有预训练归因研究多聚焦单一任务或事实追踪，缺乏从“参数轨迹收敛”这一全局视角出发的阶段演化刻画。

## 核心贡献（创新点）
1. **提出任务无关的影响力重定义**：用样本更新带来的“最终参数距离缩减量”替代下游任务指标，使影响力定义摆脱任务依赖，可与不同训练阶段直接比较。
2. **推导 Checkpoint-based 事后近似算法**：将连续训练步的参数更新替换为相邻保存 checkpoint 的差值，仅需最终权重与稀疏 checkpoint 即可估计示例级贡献，避免全量重训。
3. **揭示预训练数据贡献的阶段性演化规律**：在 Pythia 与 PolyPythia 系列上系统分析发现，文学类文本在早期高度对齐收敛轨迹，STEM 类文本在后期贡献显著上升，呈现一致的 crossover 模式。
4. **建立方法的可靠性与适用范围**：全面验证了 checkpoint 近似的准确性、参考终点的局部稳定性、跨模型配置/初始化/数据排序的鲁棒性，并证明任务特定方法无法复现该全局 crossover。

## 方法详解
- **Mini-Batch 影响力定义**：设最终预训练参数为 $\theta^*$，第 $t$ 步更新前参数为 $\theta_t$，定义平方距离 $S_t = \|\theta^* - \theta_t\|_2^2$。一个 mini-batch $B_t$ 的贡献为距离缩减量：
  $$\operatorname{Cont}(B_t) = S_t - S_{t+1} = 2\Delta_t^\top(\theta^* - \theta_t) - \|\Delta_t\|_2^2$$
  其中第一项衡量更新方向与“指向最终参数方向”的对齐程度，第二项惩罚更新步长的模长；贡献值可为负，表示该步将参数推离最终状态。
- **Example-Level 分解**：假设标准 SGD，单样本更新 $\Delta_{t,k} = -\eta_t \nabla_\theta \ell(x_k^t; \theta_t)$，示例级贡献定义为：
  $$\operatorname{Cont}(x_k^t) = 2\Delta_{t,k}^\top(\theta^* - \theta_t) - \Delta_{t,k}^\top \Delta_t$$
  满足可加性 $\sum_k \operatorname{Cont}(x_k^t) = \operatorname{Cont}(B_t)$。若假设 batch 内样本更新向量两两正交，则退化为与 TracIn 自影响力项形式相同的结构，天然对噪声/错误标注样本施加强惩罚。
- **Checkpoint-Based 近似**：为避免访问每一步参数，取相邻 checkpoint $c$ 与 $c'$，令 $\theta_t \approx \theta_c$，$\Delta_t \approx \theta_{c'} - \theta_c$，则：
  $$\operatorname{Cont}(x_k^t) \approx 2\Delta_{c,k}^\top(\theta^* - \theta_c) - \Delta_{c,k}^\top(\theta_{c'} - \theta_c)$$
  该近似仅需 $\theta_c, \theta_{c'}, \theta^*$ 与单样本在 $\theta_c$ 处的梯度，支持 post-hoc 批量计算。
- **辅助分析工具**：使用 NeMo Curator Domain Classifier 将文本划分为 26
