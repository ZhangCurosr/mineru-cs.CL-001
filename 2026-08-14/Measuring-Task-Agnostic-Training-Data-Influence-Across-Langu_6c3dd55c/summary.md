---
title: "Measuring-Task-Agnostic-Training-Data-Influence-Across-Langu"
source: https://arxiv.org/pdf/2608.13515v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:45:28"
field: "大语言模型预训练分析"
keywords: ["training data influence", "task-agnostic attribution", "pretraining trajectory", "checkPoint approximation", "domain crossover"]
innovations: ["提出基于最终参数距离的任务无关贡献度量", "推导检查点后验近似公式（式4）", "揭示文献→STEM跨期贡献cross-over模式"]
benchmarks: ["Pythia-Deduped (70M-12B)", "PolyPythia (160M/410M)"]
---

# 论文速读：Measuring Task-Agnostic Training Data Influence Across Language Model Pretraining

## 一句话总结
本文提出了一种无需依赖下游任务或验证集的**任务无关型（task-agnostic）训练数据影响力度量方法**，以"梯度更新使参数向最终预训练权重靠近的程度"作为贡献定义，并在 Pythia 和 PolyPythia 系列模型上揭示了**文献类数据在训练早期更重要、STEM 数据在训练后期更重要**的系统性跨期转变。

## 研究问题与动机
- **核心问题**：语言模型预训练目标是发展通用能力，而非优化单一任务，因此难以选择具有代表性的下游任务或验证集来衡量训练数据贡献。
- **现有方法不足**：
  - Influence Functions、Data Shapley、TracIn、TRAK 等方法均将影响力绑定于特定测试点损失/预测或下游任务性能。
  - 不同中间检查点的任务表现难以横向比较——早期 checkpoint 可能已习得某任务部分能力但尚不能完整解决，导致跨阶段影响评估不一致。
  - 任务特定的归因会因验证域选择不同而产生矛盾结论（§5.4 对比实验）。

## 核心贡献（创新点）
1. **提出任务无关的预训练数据贡献度量**：以"示例梯度更新对最终参数 squared L2 距离的缩减量"定义贡献，从根本上摆脱对下游任务/验证集的依赖。
2. **推导基于检查点的后验近似公式**：利用稀疏保存的检查点（而非全步参数）估计示例级贡献，无需重新训练即可应用于大规模预训练轨迹。
3. **揭示系统性跨期数据贡献转变**：文献/非 STEM 数据在训练早期更具影响力，STEM 数据在后期上升，形成稳定的"cross-over"模式。
4. **建立方法的可靠性与适用范围**：验证了检查点近似精度、参考端点敏感性、跨模型配置一致性，以及与任务特定方法的定位差异。
5. **为预训练数据配方与课程学习提供轨迹级视角**：支持"分阶段感知（stage-aware）数据组成"的设计思路，而非静态混合。

## 方法详解
### 核心定义
令 $t = 0, \dots, T-1$ 为更新步索引，$\theta_t$ 为更新前参数，$B_t = (x_1^t, \dots, x_N^t)$ 为 mini-batch，$\Delta_t = \theta_{t+1} - \theta_t$ 为参数更新量，$\theta^* = \theta_T$ 为最终预训练参数，$S_t = \|\theta^* - \theta_t\|_2^2$ 为当前参数到最终参数的 squared L2 距离。

**Mini-batch 贡献**（式 1-2）：
$$\mathrm{Cont}(B_t) = S_t - S_{t+1} = 2\Delta_t^\top(\theta^* - \theta_t) - \|\Delta_t\|_2^2$$
第一项度量更新方向与"朝向最终参数方向"的对齐程度，第二项为更新范数的惩罚。贡献可为负——此时该 batch 将参数推离最终状态（作者称此类示例为 **opponents**）。

**示例级贡献**（式 3，假设 SGD）：
$$\mathrm{Cont}(x_k^t) = 2\Delta_{t,k}^\top(\theta^* - \theta_t) - \Delta_{t,k}^\top \Delta_t$$
其中 $\Delta_{t,k} = -\eta_t \nabla_\theta \ell(x_k^t; \theta_t)$。所有示例贡献之和等于 mini-batch 贡献。

**检查点近似**（式 4，§3.3）：
利用间隔 $[c, c')$ 内的检查点 $\theta_c, \theta_{c'}$ 近似步级量：
$$\mathrm{Cont}(x_k^t) \approx 2\Delta_{c,k}^\top(\theta^* - \theta_c) - \Delta_{c,k}^\top(\theta_{c'} - \theta_c)$$
其中 $\Delta_{c,k} = -\eta_t \nabla_\theta \ell(x_k^t; \theta_c)$。该方法可在已有检查点上映射，无需重训。

## 实验与结果
- **数据集与模型**：Pythia 系列（70M–12B，共 6 规模×去重版本），18 个配置，每个 154 个检查点；PolyPythia（160M/410M 各 3 次独立运行，含解耦种子变体）。预训练语料为 The Pile（300B tokens）。
- **评估方式**：每区间均匀采样 1,000 示例，按贡献分布均值、标准差、opponent 占比统计；按文本 PPL（用 Pythia-12B 最终 checkpoint 计算）分五箱分析难度关系；按 NeMo Curator 领域分类器（26 类）分析领域关系，聚焦 top/bottom 5%。
- **核心结果**：
  - **贡献分布动态**（图 2）：均值在约 40k 步达到峰值后下降；opponent 占比从早期 ~0–1% 上升至后期 ~8–10%。
  - **难度关系**（图 3）：中期高 PPL 示例贡献占比从 ~20% 升至 ~25%，低 PPL 从 ~20% 降至 ~15%。
  - **领域跨期转变**（图 4）：早期 bottom 5% 集中于 Computers & Electronics / Science 等 STEM 领域，late-stage 转为 Books & Literature；top 5% 呈现相反趋势，形成**Literature → STEM cross-over**。
  - **检查点近似验证**（表 1）：Early 区间 Pearson r = 0.622，Mid 区间 r = 0.808，Late 区间 r = 0.950（Pythia-70M-Deduped 重训验证）。
  - **参考端点敏感性**（表 3）：以 140k 为参考 vs 143k 原参考，Spearman r = 0.997，top-5% 重叠 0.972；以 30k/70k 为参考则显著偏离。
  - **跨配置一致性**（§5.3）：不同模型规模、权重初始化、数据顺序下定性模式稳定；数据顺序比权重初始化引入略大变异。
  - **与任务特定方法对比**（§5.4，图 13）：TracIn-style 跨域归因高度依赖验证目标选择，无法复现 Literature→STEM cross-over 模式。

## 相关工作脉络
1. **Influence Functions**（Koh & Liang, 2017）：定义示例对测试点损失/预测的影响，任务绑定；本文替换为参数空间距离。
2. **Data Shapley**（Ghorbani & Zou, 2019）：按边际预测效用分配数据价值，同样需定义效用函数/任务。
3. **TracIn**（Pruthi et al., 2020）：通过梯度追踪估计影响；本文公式在正交更新假设下退化为 TracIn 自影响项。
4. **TRAK**（Park et al., 2023）：改进可扩展性-效果权衡；本文共享"用已有检查点做后验估计"的目标。
5. **LoGra**（Choe et al., 2025）：面向 LLM 的可扩展 influence functions；本文进一步摆脱对测试点的依赖。
6. **Wang et al. (2025)、Chang et al. (2025)、DATE-LM**（Jiao et al., 2025）：展示任务特定归因的异质性与敏感性问题， motivate 本文的 task-agnostic 定位。

## 局限性与未来方向
- **参考端点依赖**：贡献分数定义于特定预训练运行的最终参数，不同参考 checkpoint 可产生显著不同的排名（§5.2）。
- **参数空间距离的局限性**：squared L2 距离对参数化敏感，不直接度量模型行为变化；基于预测散度或信息几何的 function-aware 替代更昂贵。
- **检查点近似未在大模型上精确验证**：仅在小规模 Pythia-70M 上重训验证，大模型精度待检验。
- **示例级分解假设 SGD**：实际使用自适应优化器（Adam），未精确建模优化器状态历史。
- **外推性未确立**：实验局限于相近模型家族与预训练数据，其他架构/数据集/配方的适用性待验证。
- **相关性 vs 因果性**：刻画了贡献随训练的变异，但未直接建立"改变数据混合"的因果效应，需干预式评估。
- **领域与文本难度未解耦**：STEM 领域的早期/后期表现可能与文本难度等混杂属性相关。

## 研究启发与可借鉴点
1. **Trajectory-level 视角的价值**：放弃"固定验证集=贡献目标"的范式，改用最终参数作为统一参照，使跨阶段比较成为可能——这一思路可迁移到其他预训练过程分析。
2. **检查点近似公式的工程友好性**：式 (4) 只需梯度 + 两个检查点参数，即可做后验估计，无需重训；可直接嵌入现有 checkpoint 分析管线。
3. **opponent 占比作为训练健康指标**：晚期 opponent 占比上升至 8–10%，可作为课程学习或数据清洗的质量监控信号。
4. **为 stage-aware 数据配比提供实证依据**：文献支持"后期增加 STEM 数据比例"的工程启发（如 SmolLM3、Eurollm 的做法），可从轨迹级贡献角度加以验证。
5. **可与本团队方向结合**：数据溯源（data provenance）、毒性/偏差过滤、事实归因等应用中，task-agnostic 度量可作为补充视角，与任务特定归因联合使用。

## 关键术语表
- **Task-agnostic influence**：不依赖下游任务或验证集的贡献度量，以参数空间轨迹为参照。
- **Contribution（贡献）**：训练更新使当前参数到最终参数的 squared L2 距离的缩减量。
- **Opponent（对抗示例）**：贡献为负的示例，其梯度更新将参数推离最终状态。
- **Checkpoint-based approximation**：用稀疏检查点代替连续步级参数，实现后验估计。
- **Cross-over（跨期转变）**：文献类数据在早期高贡献、STEM 数据在后期高贡献的系统性模式切换。
- **TracIn**：通过追踪梯度下降路径估计训练数据影响的方法（Pruthi et al., 2020）。
- **NeMo Curator Domain Classifier**：用于对文本进行 26 类领域分类的工具。
- **PolyPythia**：Pythia 的变体套件，用于分离权重初始化与数据顺序的扰动效应。

## 可复现要素
- **数据集**：Pythia 与 PolyPythia 模型及其检查点公开可获取；预训练语料为 The Pile。
- **代码/权重**：论文未提供单独代码仓库声明，但方法仅依赖公开检查点与梯度计算。
- **关键超参**：batch size = 1024 sequences，每 sequence 2048 tokens；采样 1,000 示例/区间；PPL 用 Pythia-12B-Deduped 最终 checkpoint 计算；领域分类用 NeMo Curator（置信度 > 0.95）。
- **未提及**：学习率具体 schedule、优化器超参（仅说明使用 Adam 类自适应优化器）。
