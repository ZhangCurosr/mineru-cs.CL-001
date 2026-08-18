---
title: "Every Token Counts: Exact Likert-Scale Distributions for Measuring LLM Attitudes and Biases"
source: https://arxiv.org/pdf/2608.10503v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:57:20"
---

# 论文速读：Every Token Counts: Exact Likert-Scale Distributions for Measuring LLM Attitudes and Biases

## 一句话总结
本文提出一个精确的分析框架，将 LLM 行为评估形式化为完全交叉的因子实验，并直接在 token 级概率质量函数（PMFs）上运算以消除 Monte Carlo 采样噪声；通过分布型 ANOVA 和多元有序共识度量，能够数学隔离因果主效应与交互效应，并据此揭示了 LLMs 的系统性国家起源偏见。

## 研究问题与动机
1. **无结构基准无法解构因果机制**：现有 NLP 评估依赖大规模无结构基准数据集，即便能检测到聚合偏见，也无法区分偏见源自基线特征、上下文混淆因素还是高阶交互效应。
2. **生成文本采样的统计噪声**：传统做法通过 Monte Carlo 多次采样模拟"人群"响应，但这会引入不可忽略的采样误差，且 temperature/top-p 等非默认解码策略会产生系统性偏差。
3. **熵度量忽略有序性**：NLP 领域广泛使用的 Shannon 熵将输出视为无序类别变量，无法区分"强烈同意/强烈反对两极分化"与"稳定集中在中性"这两种截然不同的行为模式。
4. **心理测量学方法迁移空白**：人类行为科学依赖经过验证的量表（如 Likert 量表）和受控实验设计，但将其直接移植到 LLM 评估时存在实验设计、测量和分析三方面的方法论断层。

## 核心贡献（创新点）
1. **完全交叉因子实验设计框架**：将 LLM 评估转化为固定效应因子实验，把模型身份、目标人口/国家等变量作为独立因子，系统隔离因果主效应与交互效应；与以往工作本质区别在于用行为科学的实验控制替代"规模优先"的无结构数据扩展。
2. **精确 token 级 PMF 运算替代采样**：利用 LLM 对给定提示存在恒定、可观测内部状态的特性，直接操作精确的 token 级概率质量函数，彻底消除采样误差和 temperature 引入的系统偏差；与 Wadi & Fredette (2025) 的 Monte Carlo 采样框架形成根本对立。
3. **多变量有序共识度量（Cns）**：针对 Likert 量表显式缩放概率与均值的有序距离，严重惩罚极端两极分化而奖励集中质量；与 Farquhar et al. (2024) 的语义熵本质区别在于熵对有序结构完全不敏感。
4. **分布型 ANOVA（Distributional ANOVA）**：通过分布 Hoeffding 分解将复合构念得分分布拆解为基线、主效应和交互效应分布，并在期望层面恢复经典唯一 Hoeffding 分解（Theorem 3.1）；与经典 ANOVA 的本质区别是保留每个效应的完整 PMF 而非仅点估计。
5. **最优传输配对对比机制**：使用共单调耦合（1D 单调最优传输）最小化配对差异方差，隔离因子归因的系统性偏移；与简单卷积减法（假设独立）的本质区别是不人为 inflate 方差。

## 方法详解
1. **实验设计**：定义因子集合 $\mathcal{C}$，每个因子 $c$ 有离散水平集。实验设计空间 $\mathcal{D} = \prod_{c \in \mathcal{C}} \{1, \dots, \Lambda_c\}$ 为所有因子水平的笛卡尔积。在每个条件 $\lambda$ 下，模型评估由 $K$ 个项目组成的心理测量工具，响应限制在有序 Likert 量表 $\mathcal{V} = \{y_{\min}, \dots, y_{\max}\}$（如 1–7）。关键假设：项目间给定 $\lambda$ 条件独立，联合 PMF 可精确因子化。

2. **约束层（Constraint）**：定义有效 token 集 $\mathcal{V}_{\text{val}} \subset \mathcal{V}$（包含所有合法数字表面形式，严格排除语义等价物如 "seven"），通过映射 $\phi: \mathcal{V}_{\text{val}} \to \mathcal{V}$ 计算精确有序项目 PMF：$P(Y_{k,\lambda} = y \mid \lambda) = \sum_{t \in \phi^{-1}(y)} P(t \mid x_{k,\lambda}, t \in \mathcal{V}_{\text{val}})$。失败率定义为 $1 - \sum_{t \in \mathcal{V}_{\text{val}}} P_{\text{raw}}(t \mid x_{k,\lambda})$。

3. **共识层（Consensus）**：计算多变量 Consensus 度量：
$$\text{Cns}(\mathbf{Y}_\lambda) = 1 + \sum_{\mathbf{y} \in \mathcal{V}^K} P(\mathbf{y}) \log_2\left(1 - \frac{\|\mathbf{y} - \pmb{\mu}\|_2}{d_{\max}}\right)$$
其中 $d_{\max} = \sqrt{K}(y_{\max} - y_{\min})$。该度量对极端两极分化严重惩罚，对集中质量给予奖励，解决了熵无法区分有序距离的问题。

4. **构念层（Construct）**：
   - **复合得分 PMF**：通过离散卷积传播不确定性：$P_{S_\lambda} = P_{Y_{1,\lambda}} \circledast \cdots \circledast P_{Y_{K,\lambda}}$。
   - **分布型 Hoeffding 分解**：将总得分分布分解为 $S_0$（全局混合基线）、$E_c(\lambda_c)$（主效应分布）和 $E_U(\lambda_U)$（交互效应分布），满足期望层面的加法恒等式（Theorem 3.1 保证与经典固定效应 ANOVA 一致）。
   - **配对差异计算**：主效应通过共单调耦合配对差分计算：$E_c(\lambda_c) = \mathbb{E}_{\lambda_{-c}}[S_{(\lambda_c, \lambda_{-c})} \ominus S_{0|\lambda_{-c}}]$；交互效应通过 Möbius 反演风格的配对对比计算。
   - **效应摘要指标**：Mean Shift $\mathbb{E}[E]$、Predictive Dispersion $\text{SD}(E)$、Discrete Probability of Direction $\text{dPD}(E) = \max(\mathbb{P}(E>0), \mathbb{P}(E<0))$、Signal-to-Noise Ratio $\text{SNR}(E) = |\mathbb{E}[E]|/\text{SD}(E)$。

## 实验与结果
- **数据集与基线**：Consumer Ethnocentrism Tendencies Scale (CETSCALE, Shimp & Sharma, 1987)，$K=17$ 项 Likert 量表（$\mathcal{V}=\{1,\dots,7\}$）。5 个 LLM：Llama 3.3 70B、Gemma 3 27B（美国）；Qwen3 Next 80B（中国）；Aya Expanse 32B（加拿大）；Ministral 14B（法国）。目标国家：USA、China、Canada、France。设计为对称 $5 \times 4$ 完全交叉固定效应。
- **约束遵循**：Gemma 3、Llama 3.3、Qwen3 接近完美遵循（$M < 0.001$, $\text{SD} < 0.001$）；Aya Expanse 失败率最高（$M = 0.006$, $\text{SD} = 0.049$）。
- **共识度量**：四模型 Cns > 0.88；Ministral 为显著离群值（Cns ≈ 0.65–0.67），显示严重行为极化。
- **与人类基线对比（TARGET=USA）**：Aya Expanse 32B 得分最高（$E[S]=89.11$），显著超过最民族中心主义的人类样本（Detroit: 68.58）；Llama 3.3 70B（71.99）和 Ministral 14B（70.20）也超出人类均值；Qwen3 Next 80B 最低（53.85）。
- **主效应**：Aya-32B 严重正向偏离全局均值（$E=21.19$, $\text{SNR}=1.63$, $\text{dPD}>0.99$）；Qwen3-80B 高度稳健负向主效应（$E=-14.63$, $\text{SNR}=1.20$, $\text{dPD}=0.93$）。目标国家主效应：对中国系统性降低民族中心主义（$E=-6.46$, $\text{SNR}=1.04$），与美国（$+2.84$）和加拿大（$+4.62$）形成鲜明对比。
- **交互效应与国家起源偏见**：GEMMA 3-27B（$+3.21$）和 LLAMA 3.3-70B（$+2.59$）显示正向本国（美国）交互；但 AYA-32B（$+3.77$ 对加拿大）和 QWEN3-80B（$+0.20$ 对中国）未显示正向本国交互，MINISTRAL-14B 对法国仅 $+2.45$（小于对中国的 $+4.54$）。
- **采样成本量化**：在 $N=10$ 时标准采样以 18% 概率翻转小效应（$|E|<1$）符号，$N=100$ 时仍为 6.3%；精确 PMF 框架在单次前向传播中达到 $N \to \infty$ 极限。非默认温度对低共识模型（如 Ministral）引入高达 2.79 分系统性偏差。

## 相关工作脉络
1. **Kuribayashi et al. (2024)** 证明精确 next-word 概率优于指令微调提示模拟人类认知行为；本文定位：进一步将精确概率与因子实验设计和分布型 ANOVA 结合，实现因果效应隔离而非仅行为模拟。
2. **Farquhar et al. (2024)** 用语义熵检测幻觉；本文定位：熵对有序变量无效，本文的 Cns 度量显式处理有序距离，解决 Likert 量表评估中的度量失效问题。
3. **Wadi & Fredette (2025)** 提出 Monte Carlo 采样评估框架；本文定位：完全绕过采样范式，直接在精确 PMFs 上运算，消除采样误差和 temperature 偏差。
4. **Bhatia et al. (2024, 2025)** 评估视觉语言模型文化理解和价值漂移；本文定位：提供严格的因果实验控制，通过因子分解隔离混淆变量，而非依赖观察性数据集的聚合统计。
5. **Shwartz (2022), Kabir et al. (2025)** 研究地缘政治偏见和文化对齐；本文定位：通过 MODEL×TARGET 交互效应的精确分解，揭示聚合基准无法捕捉的模型特异性偏见模式（如 Ministral 对法国的正向偏袒被聚合视图错误地掩盖）。
6. **Tastle & Wierman (2007)** 原始 Consensus 度量针对人类样本标量响应设计；本文定位：将其扩展为多变量版本直接作用于 PMF，解决项目间异质性问题。

## 局限性与未来方向
1. 实证验证仅聚焦消费者民族中心主义单一构念，方法论建立后需扩展至更广泛的心理特质、道德价值观和地缘政治偏见评估。
2. 所有实验用英语进行，其他语言/脚本下的系统性偏见是否成立仍是开放实证问题。
3. 快照评估结果反映测试时模型状态，可能不适用于未来权重更新或后续模型版本。
4. 框架依赖精确 next-token 概率，完全遮蔽概率的提供商（部分商业 API）不在适用范围内。
5. 随着 LLMs 越来越多地使用多 token 推理协议（如 Chain-of-Thought），将 Consensus 和 Construct 统计扩展到积分 latent multi-token trajectory spaces 是复杂的开放挑战。

## 研究启发与可借鉴点
1. **因子实验设计迁移价值高**：完全交叉的因子设计可推广至道德判断、个性特质、政治倾向等 LLM 行为评估任务，提供传统无结构基准无法实现的因果推断能力。
2. **精确 PMF 运算消除采样噪声**：任何需要概率分布分析的任务均可借鉴此范式，避免 $1/\sqrt{N}$ 收敛缓慢的采样误差和 temperature 引入的系统偏差。
3. **多变量 Cns 度量适用于有序响应评估**：对于任何使用 Likert 量表或有序评分的 LLM 评估任务，Cns 比熵更能准确捕捉内部一致性和极化模式。
4. **分布型 ANOVA 的交互效应隔离**：框架提供的交互效应分布可直接用于理解多因素如何共同塑造 LLM 行为，避免聚合分析导致的方向性误判（如本文揭示的聚合视图错误掩盖 Ministral 对法国的正向偏袒）。
5. **Prompt framing 作为第一类因子**：将提示变体提升为实验因子并通过 ANOVA 隔离其主效应和交互效应，为构建鲁棒评估管道和量化模型提示敏感性提供系统化方法。

## 关键术语表
**Probability Mass Function (PMF)**：离散随机变量的概率分布，给出每个可能取值的确切概率质量。
**Fully Crossed Factorial Design**：所有因子的所有水平相互组合的实验设计，使研究者能独立估计每个因子的主效应及因子间的交互效应。
**Distributional ANOVA**：在概率分布层面执行的方差分析，通过 Hoeffding 分解将复合得分分布拆解为基线、主效应和交互效应的完整 PMF。
**Multivariate Consensus (Cns)**：针对有序变量的共识度量，显式考虑响应向量与均值向量的有序距离，对极端两极分化严重惩罚。
**Comonotone Coupling**：基于 1D 单调最优传输的配对方法，最大化配对 Draws 之间的协方差，从而最小化配对差异的方差。
**Hoeffding Decomposition**：将多元函数分解为加法成分的数学工具，用于隔离主效应
