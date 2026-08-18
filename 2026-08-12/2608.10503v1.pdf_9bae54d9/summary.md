---
title: "Every Token Counts: Exact Likert-Scale Distributions for Measuring LLM Attitudes and Biases"
source: https://arxiv.org/pdf/2608.10503v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:57:44"
field: "LLM行为评估与对齐"
keywords: ["LLM评估", "心理测量", "概率质量函数", "因子实验设计", "民族中心主义", "ANOVA", "偏见测量", "精确推理"]
innovations: ["将LLM评估形式化为完全交叉因子实验以隔离因果效应", "直接在token级PMF上操作消除采样噪声", "提出多元序数共识度量与分布式ANOVA分解框架"]
benchmarks: ["CETSCALE", "消费者民族中心主义量表"]
---

# 论文速读：Every Token Counts: Exact Likert-Scale Distributions for Measuring LLM Attitudes and Biases

## 一句话总结
论文提出了一个解析精确的框架，通过将LLM行为评估转化为完全交叉的因子实验并直接在token级概率质量函数(PMF)上操作，实现了对LLM态度和偏见的因果效应隔离与量化分析，避免了传统蒙特卡洛文本采样的噪声干扰。

## 研究问题与动机
- **现有基准的因果混淆问题**：NLP社区常用的大规模非结构化基准测试虽然能检测聚合偏差，但无法分离导致行为的潜在因果机制（基线特质、情境混淆变量或交互效应）。
- **心理测量学方法的迁移障碍**：将人类心理测量方法迁移到LLM存在设计、测量和分析三方面的方法论空白，现有基于文本生成的测评方法稳定性差且生态效度低。
- **熵度量在序数量表上的失效**：标准的不确定性度量（如Shannon熵）将token视为无序类别变量，忽略了Likert量表的序数特性，无法区分极端极化分布与中性集中分布。
- **采样噪声导致的小效应检测困难**：传统采样方法的标准误差以$1/\sqrt{N}$速率衰减，对小幅但重要的交互效应（如24.1%的参数满足$|E|<1$）存在显著的符号翻转风险。

## 核心贡献（创新点）
- **完全交叉因子实验设计**：将LLM评估形式化为固定效应因子实验，每个变量作为独立因子系统隔离因果主效应和交互效应，与已有工作的本质区别在于从"大数据集扩展"转向"受控实验设计"。
- **精确PMF测量框架**：直接在token级概率质量函数上操作而非生成文本采样，消除了蒙特卡洛采样的随机噪声，与已有工作的本质区别在于利用了LLM静态数学分布的可观测性而非模拟人类响应采样。
- **多元序数共识度量**：提出尊重序数距离的Consensus度量，能够正确量化模型内部一致性并惩罚极端极化，与已有工作的本质区别在于显式建模Likert量表的序数结构而非 treating tokens as unordered categories。
- **分布式ANOVA分解**：推导了分布型Hoeffding分解，将复合构建分数分解为基线、主效应和交互效应分布，通过最优传输配对对比确保方差最小化，与已有工作的本质区别在于在PMF层面实现而非仅点估计层面的ANOVA。

## 方法详解
- **实验设计**：采用完全交叉的固定效应设计，定义因子集合$\mathcal{C}$（如MODEL、TARGET COUNTRY），每个因子有不同水平，实验设计空间$\mathcal{D} = \prod_{c \in \mathcal{C}} \{1, \ldots, \Lambda_c\}$，每个条件$\lambda$对应因子的特定组合。
- **约束层**：处理tokenizer特异性问题，定义有效token集$\mathcal{V}_{val}$（如数字1-7的所有表面形式），计算失败率$1 - \sum_{t \in \mathcal{V}_{val}} P_{raw}(t|x)$，并通过重归一化得到精确序数项目PMF：$P(Y_{k,\lambda}=y|\lambda) = \sum_{t \in \phi^{-1}(y)} P(t|x_{k,\lambda}, t \in \mathcal{V}_{val})$。
- **共识层**：计算多元Consensus度量$\mathrm{Cns}(\mathbf{Y}_\lambda) = 1 + \sum_{\mathbf{y} \in \mathcal{V}^K} P(\mathbf{y}) \log_2\left(1 - \frac{\|\mathbf{y} - \pmb{\mu}\|_2}{d_{max}}\right)$，其中$d_{max} = \sqrt{K}(y_{max} - y_{min})$，该度量惩罚极端极化并奖励集中概率质量。
- **构建层-卷积**：利用项目条件独立性假设，通过离散卷积传播不确定性：$P_{S_\lambda} = P_{Y_{1,\lambda}} \circledast \cdots \circledast P_{Y_{K,\lambda}}$，获得复合构建分数的精确PMF。
- **构建层-分解**：采用分布型Hoeffding分解，定义grand mixture $P_{S_0} = \mathbb{E}_{\lambda \sim \pi}[P_{S_\lambda}]$，主效应分布通过配对单调最优传输对比计算：$E_c(\lambda_c) = \mathbb{E}_{\lambda_{-c}}[S_{(\lambda_c, \lambda_{-c})} \ominus S_{0|\lambda_{-c}}]$，交互效应通过Möbius反演推导。
- **分析指标**：提取四个关键指标——均值偏移$\mathbb{E}[E]$（效应大小）、预测色散$\mathrm{SD}(E)$、离散方向概率$\mathrm{dPD}(E) = \max(\mathbb{P}(E>0), \mathbb{P}(E<0))$、信噪比$\mathrm{SNR}(E) = |\mathbb{E}[E]|/\mathrm{SD}(E)$。

## 实验与结果
- **实验设计**：使用消费者民族中心主义倾向量表(CETSCALE，K=17项，7点Likert)，采用$5 \times 4$对称完全交叉设计，五个模型（Llama 3.3 70B、Gemma 3 27B、Qwen3 Next 80B、Aya Expanse 32B、Ministral 14B）对应四个原产国（美国、中国、加拿大、法国）。
- **约束结果**：Gemma 3、Llama 3.3、Qwen3几乎完美遵循约束（$M < 0.001$），Aya Expanse失败率最高（$M = 0.006$，SD=0.049）。
- **共识结果**：四个模型Cns > 0.88，Ministral显著异常（Cns ≈ 0.65-0.67），表明其item-level PMFs在Likert尺度两端分配大量概率质量。
- **构建得分**：与人类基线比较（Table 1），Aya Expanse 32B在TARGET=USA条件下期望得分89.11远超人类最民族中心群体（底特律68.58），Qwen3 Next 80B最低（53.85）。
- **主效应**：Aya-32B表现出严重正向偏差（$\mathbb{E} = 21.19$，$\mathrm{SNR} = 1.63$，$\mathrm{dPD} > 0.99$），Qwen3-80B显示稳健负向效应（$\mathbb{E} = -14.63$，$\mathrm{SNR} = 1.20$，$\mathrm{dPD} = 0.93$）；中国目标国家系统性降低民族中心主义（$\mathbb{E} = -6.46$，$\mathrm{SNR} = 1.04$）。
- **交互效应**：Gemma 3和Llama 3.3（美国开发）显示对美国/加拿大的正向交互及与中国的显著负向交互；Aya-32B（加拿大）和中国Qwen3未显示正向本国交互，Ministral-14B显示正向法国交互（+2.45）但小于对中国交互。
- **采样成本**：$N=100$时标准采样对低共识模型（Ministral）产生最大SE=0.52分；小效应（$|E|<1$，占24.1%）在$N=10$时翻转概率18%，$N=100$时仍6.3%；非默认解码（temperature变化）对Ministral可引入高达2.79分的系统性偏差。
- **提示敏感性**：提示框架作为因果主效应，可移动 latent construct score 多达13.5分；Gemma 3对措辞变化敏感度低，Aya和Ministral与特定框架强烈交互。

## 相关工作脉络
- **心理测量LLM评估**：Jiang et al. (2024)、Helwe et al. (2025)等使用心理测试评估LLM，但文本生成 elicitation 高度不稳定且生态效度低，本文通过精确PMF操作解决此问题。
- **LLM不确定性与概率**：Kuribayashi et al. (2024)证明精确next-word概率优于指令微调提示，本文在此基础进一步处理序数量表而不仅是分类事实。
- **熵度量局限**：Farquhar et al. (2024)等依赖Shannon熵量化不确定性，但熵对序数不敏感，本文提出Consensus度量弥补这一缺陷。
- **地缘政治偏见研究**：Shwartz (2022)、Bhatia & Shwartz (2023)等记录LLM系统性地缘偏见，但依赖开放生成面临解析困难和fence-sitting问题，本文通过约束响应schema解决。
- **文化对齐评估**：Kabir et al. (2025)指出标准多项选择调查易受微小扰动影响，本文采用完全交叉设计提供系统性控制。
- **Monte Carlo采样研究**：Wadi & Fredette (2025)提出采样框架，本文证明精确PMF可避免采样噪声并消除可靠性-计算权衡。

## 局限性与未来方向
- **单一构念验证**：经验实验仅聚焦消费者民族中心主义，需要扩展到更广泛的心理特质、道德价值观和地缘政治偏见。
- **语言限制**：所有实验以英语进行，模型在其他语言或文字下的系统性偏见尚未验证。
- **快照评估**：发现反映模型在测试时的潜在状态，可能不适用于未来权重更新或后续模型版本。
- **方法论范围**：框架依赖精确next-token概率分布，完全遮蔽token概率的提供商（如部分专有API）不在适用范围内。
- **多token推理扩展**：随着LLM越来越多地使用Chain-of-Thought等多token推理协议，将Consensus和Construct统计量扩展到集成潜在多token轨迹空间仍是复杂开放挑战。

## 研究启发与可借鉴点
- **精确PMF操作范式**：对于需要高可靠性评估的研究场景，直接操作token级概率分布而非文本采样可消除采样噪声，特别适用于小效应检测和高稳定性要求的评估。
- **序数-aware度量设计**：Consensus度量将概率质量按序数距离加权，为处理Likert量表等序数数据提供了可复用的分析工具，优于标准熵度量。
- **因子实验设计迁移**：将控制实验思想引入LLM评估，任何潜在混淆变量可作为额外因子引入并通过分布式ANOVA隔离，为系统性评估提供了可扩展框架。
- **最优传输配对对比**：主效应和交互效应的计算采用comonotone coupling最小化配对差异方差，避免了独立差分的方差膨胀问题，可作为分布对比的标准方法。
- **多因素敏感性量化**：通过引入提示框架作为因子并分析MODEL×PROMPT交互，可量化模型特异性敏感性，为评估鲁棒性提供系统方法。

## 关键术语表
- **Probability Mass Function (PMF)**：概率质量函数，描述离散随机变量在各个取值点的概率分布，本文指LLM对每个有效token的精确概率分布。
- **Fully Crossed Factorial Design**：完全交叉因子设计，所有因子水平的所有可能组合都被测试的实验设计，允许分离主效应和交互效应。
- **Distributional ANOVA**：分布型方差分析，在PMF层面执行ANOVA分解的方法，将复合分布分解为基线、主效应和交互效应分布。
- **Multivariate Consensus (Cns)**：多元共识度量，衡量模型在多项目Likert量表上响应一致性的指标，惩罚极端极化分布并奖励集中质量。
- **Hoeffding Decomposition**：Hoeffding分解，将函数分解为各阶交互项之和的正交分解方法，本文用于分离实验因子的主效应和交互效应。
- **Comonotone Coupling**：单调耦合，通过共享潜变量最大化协方差的配对方式，用于最小化配对差异方差。
- **Discrete Probability of Direction (dPD)**：离散方向概率，度量效应分布具有一致方向性程度的指标，定义为$\max(\mathbb{P}(E>0), \mathbb{P}(E<0))$。
- **Consumer Ethnocentrism Tendencies Scale (CETSCALE)**：消费者民族中心主义倾向量表，包含17个Likert项目的成熟心理测量工具，用于量化个体对本国产品的偏好和对外国产品的排斥。

## 可复现要素
- **数据集**：CETSCALE量表（Shimp & Sharma, 1987），K=17项，7点Likert评分，论文未提及公开但为标准公开量表。
- **代码/权重**：论文未提及代码开源，使用模型包括Llama 3.3 70B、Gemma 3 27B、Qwen3 Next 80B、Aya Expanse 32B、Ministral 14B等open-weights模型。
- **关键超参**：响应约束为单个数字token（1-7），temperature=top-p=1.0默认解码，实验条件$5 \times 4 \times 17 = 340$个精确item-level PMFs，提示框架扩展实验$3 \times 4 \times 6 \times 17 = 1224$个PMFs。
