---
title: "Every Token Counts: Exact Likert-Scale Distributions for Measuring LLM Attitudes and Biases"
source: https://arxiv.org/pdf/2608.10503v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:58:54"
---

# 论文速读：Every Token Counts: Exact Likert-Scale Distributions for Measuring LLM Attitudes and Biases

## 一句话总结
本文提出了一套基于精确词元级概率质量函数（PMF）的因子实验框架，用于无噪声地评估LLM的态度与偏见；通过将心理测量学量表与分布论ANOVA结合，能够精确隔离因果主效应与交互效应，案例研究表明该方法可有效揭示模型的国家偏见，同时证明传统蒙特卡洛采样会引入不可忽视的估计方差与解码偏差。

## 研究问题与动机
- **非结构化基准无法解耦因果机制**：现有NLP评估依赖大规模未验证提示集，即便检测到聚合偏见，也无法区分该偏见源于模型基线特质、上下文混淆变量，还是多因素的高阶交互效应。
- **采样噪声与解码偏差破坏评估可靠性**：传统方法通过蒙特卡洛文本采样模拟“响应人群”，其标准误仅按 $1/\sqrt{N}$ 衰减；非默认解码参数（temperature/top-p）还会对低共识模型引入可达 2.79 分的系统性偏差。
- **无序度量无法适配序数量表**：NLP社区广泛使用的 Shannon 熵等不确定性指标将词元视为无序分类变量，无法区分“两极分化（Strongly Agree/Disagree）”与“中性集中（Neutral）”，导致对模型内部一致性估计失真。
- **心理测量学方法迁移存在三重鸿沟**：将人类行为实验设计、测量工具与分析方法平移至LLM时，缺乏实验设计控制、精确概率测量与分布论分析的基础设施。

## 核心贡献（创新点）
1. **完全交叉因子实验设计**：将LLM行为评估重构为受控的固定效应因子实验，系统隔离因果主效应与高阶交互效应。（与已有工作的本质区别：现有工作依赖大规模合成提示或开放生成，本文采用严格心理测量学控制，从设计上消除上下文混淆变量。）
2. **精确词元级PMF操作范式**：绕过蒙特卡洛采样，直接对模型在完整词汇表上的下一个词元概率分布进行数学解析，单次前向传播即获知完整分布。（与已有工作的本质区别：主流NLP依赖多次采样取均值，本文彻底消除采样方差与温度敏感偏差，实现确定性评估。）
3. **多变量序数共识度量（Multivariate Consensus）**：针对李克特量表开发了对序数距离敏感的离散度量，显式惩罚极化分布并奖励集中于均值的分布。（与已有工作的本质区别：传统熵类度量无视序数结构，本文度量能准确识别“语义矛盾但数值居中”与“极端对立”等真实行为模式。）
4. **分布论ANOVA与Hoeffding分解**：通过最优传输配对对比（comonotone coupling）解析计算效应分布，支持对复合构念得分进行精确的方差分析与效应大小量化。（与已有工作的本质区别：传统ANOVA仅处理样本均值，本文处理完整的PMF，将aleatoric uncertainty精确传播至最终构念得分，并提供方向一致性dPD与信噪比SNR。）

## 方法详解
- **实验设计**：定义独立因子（如 $\text{MODEL}$、$\text{TARGET COUNTRY}$），每个因子取离散水平，所有条件组合构成完整设计空间 $\mathcal{D}$。在每个条件 $\lambda$ 下，输入固定模板化的心理测量量表（如CETSCALE的17个item），模型响应被约束为单个数字词元（$y \in \{1,\dots,7\}$）。
- **Constraint层（格式约束）**：定义合法词元集合 $\mathcal{V}_{\text{val}}$（包含所有有效的数字表面形式，严格排除语义等价词如"seven"以惩罚句法失败）。失败率为 $1-\sum_{t\in\mathcal{V}_{\text{val}}} P_{\text{raw}}(t|x)$。对合法词元重归一化后，映射到序数PMF：$P(Y_{k,\lambda}=y|\lambda)=\sum_{t\in\phi^{-1}(y)} P(t|x_{k,\lambda}, t\in\mathcal{V}_{\text{val}})$。
- **Consensus层（序数共识）**：计算多变量共识度量以量化模型内部一致性：
  $$\mathrm{Cns}(\mathbf{Y}_{\lambda}) = 1 + \sum_{\mathbf{y}\in\mathcal{V}^K} P(\mathbf{y}) \log_2\left(1 - \frac{\|\mathbf{y}-\pmb{\mu}\|_2}{d_{\max}}\right), \quad d_{\max}=\sqrt{K}(y_{\max}-y_{\min})$$
  该度量直接作用于item-level PMF，避免聚合后丢失item间歧义信息。
- **Construct层（分布论ANOVA）**：基于items的条件独立性，复合构念得分PMF通过离散卷积获得：$P_{S_\lambda} = P_{Y_{1,\lambda}} \circledast \cdots \circledast P_{Y_{K,\lambda}}$。采用分布论Hoeffding分解将总分布拆解为基线、主效应与交互效应分布。主效应通过最优传输配对对比计算以避免方差膨胀：
  $$E_c(\lambda_c) = \mathbb{E}_{\lambda_{-c}}[S_{(\lambda_c,\lambda_{-c})} \ominus S_{0|\lambda_{-c}}]$$
  其中 $\ominus$ 表示
