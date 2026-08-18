---
title: "Asymptotic-Risk-Calibration-for-Selective-Question-Answering"
source: https://arxiv.org/pdf/2608.12008v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:33:57"
field: "大语言模型可靠性与不确定性校准"
keywords: ["selective question answering", "uncertainty quantification", "risk calibration", "conformal prediction", "asymptotic risk control", "non-monotone loss"]
innovations: ["将选择条件误差率线性期望化并构造单调化经验风险", "结合渐近修正项实现非单调损失的风险控制", "后验无训练校准框架兼容多种不确定性估计器"]
benchmarks: ["CoQA", "MedMCQA"]
---

# 论文速读：Asymptotic-Risk-Calibration-for-Selective-Question-Answering

## 一句话总结
本文提出 A-CRC-QA，一种后验校准框架，将选择性问答中的误差控制 reformulate 为线性期望约束，并结合共形风险控制中的单调化经验风险与渐近修正，在无需额外训练的前提下为LLM答案提供统计可控的错误率保障。

## 研究问题与动机
- LLM可能生成流畅但事实错误的答案，尤其在医学等高风险领域，用户难以独立验证生成内容
- 现有启发式不确定性分数（token概率、序列似然、自一致性等）只能对答案排序，无法完美区分正确/错误预测
- 直接选取固定不确定性阈值缺乏统计保证，不同数据集/模型/任务 splits 上表现不稳定
- 标准共形风险控制（CRC）要求损失函数关于校准参数单调，但选择条件误差指标在该意义下是非单调的

## 核心贡献（创新点）
1. **将选择性问答中的接受条件误差率（SCER）reformulate为线性期望约束**：定义实例级线性损失 $L(\lambda) = S(\lambda)(E - \alpha)$，使风险控制在数学上可操作。
2. **提出单调化经验风险与渐近修正的校准 procedure**：构造 $\widehat{g}_n^{\uparrow}(\lambda) = \sup_{t \geq \lambda} \widehat{g}_n(t)$ 作为经验风险的上包络，并结合 vanishing correction $\gamma_n = (1-\alpha)/(n+1)$ 实现渐近风险保证。
3. **证明渐近边际风险控制的理论保证**：在i.i.d.假设下，校准阈值 $\widehat{\lambda}_n$ 满足 $\lim_{n\to\infty} \mathbb{E}[L_{n+1}(\widehat{\lambda}_n)] \leq 0$，且当选择概率非零时SCER被控制在目标水平 $\alpha$ 以下。
4. **提供轻量级、模型无关的后验校准框架**：无需微调或修改LLM，可与多种不确定性估计器（Semantic Entropy、Word-Sequence Entropy、Predictive Entropy、MSP）自由组合。

## 方法详解
- **问题设定**：给定预训练LLM $\mathcal{G}$ 和不确定性估计器 $\mathcal{U}$，将不确定性分数 $u$ 转换为可靠性分数 $r = h(u)$（严格递减函数）。选择条件为 $S(\lambda) = \mathbb{1}\{r \geq \lambda\}$。
- **选择条件误差率（SCER）**：$\text{SCER}(\lambda) = \Pr(E=1 | S(\lambda)=1) = \frac{\mathbb{E}[S(\lambda)E]}{\mathbb{E}[S(\lambda)]}$，目标是在 $\text{SCER}(\lambda) \leq \alpha$ 约束下最大化接受率。
- **线性期望 reformulation**：定义接受-误差指示 $Z(\lambda) = S(\lambda)E$，将条件风险等价转化为 $\mathbb{E}[Z(\lambda)] - \alpha \mathbb{E}[S(\lambda)] \leq 0$。
- **实例级线性损失**：$L(\lambda) = S(\lambda)(E - \alpha)$，取值范围 $[-\alpha, 1-\alpha]$，有界性使其适合共形风险控制视角。
- **非单调性挑战**：增大阈值 $\lambda$ 可能拒绝正确样本（损失从 $-\alpha$ 增至 0）或错误样本（损失从 $1-\alpha$ 降至 0），两者方向相反，导致标准CRC的单调性假设失效。
- **单调化经验风险**：$\widehat{g}_n^{\uparrow}(\lambda) = \sup_{t \geq \lambda} \widehat{g}_n(t)$，其中 $\widehat{g}_n(\lambda) = \frac{1}{n}\sum_{i=1}^n S_i(\lambda)(E_i - \alpha)$。单调化确保更保守阈值也满足约束。
- **渐近校准阈值**：$\widehat{\lambda}_n = \inf\{\lambda \in \Lambda : \widehat{g}_n^{\uparrow}(\lambda) + \gamma_n \leq 0\}$，其中 $\gamma_n = (1-\alpha)/(n+1)$ 提供有限样本保守边际。
- **高效计算**：仅需枚举校准集中不同的可靠性分数 $r_{(k)}$，通过前缀最大值 $M_k = \max_{1\leq j\leq k} C_j$ 确定最优接受数量 $k^*$。

## 实验与结果
- **数据集**：CoQA（开放式对话问答，最大64 token greedy解码，token-level F1≥0.5为正确）、MedMCQA（封闭式医学多选，4选1 exact match）
- **模型**：LLaMA-3.1-8B-Instruct、Qwen2.5-7B-Instruct，均在公开模型上测试，无微调
- **不确定性估计器**：CoQA用Semantic Entropy和Word-Sequence Entropy；MedMCQA用Predictive Entropy和MSP
- **主要结果（$\alpha = 0.15$）**：
  - CoQA：A-CRC-QA平均SCER为0.143，接受率46.2%，相比UCB-CLP提升约6.1个百分点接受率
  - MedMCQA：A-CRC-QA平均SCER为0.138，接受率58.7%，相比UCB-CLP提升约7.4个百分点接受率
  - 相比LEC-Direct，A-CRC-QA将违反率从25.5%降至12.5%（CoQA）、从19.0%降至8.0%（MedMCQA），代价是接受率降低约3.3个百分点
- **跨风险水平**：SCER在所有目标水平下均低于设定值；$\alpha=0.05$时不可行率分别为16%（CoQA）和22%（MedMCQA）
- **校准集大小**：$n_{cal}$ 从100增至1500，SCER逐渐逼近目标值，违反率从26%降至12%，接受率从30.8%升至48.0%（CoQA）

## 相关工作脉络
1. **Uncertainty Quantification for LLMs**：Confidence-based方法（token概率、序列似然、自评估）与Consistency-based方法（SelfCheckGPT、Semantic Entropy、Word-Sequence Entropy）提供排名信号但缺乏统计阈值控制
2. **Conformal Prediction for Language Generation**：Conformal Language Modeling、ConU、SConU构建set-valued输出并提供覆盖率保证，但不适合单答案或弃权场景
3. **Conformal Risk Control (CRC)**：Angelo poulos等提出将共形预测扩展至有界损失期望控制，要求损失关于校准参数单调，标准结果不能直接应用于选择条件误差
4. **COIN**：使用Hoeffding型上置信界校准选择性问答阈值，提供高概率风险保证，但接受率较低
5. **LEC**：将选择条件误差控制 reformulate为线性期望约束，允许更激进的校准，但未处理非单调损失导致的局部可行阈值不稳定问题

## 局限性与未来方向
- **渐近保证而非有限样本精确控制**：定理1为marginal渐近结果，不保证每次校准split都满足风险约束，需配合violation rate报告
- **小风险水平可能不可行**：当基础模型-不确定性信号不支持目标operating point时，系统需全弃权
- **未考虑分布偏移**：校准集与测试集假设同分布，实际部署中domain shift可能影响保证
- **可扩展至多模型路由与人类-AI协作**：作者指出未来方向包括finite-sample guarantee、分布偏移、多模型路由扩展

## 研究启发与可借鉴点
1. **线性期望约束的通用化思路**：将条件风险控制转化为线性期望形式，为其他选择性决策问题（如模型路由、拒绝分类）提供可复用的数学框架
2. **单调化经验风险处理非单调损失**：通过上包络 $\widehat{g}_n^{\uparrow}$ 克服实例级非单调性，同时保留比instance-wise envelope更宽松的风险控制，是稳定性-保留率的实用折衷
3. **渐近修正项 $\gamma_n$ 的设计**：$(1-\alpha)/(n+1)$ 在有限样本提供保守边际、渐近消失，可与各类单调化风险结合
4. **校准-测试split评估协议**：要求同时报告average SCER、violation rate、infeasibility rate和calibration-size sensitivity，防止将平均风险保证误述为高概率有限样本保证

## 关键术语表
- **Select Question Answering（选择性问答）**：模型对低不确定性问题作答、对高不确定性问题弃权的双模式输出机制
- **Selection-Conditioned Error Rate (SCER)**：被接受答案中的错误比例，即 $\Pr(E=1 | S(\lambda)=1)$
- **Conformal Risk Control (CRC)**：将共形预测从误覆盖控制推广至有界损失期望控制的框架
- **Linear Expectation Constraint**：将条件风险控制转化为 $\mathbb{E}[Z(\lambda)] - \alpha \mathbb{E}[S(\lambda)] \leq 0$ 的线性形式
- **Monotonized Empirical Risk**：对非单调经验风险取右 sup 上包络，构造单调非增的校准函数
- **Asymptotic Risk Guarantee**：在i.i.d.假设下随校准集增大SCER渐近受控于目标水平 $\alpha$ 的边际保证

## 可复现要素
- **数据集**：CoQA（公开）、MedMCQA（公开）
- **代码/权重**：论文未提供开源代码；使用公开LLM（LLaMA-3.1-8B-Instruct、Qwen2.5-7B-Instruct）
- **关键超参**：目标风险水平 $\alpha = 0.15$；校准集大小 $n_{cal} \in \{100, 250, 500, 1000, 1500\}$；温度采样 $T=0.7$、top-$p=0.9$、10次采样；CoQA最大64新token；F1阈值0.5
- **评估协议**：100次独立校准-测试split，报告均值±标准差
