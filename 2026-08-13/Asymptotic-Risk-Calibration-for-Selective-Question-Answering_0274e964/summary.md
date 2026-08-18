---
title: "Asymptotic-Risk-Calibration-for-Selective-Question-Answering"
source: https://arxiv.org/pdf/2608.12008v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:35:47"
field: "大语言模型可靠性与不确定性校准"
keywords: ["selective question answering", "risk calibration", "conformal prediction", "uncertainty quantification", "language models"]
innovations: ["将选择性问答风险控制重构为线性期望约束并配合单调化经验风险处理非单调损失", "提出后验渐进校准框架A-CRC-QA无需额外训练", "在前缀最大值意义下实现统计可控的接受率-风险权衡"]
benchmarks: ["CoQA", "MedMCQA"]
---

# 论文速读：Asymptotic-Risk-Calibration-for-Selective-Question-Answering

## 一句话总结
提出 A-CRC-QA，一个后验（post-hoc）校准框架，通过将选择性问答中的接受答案错误率控制转化为线性期望约束，并结合单调化经验风险与渐进校准修正，在不修改模型或额外训练的前提下实现统计可控的选择性回答。

## 研究问题与动机
1. **LLM 幻觉问题**：大语言模型可生成流畅但事实错误的答案，在医疗等高风险领域中用户难以独立验证，需要可靠的"何时回答、何时拒绝"机制。
2. **启发式不确定性分数不足**：现有 token 概率、语义熵等方法仅提供排序信号，无法保证接受答案的错误率低于用户指定的风险水平，固定阈值在不同模型/数据集上泛化性差。
3. **标准 CRC 假设不满足**：Conformal Risk Control 要求损失函数关于校准参数单调，但选择性问答的实例级线性损失 $L(\lambda) = S(\lambda)(E - \alpha)$ 在阈值变化时，正确/错误样本对 loss 的贡献变化方向相反，导致非单调性。
4. **需要轻量后验校准**：现有一般风险控制方法需要额外训练或修改 LLM，而实际部署更倾向于无需重训的 post-hoc 模块。

## 核心贡献（创新点）
1. **将选择性问答建模为接受答案错误率控制问题**，定义 SCER（Selection-Conditioned Error Rate）作为核心指标，以用户指定风险水平 $\alpha$ 为目标优化保留率。与仅依赖不确定性排序的 Heuristic 方法本质不同。
2. **结合 LEC 线性期望约束与 CRC 渐进非单调损失处理**，将风险约束重构为 $\mathbb{E}[S(\lambda)E] - \alpha \mathbb{E}[S(\lambda)] \leq 0$ 的线性形式，并通过单调化经验风险 $\widehat{g}_n^\uparrow(\lambda) = \sup_{t \geq \lambda} \widehat{g}_n(t)$ 绕过非单调性障碍。与标准 CRC 的有限样本单调假设本质不同。
3. **提出单调化经验风险校准阈值选择算法**，在持出校准集上搜索满足修正约束的最大可行阈值，以递增前缀最大值形式高效计算，避免局部波动导致的不可靠阈值。
4. **模型无关的后验框架**，无需对 LLM 重新训练，可与任意标量不确定性估计器（如 Semantic Entropy、Word-Sequence Entropy、Predictive Entropy）组合，适配开放生成与封闭选择两种任务。

## 方法详解
**问题设定**：给定预训练 LLM $\mathcal{G}$，不确定性估计器 $\mathcal{U}$ 输出分数 $u$（越小越确定），转换为可靠性分数 $r = h(u)$（单调递减函数）。定义正确指示 $A(y^*, \hat{y})$ 与错误指示 $E = 1 - A$。选择规则 $S(\lambda) = \mathbf{1}\{r \geq \lambda\}$，目标是在 SCER $(\lambda) = \frac{\mathbb{E}[S(\lambda)E]}{\mathbb{E}[S(\lambda)]} \leq \alpha$ 约束下最大化保留率。

**线性期望重构**：定义实例损失 $L(\lambda) = S(\lambda)(E - \alpha)$，其人口风险 $g(\lambda) = \mathbb{E}[L(\lambda)] = \mathbb{E}[S(\lambda)E] - \alpha \mathbb{E}[S(\lambda)]$。则 $g(\lambda) \leq 0 \iff \text{SCER}(\lambda) \leq \alpha$，且损失有界 $-\alpha \leq L(\lambda) \leq 1-\alpha$。

**单调化经验风险**：在校准集 $\mathcal{D}_{\text{cal}}$ 上计算经验风险 $\widehat{g}_n(\lambda) = \frac{1}{n}\sum_{i=1}^n S_i(\lambda)(E_i - \alpha)$。由于非单调，构造单调化版本 $\widehat{g}_n^\uparrow(\lambda) = \sup_{t \geq \lambda} \widehat{g}_n(t)$，确保约束在所有更保守阈值下也成立。

**校准修正项**：引入 $\gamma_n = \frac{1-\alpha}{n+1}$ 作为有限样本保守边界修正，满足 $\gamma_n \to 0$。

**阈值选择**：$\widehat{\lambda}_n = \inf\{\lambda : \widehat{g}_n^\uparrow(\lambda) + \gamma_n \leq 0\}$。若可行集为空则触发全拒绝（all-abstain）规则。

**高效计算**：将校准集按可靠性降序排列 $r_{(1)} \geq r_{(2)} \geq \cdots \geq r_{(n)}$，对每个候选阈值 $k$ 计算累积值 $C_k = \frac{1}{n}\sum_{j=1}^k (E_{(j)} - \alpha)$ 和前缀最大值 $M_k = \max_{1 \leq j \leq k} C_j$，取满足 $M_k + \gamma_n \leq 0$ 的最大 $k^*$ 作为输出。

**渐进保证**：在 i.i.d. 假设下，$\lim_{n \to \infty} \mathbb{E}[L_{n+1}(\widehat{\lambda}_n)] \leq 0$，若选择概率不趋于零则 $\lim_{n \to \infty} \text{SCER} \leq \alpha$。

## 实验与结果
**数据集**：CoQA（开放对话式问答，token-level F1≥0.5 判定正确）、MedMCQA（封闭式医学多选题，精确匹配）。

**模型**：LLaMA-3.1-8B-Instruct、Qwen2.5-7B-Instruct（均未在评测集上微调）。

**不确定性估计器**：CoQA 使用 Semantic Entropy (SemEnt) 和 Word-Sequence Entropy (WSE)；MedMCQA 使用 Predictive Entropy (PE) 和 Max Softmax Probability (MSP)。各采样10个回答（temperature=0.7, top-p=0.9）。

**基线**：Fixed-50（中位数阈值）、Empirical（无修正经验阈值）、UCB-HFD（霍夫丁上界）、UCB-CLP（Clopper-Pearson 精确界）、LEC-Direct（原始有限样本线性约束）。

**主要结果（$\alpha=0.15$）**：
- A-CRC-QA 在 CoQA 上 SCER=0.143、接受率=46.2%；在 MedMCQA 上 SCER=0.138、接受率=58.7%，均低于目标风险。
- 相比 UCB-CLP，A-CRC-QA 接受率提升约 6.1pp（CoQA）和 7.4pp（MedMCQA）。
- 相比 LEC-Direct，A-CRC-QA 违反率从 25.5% 降至 12.5%（CoQA）、从 19.0% 降至 8.0%（MedMCQA），接受率降低约 3.3pp。
- Fixed-50/Empirical 违反率高达 61%-93%，说明不确定性排序本身不足以可靠控制风险。
-  Across risk levels ($\alpha \in [0.05, 0.25]$)，SCER 始终低于目标，但极低风险（$\alpha=0.05$）时不可行率分别为 16%（CoQA）和 22%（MedMCQA）。
- 校准集大小从 100 增至 1500，CoQA 接受率从 30.8% 升至 48.0%，违反率从 26% 降至 12%。

## 相关工作脉络
1. **Conformal Prediction for Language Generation**：Conformal Language Modeling、ConU、SConU 等构建包含正确答案的集合，但返回多候选而非单点决策，不适用于必须返回单一答案或拒绝的场景。
2. **SelectNet / 选择性预测理论**：Geifman & El-Yaniv 的 SelectiveNet 联合训练预测与拒绝函数，但缺乏分布无关的风险控制保证。
3. **COIN（Confidence-bounded calibration）**：使用霍夫丁/Clopper-Pearson 上界校准阈值，提供高概率风险保证但较保守，接受率低。
4. **LEC（Linear Expectation Constraint）**：将选择条件风险重构为线性期望约束，允许更宽松的校准并扩展至模型路由，但未处理非单调损失。
5. **Conformal Risk Control (CRC)**：将 conformal 推广至有界损失期望控制，但要求损失单调，无法直接应用于选择性问答的线性损失。
6. **不确定性量化方法**：SelfCheckGPT、Semantic Entropy、Word-Sequence Entropy 等提供有用排序信号，但无法指定操作性接受阈值或保证风险水平。

## 局限性与未来方向
1. **渐进保证而非有限样本保证**：定理仅在 $n \to \infty$ 下成立，有限样本下存在违反风险（实验中 VR=8%-13%），不适合需要严格有限样本控制的场景。
2. **校准集大小敏感**：小校准集（如 100 样本）导致保守阈值和高方差，实际部署需要足够校准数据。
3. **极低风险水平不可行**：当模型本身低不确定性子集不足时（如 $\alpha=0.05$），框架会触发全拒绝，无法"创造"不可靠模型的高可靠性输出。
4. **未考虑分布偏移**：假设校准集与测试集同分布，现实中域漂移可能破坏保证。
5. **未来方向**：有限样本保证、分布偏移鲁棒性、扩展至多模型路由和人工-AI 协作。

## 研究启发与可借鉴点
1. **线性期望约束重构风险控制**：将条件风险 $\frac{\mathbb{E}[SE]}{\mathbb{E}[S]} \leq \alpha$ 转化为无条件的线性约束 $\mathbb{E}[S(E-\alpha)] \leq 0$，避免了分式约束的优化困难，可迁移至其他选择性预测场景（如图像分类、时间序列异常检测）。
2. **单调化经验风险的构造**：通过前缀最大值 $\sup_{t \geq \lambda} \widehat{g}_n(t)$ 将非单调经验风险转为单调，同时保持保守性，这一技巧适用于任何非单调损失的校准问题。
3. **校准修正项 $\gamma_n$ 的设计**：采用 vanishing 修正（如 $\frac{1-\alpha}{n+1}$）平衡有限样本保守性与渐进有效性，可在其他 conformal 方法中借鉴。
4. **实验设计的严谨性**：使用 100 次随机校准-测试划分、报告违反率/不可行率/标准差，而非仅报告平均 SCER，为后续研究提供了评估协议范本。
5. **与不确定性感知的互补关系**：校准指定统计意义的操作点，不确定性估计器决定可安全接受的回答数量，两者解耦设计便于模块化替换。

## 关键术语表
**Selective Question Answering**：模型对低不确定性问题生成回答、对高不确定性问题选择拒绝（abstain）的问答机制。

**Selection-Conditioned Error Rate (SCER)**：在被接受的答案中错误答案的比例，即 $\text{Pr}(E=1 | S(\lambda)=1)$。

**Linear Expectation Constraint**：将风险控制目标重构为 $\mathbb{E}[S(\lambda)(E-\alpha)] \leq 0$ 的线性期望形式，便于优化与校准。

**Monotonized Empirical Risk**：对经验风险取上确界单调化 $\widehat{g}_n^\uparrow(\lambda) = \sup_{t \geq \lambda} \widehat{g}_n(t)$，构造非递减的风险上包络。

**Conformal Risk Control (CRC)**：将 conformal prediction 从错误覆盖控制推广至有界损失期望控制的一般框架。

**Post-hoc Calibration**：在已训练模型输出之后进行的阈值校准，无需修改模型权重或重新训练。

**Violation Rate (VR)**：在多次随机校准-测试划分中，观测到测试 SCER 超过目标风险水平的比例，衡量校准规则的稳定性。

**Infeasibility Rate (IF)**：校准集中不存在满足约束的非空阈值的选择比例，反映目标风险水平对模型-估计器组合的可行性。

## 可复现要素
- **数据集**：CoQA（公开）、MedMCQA（公开）
- **代码/权重**：论文未明确开源声明；基础模型 LLaMA-3.1-8B-Instruct 和 Qwen2.5-7B-Instruct 为开源模型
- **关键超参**：目标风险水平 $\alpha=0.15$；校准修正 $\gamma_n = \frac{1-\alpha}{n+1}$；采样设置 temperature=0.7, top-p=0.9, 10 次采样；CoQA 最大生成 token 数 64；正确性阈值 token-level F1 ≥ 0.5
