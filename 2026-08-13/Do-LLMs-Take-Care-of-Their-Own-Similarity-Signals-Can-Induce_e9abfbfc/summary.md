---
title: "Do-LLMs-Take-Care-of-Their-Own-Similarity-Signals-Can-Induce"
source: https://arxiv.org/pdf/2608.12125v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:47:52"
field: "多智能体合作与战略交互"
keywords: ["cooperative AI", "multi-agent systems", "LLM decision making", "similarity signaling", "game theory", "evidential decision theory"]
innovations: ["提出b-相似性均衡框架连接Nash与EDT", "首次系统评估LLM对分级相似性信号的响应", "构建外生/内生两种相似性接地机制"]
benchmarks: ["CoopEval", "Humanity's Last Exam", "TRAIT", "Moral Choice", "Newcomb-like Problems"]
---

# 论文速读：Do LLMs Take Care of Their Own? Similarity Signals Can Induce Cooperation

## 一句话总结
本文首次系统评估了LLM智能体在获得与对手**分级相似性信号**时的战略决策行为，证明高相似性分数通常能诱导合作，并提出了一个介于Nash均衡与证据决策理论之间的**b-相似性均衡**形式化框架，用于建模LLM在相似性信息下的推理逻辑。

## 研究问题与动机
- **现实需求**：随着LLM驱动的智能体大规模部署于多智能体交互场景（交通、社交网络、金融市场等），如何确保它们能有效合作、协调与解决冲突成为关键安全问题。
- **理论缺口**：既有研究表明，在智能体知晓彼此决策模式高度相似的环境中，囚徒困境等合作难题是可解的（Evidential Decision Theory, EDT风格推理），但缺乏对**连续相似性信号**的系统性实验研究。
- **方法论空白**：Oesterheld等人（2023）虽研究了基于可信差异信号的合作，但未解决相似性信号的**构造与隔离**问题；本文首次尝试将相似性信号落地到实际可用的基准测试中。
- **模型异质性**：不同LLM对相似性信号的响应存在巨大差异（如GPT完全不受影响、Claude呈非单调趋势），需要系统对比理解这一现象。

## 核心贡献（创新点）
1. **首个LLM相似性信号评估框架**：覆盖9个LLM、5种混合动机博弈、10个基准测试，系统研究相似性信息如何影响战略决策。
2. **b-相似性均衡的形式化理论**：提出一个标量插值框架，将Nash均衡（b=0）与EDT/Kantian均衡（b=1）连接，并证明在高相似性下可支持近似最优福利。
3. **相似性信号接地机制**：首次探索基于LLM基准测试（道德困境、科学理解、人格测试等）计算配对相似性的可行性，区分外生与内生两种计算方式。
4. **CoT推理模式的系统分类**：通过LLM-as-judge框架分析17类推理依据，发现"超理性（Superrationality）"推理随相似性升高而增加（最高达96%），而"社会福利最大化"几乎缺席。
5. **可靠性风险分析**：揭示相似性信号的脆弱性——LLM甚至对随机噪声信号也表现出敏感性，且高相似性评分可能来自对自身判断的过度认同而非真实行为一致性。

## 方法详解

### 1. b-相似性均衡形式化
- **核心定义**：在对称博弈G中，给定相似度矩阵b∈[0,1]^{N×N}，对称策略配置s=(s,...,s)是**b-相似性均衡**当且仅当对每个玩家i和替代策略s'，满足：
  $$u_i(s) \geq u_i\big(s', \sigma_{-i}(s, s', b_i)\big)$$
  其中混合策略σ(s,s',b_ij)=b_ij·s'+(1-b_ij)·s，表示对手j以概率b_ij跟随偏离、以1-b_ij保持原策略。

- **边界情形**：
  - b≡0：退化为Nash均衡（独立决策假设）
  - b≡1：退化为EDT/Kantian均衡（完全复制器假设）

- **关键定理（Theorem 1）**：对任意b-相似性均衡，福利损失上界为：
  $$\text{Welfare}(s) \geq \text{Welfare}(s') - \sum_i R_i \cdot (1 - \prod_{j\neq i} b_{ij})$$
  其中R_i为支付范围。对同质相似性b，误差项为R_i(1-b^{n-1})，当b→1时线性收敛。

- **具体博弈阈值**：
  - Prisoners：b>1/2支持全合作
  - PublicGood：b>1/2支持全贡献
  - Travelers：b>2/3支持最高价格

### 2. 实验设置
- **LLM池**：Gemini 3 Flash, GPT 5.4 mini, Claude Haiku 4.5, Grok 4.20, DeepSeek V4 Pro, Kimi K2.6, Gemma 4 31B, Qwen 3.5 27B, GPT 4o
- **博弈类型**：Prisoner's Dilemma, Public Good (3-player), Traveler's Dilemma, Stag Hunt, Chicken
- **相似性信号**：X∈{0%,10%,...,100%}的抽象分数或基于基准测试的接地分数
- **提示设计**：强调相似性指"决策推理过程的相似程度"，同时保留各自独立最大化收益的目标

### 3. 相似性接地方法
- **外生计算**：基于基准测试的答案匹配率（agreement rate）或QWK（Likert量表）
- **内生计算**：让一个模型查看另一个模型的回答/推理链，自主评估相似性（0-100分）
- **基准测试**：HLE, Newcomb-like Problems, Greatest Good, Moral Choice, DailyDilemmas, TRAIT, CABIN，以及自定义的Random Die/Coin控制组

## 实验与结果

### RQ1：抽象相似性信号的影响
| 模型 | 行为特征 |
|------|----------|
| Gemini, DeepSeek, Kimi, Gemma, Grok, Qwen | 单调递增：0%时完全背叛，60%-80%跃升至完全合作 |
| GPT-4o | 随机化严重，合作率随相似性缓慢上升（Base: 49%）|
| GPT 5.4 mini | 完全不受影响，始终背叛 |
| Claude Haiku 4.5 | 非单调：80%时达到70%峰值后回落至0% |

- 关键发现：大多数模型在**50%相似度阈值**附近出现合作跃迁，与理论预测一致。

### RQ2：设置变化适应性
- **支付结构**：放大合作收益使跃迁点前移（10%-30%即达高合作率）；序数偏好下仍保持定性一致
- **推理努力**：Gemini在"高推理"模式下呈现更陡峭的阈值跃迁，符合b-相似性模型的精确预测
- **提示框架**：从"相似"改为"不同/差异"时，DeepSeek和Gemma合作率下降，Claude转为完全背叛
- **多玩家扩展**：Public Good中合作显著困难（仅Gemini在60%+时合作），因n-player下联合相似概率为b^{n-1}

### RQ3：CoT推理模式
- **"个体效用最大化"**在所有场景中都是重要考量
- **"超理性（Superrationality）"**推理比例随相似性升高至最高96%
- **"社会福利最大化"**几乎不出现
- 典型推理路径：将相似性解释为"对手与我采取相同行动的概率"，计算期望效用

### RQ4：基准测试域的影响
- **意外发现**：相似性信号来自哪个基准测试对合作率影响**很小或无影响**
- 仅DeepSeek和Gemma能正确识别Random Die/Coin为"噪声"信号
- Claude对Newcomb、Moral、GGB更敏感；其他模型表现趋同
- 暴露**可信度问题**：相似性信号可能沦为"说服性标签"

### RQ5：相似性分数计算方式
- 外生分数：62%-99%（HLE除外，因答案差异大）
- 内生分数：整体更高，尤其是看到CoT解释时
- **关键发现**：LLM倾向于**高估**自身与对手的相似性，尤其是当提供推理过程而非仅答案时

### RQ6：实际合作效果
| 方法 | 平均支付 | 排名 |
|------|----------|------|
| Exogenous (Moral/TRAIT) | ~1.72 | 第2（仅次于Contracting）|
| Endogenous (explanation-only) | 1.80 (TRAIT) | 第1-4之间波动 |
| Endogenous (decision-only) | 1.40 (HLE) | 较低 |
| Exogenous (HLE) | 1.02 | 几乎不合作 |

- 内生相似性（尤其是基于解释评估）显著优于外生答案匹配
- 排名高度依赖基准测试选择和计算方法

## 相关工作脉络
1. **Tewolde et al. (2026) CoopEval**：本文在其合作机制基准上扩展，首次引入相似性信号维度并进行系统性消融
2. **Oesterheld et al. (2023) Similarity-based Cooperative Equilibrium**：提出可信差异信号框架，但未解决信号构造问题；本文填补此空白
3. **Conitzer & Oesterheld (2023) Foundations of Cooperative AI**：奠定合作AI理论基础；本文将其"相似性信号"概念操作化
4. **Meulemans et al. (2026) A Game Theory for Foundation Models**：独立提出基于相似性推断的合作框架；差异在于本文分离信号构造与游戏环节，仅暴露标量分数
5. **Guzman Piedrahita et al. (2025)**：研究LLM在公共品游戏中的搭便车行为；本文补充了相似性机制的对比视角
6. **EDT/Superrationality文献**（Hofstadter 1985; Roemer 2010; Ahmed 2021）：b-相似性均衡在形式上连接了Nash与EDT，提供了计算可行的插值

## 局限性与未来方向
- **信号可信度风险**：LLM对随机噪声也敏感，可能被恶意利用进行操纵
- **单调性异常**：Claude等非单调行为缺乏统一解释
- **多玩家扩展不足**：Public Good等n-player场景合作率显著下降，理论阈值更高
- **内生相似性偏差**：模型可能系统性高估相似性，导致合作脆弱
- **未验证的动态场景**：实验均为单次博弈，未考察重复互动中的学习效应
- **理论存在性缺口**：b∈(0,1)时b-相似性均衡未必存在（Appendix E.3反例）

## 研究启发与可借鉴点
1. **b-相似性均衡的可迁移建模**：该插值框架可直接用于分析其他具有"相关性信念"的多智能体场景，如算法交易、自动驾驶车队协调
2. **内生相似性评估的价值**：让智能体"互相阅读推理链"比简单答案匹配更能反映真实战略兼容性，可作为团队内部的Agent对质机制
3. **CoT justification分析框架**：17类推理依据的LLM-as-judge方法可用于诊断其他合作/竞争场景中的决策模式
4. **提示工程的关键作用**：相似性定义的精确措辞（强调"推理过程"而非"输出结果"）显著影响行为，值得在其他机制设计中复用
5. **与团队方向的结合机会**：可探索将相似性信号与"价值对齐测试"结合，用于筛选真正兼容的Agent伙伴；或在多智能体安全审计中检测过度同质化风险

## 关键术语表
- **b-相似性均衡（b-similarity equilibrium）**：介于Nash与EDT之间的均衡概念，假设对手以概率b跟随我的偏离策略
- **Evidential Decision Theory (EDT)**：证据决策理论，主张将对手行为视为与自己决策相关的证据而非独立事件
- **Superrationality（超理性）**：Hofstadter提出的概念，假设所有理性玩家会得出相同结论，从而选择全局最优策略
- **外生相似性（Exogenous similarity）**：由实验者根据基准测试答案匹配率计算的相似性分数
- **内生相似性（Endogenous similarity）**：由LLM智能体自主评估的、基于对手推理/答案的相似性判断
- **Values in the Wild (VITW)**：从真实LLM对话中提取的价值分类体系，包含practical/epistemic/social/protective/personal五类
- **Chain-of-Thought (CoT)**：LLM在生成最终答案前的逐步推理过程，本文用于分析决策依据和计算内生相似性
- **Jensen-Shannon divergence (JSD)**：本文用于计算混合策略分布相似性的信息论度量

## 可复现要素
- **代码与数据**：论文声明为open-source evaluation framework，具体仓库见原文脚注
- **模型访问**：通过OpenRouter API调用9个LLM
- **关键超参**：temperature=1，reasoning effort设为"low"（Gemini实验中有4档对比）
- **样本量**：每条件10次采样，报告均值与标准误
- **基线对比**： CoopEval leaderboard (Tewolde et al. 2026)
- **评测基准**：7个公开benchmark（HLE, Newcomb, GGB, Moral, DDilemma, TRAIT, CABIN）+ 3个自定义控制组
