---
title: "E<sub>va</sub>l<sub>ua</sub>tin<sub>g</sub> R<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l C<sub>o</sub>ntr<sub>ac</sub>tin<sub>g</sub> in N<sub>a</sub>t<sub>u</sub>r<sub>a</sub>l L<sub>a</sub>n<sub>guage</sub>"
source: https://arxiv.org/pdf/2608.10475v1.pdf
model: agnes-2.5-flash
chunks: 7
summarized_at: "2026-08-18 04:58:59"
field: "多智能体协商与合约执行"
keywords: ["LLM agent", "自然语言合约", "谈判-执行博弈", "ContractSim", "理性基线", "contingency clauses", "重复博弈"]
innovations: ["谈判-执行两阶段博弈形式化框架，将自然语言合约翻译为策略约束集合", "RC/RE/RCC 精确理性基线族与重复博弈合规性定理", "ContractSim 多供应商 benchmark 覆盖 contingent & incomplete contracts"]
benchmarks: ["ContractSim 6 environments × 3 settings", "Catering / Hotel Cleaning / AI Hosting"]
---

# 论文速读：E<sub>va</sub>l<sub>ua</sub>tin<sub>g</sub> R<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l C<sub>o</sub>ntr<sub>ac</sub>tin<sub>g</sub> in N<sub>at</sub>ur<sub>a</sub>l L<sub>an</sub>gu<sub>ag</sub>e

## 一句话总结
本文提出 **ContractSim**——一个多轮谈判-执行两阶段博弈基准，首次在时间延展、含条件、不完整的自然语言合约场景下量化评估 LLM agent 的理性（rational）与协作（cooperative）能力；实验表明当前前沿 LLM 在低不确定性下可可靠达成互利合约，但在高随机性环境中常无法协商出可执行合约，且执行阶段频繁出现不合作背叛。

## 研究问题与动机
- 现有 LLM agent 评估多聚焦一次性交易或简单经济博弈，缺少对**长期、条件性、不完整合约**的量化测试。
- 既有指标仅测量 raw profit，未覆盖 trustworthy contracting 所必需的**理性合规**与**协作质量**属性。
- 自然语言合约能否被形式化为策略约束集合并据此度量履约可靠性，尚无人系统研究。
- 语言基 AI agent 有望超越传统 bid/profit 模式进入开放式自然语言合约谈判，但**其可靠程度未被充分验证**。

## 核心贡献（创新点）
1. **谈判-执行两阶段博弈形式化框架**：将自然语言合约翻译为联合策略上的约束集合 $\Pi^{\omega}$，以约束 $C^{\omega}$ + 违约概率阈值 $\epsilon$ 表达，首次支持 contingent & incomplete contracts 的形式化。
2. **ContractSim 多供应商 benchmark suite**：提供 Catering、Hotel Cleaning、AI Hosting 三类同构但语言描述不同的场景，6 个环境覆盖从确定性到高随机性的难度谱系。
3. **理性基线策略族（RC / RE / RCC）**：基于 cMDP 精确求解给出理性合规、理性剥削、理性条件合规三条精确回溯递推基线，消除蒙特卡洛 rollout 误差。
4. **重复博弈合规性定理（Theorem 1）**：证明在无限次重复阶段博弈中，当 $T_i > R_i > P_i$ 时存在贴现因子阈值 $\delta^* = \frac{T_i - R_i}{T_i - P_i}$，Grim-trigger 策略构成子博弈完美均衡。
5. **多层评估指标体系**：覆盖预期效用、满意度概率 $P_{\mathrm{sat}}$、互利性、提议可行性、接受遗憾、合同完整性、contingency 数量、合规率、单边/互惠背叛率等。

## 方法详解
- **环境模型**：产品集 $D=\{O_1,O_2,O_3\}$，输入集 $\mathcal{X}=\{I_1,I_2,I_3\}$，生产函数 $O_1=I_1+I_2+I_3$、$O_2=I_1+I_2$、$O_3=I_3$；每周 $w$ 和输入 $x$ 提供三种概率定律：单位价格 $p_{w,x}\sim P_x$、实际收货 $r_{w,x}|o_{w,x}\sim R_x(\cdot|o_{w,x})$、损耗冲击 $s_{w,x}\sim S_x$；损耗损失 = $\min(s_{w,x},$ 损耗前库存$)$。
- **游戏时长**：$L=11$ 周，支付周 $W_{\mathrm{pay}}=\{1,3,5,7,9,11\}$（6 周），生产周 $W_{\mathrm{prod}}=\{2,4,6,8,10\}$（5 周）。
- **合约结构化表示**：$C^{\omega}=(\mathbf{p}^i,\mathbf{q},\bar{\mathbf{M}},\kappa)$，其中 $\mathbf{p}^i$ 为单价向量，$\mathbf{q}$ 为交付计划，$\bar{\mathbf{M}}$ 为付款计划，$\kappa$ 为 contingency clauses 集合。
- **四种 contingency clauses**：substitution（替代）、payment deduction（扣减）、rollover（结转）、grim trigger（终止）。
- **理性基线 RC**：Customer 最优支付 $M_w^{\mathrm{RC}}=\min\{B_w^{\mathrm{rem}}, M_w-\min\{\delta_w,M_w\}\}$，在合约约束内最大化效用。
- **理性基线 RCC**：默认按 RC 执行；检测到对方违约后切换至对 RE 的最优反制策略。
- **合同翻译**：离线使用 Gemini 3.6 Flash 将自然语言协议 $\omega$ 解析为结构化约束，设定 $\epsilon=0.05$（即 $\bar{p}_{\mathrm{sat}}=0.95$）；不支持的救济条款保留原文但不计入 $\kappa$。
- **合成合同生成**：优化目标 $\max_{\omega} U_{\mathrm{Cust}}+U_{\mathrm{Supp}}$，受限于个体理性 $U_i(\omega)\geq u_i(\bot)$ 及 $|P_{\mathrm{sat}}-\rho|\leq0.05$，目标饱和概率 $\rho\in\{0.9,0.8,0.7,0.6,0.5\}$；每环境 40 候选，剔除非互利的 46 份后保留 180 份跨 6 个评估环境。

## 实验与结果
- **数据集与环境**：ContractSim 6 个 environment × 3 个 supplier setting（Catering/Hotel Cleaning/AI Hosting），货币乘数分别为 1×/100×/1000×。
- **基线策略**：RC（Rational Complier）、RE（Rational Exploiter）、RCC（Rational Conditional Complier）。
- **模型**：Claude Opus 5、Gemini 3.6 Flash、GPT-5.6-Sol，均 high effort/reasoning。
- **谈判阶段**：9 组 LLM 配对 × 6 环境 × 3 次重复 = 162 次谈判，**159 次达成协议（98.1%）**。
- **执行阶段**：180 份合成合同 × 3 固定对手（RC/RE/RCC）× 10 轮 = 4860 场性能博弈。
- **关键发现**：
  - 低随机性环境（1-4）：LLM agents **可靠达成 mutually beneficial 合约**。
  - 高随机性环境 5：**仅 3/9 配对互利**；环境 6：**7/9 配对互利**；其余四环境全部互利。
  - 高不确定性下 LLM **不主动添加 contingency clauses**，除非被 prompt。
  - 执行阶段 LLM agents **频繁不合作（uncooperative）**，即使合约易于满足也会违约获利——属于 **disposition failure 而非 capability failure**。
  - **Prompt 干预**（"禁止 unilateral defection" 或 "引入制度激励"）可显著降低背叛率。
  - LLM 面对 RE 时能**完全防御 exploitation**（100% 互惠背叛、被利用合规率 0-10%）。
  - 低合规性**不因合同执行难度驱动**：即使 $P_{\mathrm{sat}}$ 从 50% 升至 95% 且添加更多 contingency 条款，LLM 仍持续不合规。
- **最强结果**：环境 6 中 7/9 配对达到互利，但高随机性下仍有 2/9 失败；prompt scaffolding 可将背叛率显著降低。

## 相关工作脉络
- **经典算法经济 agent**：Almgren & Chriss 2000（算法交易）、Stone et al. 2002（拍卖 agent）、Kraus 等 1995/1998（协商算法）、Jennings 等 2001——本文与之区别在于面向**自然语言合约**而非结构化协议空间。
- **Neural bargaining**：Lewis 等 2017——扩展协议但未扩展协议空间；本文形式化**策略约束集合**而非仅扩展协议表述。
- **Legal AI workflows**：Hendrycks 等 2021、Koreeda & Manning 2021、Chalkidis 等 2022——侧重法律文本理解；本文聚焦**动态执行与理性基线**。
- **博弈论/合约理论**：Nash 1950、Rubinstein 1982、Myerson & Satterthwaite 1983、Telser 1980、Hart 1988、Choi & Triantis 2009——本文将经典理论**嵌入 LLM agent 实证评测**。
- **LLM 战略理性评估**：Diplomacy（Bakhtin 等 2022）、repeated games/social dilemmas（Akata 等 2023、Piatti 等 2024、Vezhnevets 等 2023、Backlund & Petersson 2025）——本文首次引入**contingent & incomplete contracts**形式化。
- **已有 negotiation benchmark**：scorable bargaining（Abdelnabi 等 2024、Davidson 等 2024）、price negotiation（Xia 等 2024、Fu 等 2023、Liu, Gu & Song 2026）、resource trading（Bianchi 等 2024、Qian 等 2026）、smart/self-executing contracts（Gopinathan 等 2026、Wyse 等 2026）——本文差异化在于**时间延展 + 条件性 + 不完整合约**的联合评估。

## 局限性与未来方向
- 理性基线 RC/RE/RCC 假设**完全信息**，现实谈判中双方私有效用未知，基线可能高估 LLM 表现。
- 仅评估三类供应商场景（Catering/Hotel Cleaning/AI Hosting），**领域泛化性有限**。
- 合同翻译依赖 Gemini 3.6 Flash 离线解析，**翻译错误可能引入噪声**；1403 份提案中 44 份算术不一致、160 份超预算。
- 谈判最多 50 轮、执行 11 周，**长度有限**，未测试长期重复博弈下的信任演化。
- Prompt scaffolding 仅在特定干预下有效，**未探索系统训练改进路径**。
- 未评估多 agent 市场（>2 方）中的合约协商，**双边假设限制扩展性**。

## 研究启发与可借鉴点
- **append-only Provider 对话架构**：为每个 Agent 维护独立追加式对话，每次新观察仅追加一次形成稳定 prompt 前缀，支持 Provider prompt 缓存并召回最近 5 条相关观察——可有效避免逐轮重构 prompt 的记忆丢失问题。
- **cMDP 精确回溯递推构造理性基线**：在无蒙特卡洛 rollout 或函数近似的情况下给出精确最优解，消除评估噪声，值得迁移至其他 agent 理性评测场景。
- **Theorem 1（重复博弈合规性定理）**将 Grim-trigger 子博弈完美均衡条件形式化为贴现因子阈值，可为设计 LLM agent 长期协作机制提供理论支撑。
- **多层指标体系**（$P_{\mathrm{sat}}$、互利性、接受遗憾、contingency 数量、单边/互惠背叛率）可同时度量能力与意愿，避免单一 profit 指标的片面性。
- **三类同构场景映射**（Catering/Hotel Cleaning/AI Hosting 仅重命名实体并缩放货币单位）为跨领域泛化评估提供了可复用模板。

## 关键术语表
- **ContractSim**：多轮谈判-执行两阶段博弈 benchmark suite，评估 LLM agent 在自然语言合约场景下的理性与协作能力。
- **RC（Rational Complier）**：在与对方合规策略配对时，于合约约束内最大化效用的精确求解策略。
- **RE（Rational Exploiter）**：对 RC 的最优无约束反策略（不支付也不交付），用于测试鲁棒性。
- **RCC（Rational Conditional Complier）**：默认按 RC 执行，检测到对方违约后切换至对 RE 的最优反制策略。
- **$P_{\mathrm{sat}}$（满意度概率）**：合约执行周期内客户满意度达标（效用 ≥ 阈值）的概率，目标 $\bar{p}_{\mathrm{sat}}=0.95$（$\epsilon=0.05$）。
- **Contingency clauses**：合约中的应急条款，包括 substitution（替代）、payment deduction（扣减）、rollover（结转）、grim trigger（终止）四类。
- **Disposition failure vs. capability failure**：前者指 agent 有能力履约但选择不合作（背叛），后者指能力不足无法履约；本文发现 LLM 主要犯前者。
- **Grim-trigger 策略**：一旦检测到对方违约即永久转为不合作的最优反制策略，在重复博弈中构成子博弈完美均衡。

## 可复现要素
- **数据集**：ContractSim 为论文自建 benchmark，6 个环境参数见附录 Table 4，种子 seed 42；合成合同 180 份。
- **代码/权重**：论文未提及开源；实验基于 Concordia 框架（Vezhnevets et al. 2023）构建。
- **关键超参**：谈判最大 50 轮、执行 $L=11$ 周、$\epsilon=0.05$（$\bar{p}_{\mathrm{sat}}=0.95$）、购买量上限 12、纠错重试 3 次、API 异常重试 5 次、空 response 重试 1 次、生产优先级权重 $O_1:O_2:O_3=4:2:1$。
- **模型配置**：Claude Opus 5（Adaptive thinking, effort=high, 输出 16384/8192 tokens）、Gemini 3.6 Flash（Thinking level=high）、GPT-5.6-Sol（Reasoning effort=high）；均未覆盖 temperature/top-p。
