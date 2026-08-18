---
title: "Mint-Agent-Introducing-Finance-Native-Agentic-Foundation-Mod"
source: https://arxiv.org/pdf/2608.16386v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:42:05"
field: "金融智能体与可验证强化学习"
keywords: ["financial agent", "verifiable reward", "on-policy distillation", "model merging", "long-horizon execution", "evidence ledger", "RLVR", "GSPO"]
innovations: ["TIES 合并 + 多教师 on-policy 蒸馏（MOPD）统一财务推理与长程执行双专家", "原子/长程双轨可验证奖励 R_atom=A·S·C 与 R_long=A·G·H 的 verifier 设计", "MintHarness 证据优先的状态化执行框架实现长轨迹可审计"]
benchmarks: ["BizFinBench", "FinanceBench", "RFC-Bench Task 2", "FinanceAgentBench v1.1", "FinanceAgentBench v2", "FinSearchComp T2", "FinSearchComp T3"]
---

# 论文速读：Mint-Agent: Introducing Finance-Native Agentic Foundation Models

## 一句话总结
本文提出了 Mint-Agent，一套面向金融领域的原生智能体基础模型（Mint-Cu 9B、Mint-Ag 27B），通过从真实金融源构建带溯源的证据驱动任务、设计 MintHarness 状态化执行框架，以及结合 SFT/RLVR/OPD/TIES 合并/多教师 on-policy 蒸馏的训练管线，实现了高精度财务推理与可审计的长程执行能力的统一。

## 研究问题与动机
- 金融研究并非单一事实检索，而是跨来源阅读披露、识别报告期、比对数值并计算决策的综合过程；现有智能体在单步看起来合理的情况下仍可能断裂链条（来源不权威、报表期间错误、单位混用等）。
- 若任务生成缺少溯源，模型只能学到财务分析的"表面形式"而非内在规范；若执行不保留状态，关键失败信息会在可监督前消失；若训练仅奖励最终文本，决定轨迹成败的关键决策得不到充分监督。
- 金融场景下"正确性"标准不仅要求最终答案精确，还要求每个结论都可通过可恢复的证据链和可重放计算进行审计。
- 当前 deep research 的两条路线（基于 harness 的外部协调系统 vs. 基于训练的内在化策略）难以同时兼顾自主性与可审计性，需要统一的"可追溯证据契约"贯穿数据、执行与训练。

## 核心贡献（创新点）
- **全栈金融原生智能框架**：以"单一可追溯证据契约"串联数据构造、状态化执行与可验证策略学习，区别于以往将领域知识、工具调用与可审计性作为独立模块处理的做法。
- **双专家统一架构（TIES + 多教师 OPD）**：从公共基模型分别训练财务推理专家与长程执行专家，先通过 TIES 合并抑制参数冲突，再通过 multi-teacher on-policy distillation（MOPD）按任务来源路由对齐行为，避免一般性权重合并后能力互相干扰的问题。
- **可验证奖励函数（RLVR）设计**：原子任务以 A_κ×S×C_κ 三元门控验证答案等价、溯源支撑与财务语义重放；长程任务以 A_long×G×H 验证最终答案、证据图重放与操作合规，避免"看似正确但无法追溯"的错误奖励。
- **MintHarness 证据优先的执行框架**：将证据账本（Evidence Ledger）、工作记忆与有界上下文解耦，持久化记录工具调用、失败路径与取证元数据，使长轨迹全程可审计。
- **金融领域双旗舰模型与全面验证**：Mint-Cu（9B）和 Mint-Ag（27B）在 7 个金融基准上全面领先，证明小参数量亦可承载长程金融执行能力，并在成本—性能 Pareto 前沿上优于更大规模通用模型。

## 方法详解
- **数据引擎（Data Engine）**
  - **原子任务（D_atom）**：从 EDGAR、XBRL、交易所记录、FRED 与会计准则本体抽取带定位的金融事实，按五类能力（知识 Knowledge、抽取 Extraction、计算 Calculation、分析 Analysis、验证 Verification）构建受约束问题。验证条件：supp(r)⊆E、Eval(r;E)=y、|Ans(q;E)|=1，确保推导完全由样本证据支撑且答案唯一。
  - **长程任务（D_long）**：复用 grounded corpus S，叠加时间切片 W_t_cut 快照保证可复现；以分析师请求库 Q_seed 为种子，挖掘源包 P_0，构建 DAG 证据图 G_0=(V_F∪V_O,A,ρ)，对终端事实 y 截取祖先子图并满足最小深度 h_min 与最小源广度 b_min；查询构造 R_ψ 完成后需满足 Eval(G_y)=y、Ans(q;G_y)={y}、Loc(G_y)=1、Expose(q;G_y)=0，防止暴露中间路径。
- **MintHarness**
  - **执行循环**：S_0=Init_ξ(T) 初始化任务输入、动作空间、预算向量与空证据账本/工作记忆/轨迹；每轮 a_t~π_θ(Pack_B(S_t))，由 Gate_ξ 校验动作合法性后得到可执行动作 ȧ_t。
  - **证据更新**：工具返回 r_t→Distill(r_t;x_T) 生成分类型溯源记录，Merge 入账本 L_{t+1}，失败调用仅留 observation 不入账本。
  - **状态转移**：S_{t+1}=F_ξ(S_t; ȧ_t,o_t,L_{t+1})，将新旧轨迹追加、预算推进、工作记忆刷新；终结时输出 (ŷ_T, Eval_ξ(ŷ_T;y), R_T)。
  - **模块设计**：证据账本持久化 typed 记录与失败路径；工作记忆保留当前计划、已确认结论、未决问题与中间计算并与账本链接；上下文管理按 token 预算从账本中选relevant条目压缩旧交互。
- **训练管线**
  - **财务推理分支（π_R）**：先在 D_atom 上做 SFT（使用 GLM-5.2/DeepSeek-V4-Pro 生成候选轨迹经正确性→模式过滤→轨迹选择三道门控）；再以 RLVR 优化，奖励 R_atom(o,T)=(A_κ·S·C_κ)∈{0,1}，按通过率动态维护 active batch B_m 并用 GSPO 做序列级更新。
  - **智能体执行分支（π_A）**：先在 D_long 上做 SFT（保留轨迹级与回合级门控，剔除重复/无效调用）；再做 Critical-Step OPD——在失败轨迹中标识关键回合，采用 GOLD 风格跨 tokenizer 最小块对齐，计算 token 级优势 Â^{OPD}_{t,l}=sg[Ñ^T_{t,l}−logπ_{old}(z_{t,l}|h_t,z_{<l})]，仅优化关键动作 token；最后以 RLVR 做轨迹级训练，奖励 R_long=(A_long·G·H)∈{0,1}，同样用 GSPO 更新。
  - **专家集成**：对 π_R 与 π_A 相对于基模型 θ_0 的 task vector 做 TIES 修剪/共符号选举/加权平均得到 π_M；再以 π_M 初始化学生 π_θ，固定 π_R 与 π_A 为教师，按 D_mix=α_R D_atom+α_A D_long 随机采样并按来源路由教师，执行 MOPD：A^{MOPD}_t=sg[logπ^T_T(z_t|h_t)−logπ_{θ−}(z_t|h_t)]，L_MOPD=−E[(1/L)Σ clip(A_t,h,−A_max,A_max)·logπ_θ(z_t|h_t)]，按学生自身状态生成监督信号，消除 off-policy 模仿差距。

## 实验与结果
- **模型与基线**：Mint-Cu（基座 Qwen3.5-9B）、Mint-Ag（基座 Qwen3.6-27B）；对比涵盖 frontier 模型（Gemini-3.5-Flash、Claude Opus 4.8、GPT-5.6-Sol、GLM-5.2、Kimi-K2.7-Code、DeepSeek-V4-Pro/Flash、MiMo-V2.5-Pro、Qwen3.7-Plus、MiniMax-M3）、开源模型（Agents-A1-35B、Nex-N2-mini、ASearcher-32B、OpenThinkerAgent-32B、OpenResearcher-30B-A3B、Tongyi-DeepResearch-30B-A3B）及 Agent 系统（Codex+GPT-5.6、Cursor+Grok 4.5、Cursor+Composer 2.5）。
- **基准**：BizFinBench、FinanceBench、RFC-Bench Task 2（原子推理）；FinanceAgentBench v1.1/v2、FinSearchComp T2/T3（长程执行）。
- **主要结果**
  - RFC-Bench：Mint-Ag **98.33%**，超 GPT-5.6-Sol 3.66 pts、超 Claude Opus 4.8 3.00 pts。
  - BizFinBench：Mint-Ag **55.71%**（+4.14 vs. 次优）、Mint-Cu **53.86%**。
  - FinanceBench：Mint-Ag **91.33%**、Mint-Cu **90.00%**。
  - FinanceAgentBench v1.1：Mint-Ag **76.00%**；v2：Mint-Ag **60.49%**（均值为 60.49±0.03）。
  - FinSearchComp T2：Mint-Ag **89.04%**；Mint-Cu **69.86%**（超 Agents-A1-35B 22.83 pts、超 Nex-N2-mini 12.78 pts）。
  - FinSearchComp T3：Mint-Ag **54.07%**（与 Codex+GPT-5.6 持平）。
- **成本表现**：两 checkpoint 均落在实证 Pareto 前沿；Mint-Cu 在 v1.1 以 $0.016/任务达 68.0% 准确；Mint-Ag 以 $0.090/任务达 76.0%（低于对比均值 72.5%）。相对 GPT-5.6-Sol，Mint-Ag 在 v2 上精度+3.70 pts 且成本降 77.8%（$0.213 vs. $0.959）。
- **集成消融**：MOPD 相对 TIES merge 在四项基准分别提升 1.33/3.86/4.00/3.19 pts（Finance Bench/BizFin/FAB v1.1/FinSearch T2）。
- **失败模式**：v1.1 最强残差为"答案遗漏"（M10, 18%）；v2 转向"证据抽取"（M5, 18.5%），反映更难基准的瓶颈从检索移到财务语义编码。

## 相关工作脉络
- **Agent 训练范式**（Luo et al., 2025; Cheng et al., 2025; Zhao et al., 2025; Chai et al., 2025; Zhou et al., 2026; Zhang et al., 2026c; Li et al., 2025a; Yu et al., 2025）：本文定位差异在于把策略学习、执行契约与可审计性在金融垂直域统一为单一证据契约，而非仅在通用环境上训练工具调用。
- **财务预训练/指令微调**（Wu et al., 2023; Yang et al., 2023; Xie et al., 2023; Liu et al., 2025）：本文将其视为起点，强调后训练阶段必须引入可验证奖励，从"词汇/约定覆盖"走向"可重放推导"。
- **披露驱动 QA 与反事实谣言检测**（Chen et al., 2021; Zhu et al., 2021; Chen et al., 2022; Islam et al., 2023; Jiang et al., 2026）：本文在其评测基础上进一步要求模型自主发现证据，并把正确性判定扩展到证据图可重放。
- **金融智能体/工具评测**（Hu et al., 2025b; Bigeard et al., 2025; Zhu et al., 2025; Haque et al., 2026; Cheng et al., 2026; Luan et al., 2026; Lu et al., 2026; Huang et al., 2026; Pauli et al., 2026; Srivastava, 2026; Xiao et al., 2026）：本文与之的区别是提供端到端数据→harness→训练的全栈方案，并给出完整的轨迹审计能力。
- **Deep Research（Harness-based）**（Li et al., 2025b; Yang et al., 2026; Zhang et al., 2026d; Cai et al., 2026; MindDR Team, 2026; Yan et al., 2026; Câmara et al., 2026; Jin et al., 2026）：本文吸收其可观察性优势，但通过训练内的 verifier 使策略本身也内化审计纪律。
- **Deep Research（Training-based）**（Tongyi DeepResearch Team, 2025; Hu et al., 2025a; Li et al., 2026b; Du et al., 2026; Dong et al., 2026; Xu et al., 2026; Xie et al., 2026; Hussain et al., 2026; Zhu et al., 2026b; Yu et al., 2026a; Zhu et al., 2026a; Wu et al., 2025; Lu et al., 2025b; Ye et al., 2025）：本文指出纯训练路线可能产出"流畅但不可信"的报告，强调需与证据型 harness 耦合，并给出 TIES+MOPD 的合并策略。

## 局限性与未来方向
- **任务域局限性**：数据与评估均聚焦上市公司披露、市场与宏观数据，尚未覆盖衍生品定价、组合优化、风控建模等其他金融子领域。
- **失败模式分布随难度迁移**：v1.1 的主导误差是"答案遗漏"（执行收尾弱），v2 则转为"证据抽取"（感知—工作记忆接口弱），说明不同难度下瓶颈不同，现有架构未完全解决选择性注意力问题。
- **对教师模型依赖**：SFT 阶段使用 GLM-5.2 与 DeepSeek-V4-Pro 生成轨迹，关键步骤 OPD 的教师评分亦来自大模型 judge，噪声或偏差可能传导至学生。
- **可复现性约束**：W_t_cut 时间切片快照机制依赖源站存档，若原始页面下线则证据链可能断裂；Q_seed 去重需避免与评测集重叠。
- **未来方向**：扩展至更多金融垂直子任务、探索零样本/少样本跨域泛化、降低对教师模型的依赖、将证据图扩展到动态图结构、结合自改进闭环持续精炼 verifier。

## 研究启发与可借鉴点
- **TIES + 多教师 OPD 的通用合并范式**：对"共享基模型→多专家→权重合并→on-policy 对齐"的流水线，可作为其他垂直领域（法律、医疗、代码）多专家统一的标准做法直接迁移。
- ** verifier 三因子乘积形式**：R=(答案等价)×(溯源支撑)×(领域规则重放) 的设计可推广到任何需要"答案+证据+过程合规"三要素的领域（如法规合规、科学计算）。
- **Critical-Step OPD 的精修策略**：在失败轨迹中标识"改变该步可恢复正确路径"的关键回合做跨 tokenizer 最小块对齐 distillation，避免浪费计算在无害回合，适合任意长程 tool-use 训练。
- **MintHarness 的持久账本+有界上下文解耦**：Evidence Ledger 作为外部可审计记忆、工作记忆仅保留当前计划的模式，可直接复用到需要长轨迹追溯的任何 agent 应用。
- **难度感知的 GSPO active batch 动态维护**：以组通过率 p_j∈(0,1) 筛除太易/太难样本并在线刷新批次，可与任意基于 rollout 的策略优化算法组合使用。

## 关键术语表
- **Mint-Agent**：面向金融领域的原生智能体基础模型家族，包含 Mint-Cu（9B）与 Mint-Ag（27B）两个旗舰模型。
- **MintHarness**：证据优先的智能体执行框架，通过 Evidence Ledger、工作记忆与有界上下文解耦实现长程金融任务的持久化可审计执行。
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：利用结构化 verifier 对原子推导或长程轨迹给出二值奖励，并在 GSPO 下做序列级策略优化的训练方式。
- **GSPO（Group Sequence Policy Optimization）**：基于 token 级几何平均 importance ratio 的序列级策略优化算法，适用于 verifier 打分下的整段轨迹优化。
- **TIES（Task-Vector Interaction for Expert Synthesis）**：对多专家相对于基模型的参数增量做低幅裁剪与共符号选举后加权平均的权重合并方法。
- **MOPD（Multi-Teacher On-Policy Distillation）**：按任务来源路由不同冻结教师，在学生自身生成状态上最小化反向 KL 的多教师 on-policy 蒸馏。
- **Evidence Ledger**：持久化存储工具调用产生的已提炼事实（含溯源定位、财务范围与获取元数据）的外部账本结构。
- **Financial Surface W_t_cut**：以截止时间 t_cut 索引的金融信息快照集合，保证跨时间访问的可复现性。

## 可复现要素
- **数据集**：原子任务来自 EDGAR、XBRL、交易所记录、FRED 与会计税则本体；长程任务在 D_atom 同源基础上叠加时间切片 W_t_cut 与分析师请求种子库 Q_seed；评测基准为 BizFinBench、FinanceBench、RFC-Bench Task 2、FinanceAgentBench v1.1/v2、FinSearchComp T2/T3。论文未提供公开的数据集下载链接。
- **代码**：论文仅给出技术报告形式描述，附录提供评测 prompt 模板；未明确声明代码仓库开源地址。
- **权重**：Mint-Cu 与 Mint-Ag 基座分别为 Qwen3.5-9B 与 Qwen3.6-27B，论文中未提供 mint-agent 最终权重的公开下载链接。
- **关键超参**：温度 0、重复惩罚 1.1、每任务 128K token 上下文；RL 中每组 G 次 rollout、通过率阈值 0.8 进入初始 RL 池、B_m 按 0<p_j<1 动态维护；MOPD 中 A_max 为裁剪边界（具体数值论文未列明）；SFT 教师为 GLM-5.2 与 DeepSeek-V4-Pro。
