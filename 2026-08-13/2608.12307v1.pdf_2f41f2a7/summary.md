---
title: "AI4AI at Test-Time: Strong-to-Weak Capability Transfer via Harnesses"
source: https://arxiv.org/pdf/2608.12307v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:01:44"
field: "LLM推理优化与能力迁移"
keywords: ["strong-to-weak scaffolding", "test-time capability transfer", "inference-time harness", "deterministic offloading", "cognitive load reduction", "Theory of Mind", "model distillation alternative", "agent harness design"]
innovations: ["形式化强到弱测试时脚手架为独立能力迁移范式，不更新目标模型参数", "发现确定性卸载（r=0.72）和headroom-gated提升是核心机制", "提出scaffolding作为builder模型评估新基准"]
benchmarks: ["BigToM", "Hi-ToM", "MMToM-QA", "MuMA-ToM"]
---

# 论文速读：AI4AI at Test-Time: Strong-to-Weak Capability Transfer via Harnesses

## 一句话总结
本文提出一种测试时强到弱能力迁移框架：强builder模型仅需5%验证数据即可迭代设计推理时harness（脚手架），将任务结构外化为确定性规则、路由逻辑和格式约束，无需更新目标模型参数，使GPT-5.4-mini在Theory-of-Mind基准上的宏平均准确率从0.488提升至0.912。

## 研究问题与动机
1. **现有蒸馏方法的局限**：主流知识蒸馏、on-policy蒸馏、指令微调等均依赖更新弱模型参数，需要额外训练成本与数据；本文探索"不改变目标模型权重"的迁移路径。
2. **推理时harness设计的理论缺口**：虽已有CoT、Self-Consistency、工具调用等方法，但缺乏系统分析——哪些设计选择真正有效？增益来自何处？何时稳定？
3. **部署场景的现实需求**：小模型在agentic pipeline中常与路由、验证、工具调用等外部结构共存，但缺乏对"强模型为弱模型构建推理环境"的系统研究。
4. **能力评估的新视角**：传统benchmark问"模型能解决什么问题"，本文转向"强模型能否构建让弱模型解决问题的条件"，为builder能力提供新评估维度。

## 核心贡献（创新点）
1. **形式化强到弱脚手架（strong-to-weak scaffolding）为独立的测试时能力迁移范式**，与训练时蒸馏形成互补——能力通过外部harness而非权重更新传递。
2. **系统性实证分析**，覆盖72次实验（11种builder×3平台×2 target×重复），揭示harness设计的有效性机制：确定性卸载、基准感知路由、严格格式控制比延长推理或增加采样更有效。
3. **发现"认知负载缩减"是核心因果机制**：最佳harness将72%以上的推理工作外化为可执行代码/规则，目标模型仅需处理剩余难以编译的case。
4. **提出"harness自我进化"与"scaffolding benchmark"的未来方向**，为agentic system设计提供可度量的自动化脚手架构建范式。

## 方法详解
**问题设定**：给定目标模型$M_{tar}$、builder模型$M_{build}$、基准$\mathcal{D}^{(j)}$，随机抽取5%作为验证集$\mathcal{V}^{(j)}$，剩余为隐藏测试集$\mathcal{T}^{(j)}$。Builder仅见$\mathcal{V}$，通过迭代优化构建harness $S$，最终在$\mathcal{T}$上评估$S(M_{tar})$。

**算法流程（Algorithm 1）**：
1. 初始化builder工作空间$\mathcal{W}_0 = \{\mathcal{R}, C_{demo}, \mathcal{V}\}$，其中$\mathcal{R}$为任务规则文件，$C_{demo}$为目标模型调用示例。
2. 迭代循环：builder读取资源→提出/修订harness $S_k$→在$\mathcal{V}$上评估得到准确率$a_k$→诊断错误集$\mathcal{E}_k$→更新工作空间$\mathcal{W}_{k+1}$。
3. 提交可执行的入口脚本$f_{\hat{S}}(x; M_{tar})$，由人工在$\mathcal{T}$上运行并计算最终准确率。

**搜索空间$\mathcal{S}$**：builder可自由实现任意推理时程序，包括prompt模板、基准路由、确定性前后处理、few-shot检索、验证通过、格式约束、符号求解器等，唯一约束是最终harness需暴露通用入口。

**优化目标**：$\hat{S} = \arg\max_{S \in \mathcal{S}_{build}} \text{Acc}(S, M_{tar}; \mathcal{V})$，以验证集准确率为代理，避免过拟合。

## 实验与结果
**数据集**：聚合4个Theory-of-Mind基准共3900题：
- BigToM（1200题）：二元信念/目标/行动问题，基于观察/未观察区分
- Hi-ToM（1200题）：递归阶数0–4的嵌套信念，含欺骗场景
- MMToM-QA（600题）：基于动作轨迹的贝叶斯目标/信念推断
- MuMA-ToM（900题）：多智能体信念/社会目标/目标信念的3选1

**基线**：
- Vanilla：GPT-5.4-mini直接调用，宏平均0.488；Gemini-3.5-flash为0.761
- UserHarness（人工设计）：GPT-5.4-mini达0.939，Gemini-3.5-flash达0.941

**主要结果（GPT-5.4-mini作为target）**：
| 配置 | 宏平均准确率 | 提升幅度 |
|------|-------------|---------|
| Vanilla baseline | 0.488 | — |
| 所有scaffolded runs均值 | 0.763 | +0.275 |
| 最佳run（GPT-5.5 on GPT Codex） | 0.912 | +0.423（86.7%相对提升） |
| 超越vanilla GPT-5.4（0.619）和GPT-OSS-120B | ✓ 全部基准 | — |

**关键发现**：
- 100%的builder配置均正提升，无退化
- 最佳harness在BigToM上达1.000，超越人工UserHarness（0.95）
- Hi-ToM、MMToM、MuMA仍有差距（0.80 vs 0.87，0.84 vs 0.98，0.88 vs 0.96）
- 验证集效率：中位数仅5次评估，最佳验证准确率与最终测试准确率高度相关（Pearson r=0.96），乐观偏差仅0.021

**Target依赖（headroom law）**：GPT-5.4-mini提升+0.262，Gemini-3.5-flash仅+0.110；提升量与目标模型在该基准上的可用headroom（1−baseline）强相关（r=0.75）。

**Builder reasoning effort**：Opus-4.7从low到extra-high，准确率单调上升（0.711→0.856，Spearman ρ=0.77）。

**技术归因**：极性/否定逻辑（+0.09）、结构化提取（+0.06）、混合回退（+0.04）关联最强；确定性卸载率与准确率强相关（r=0.72）。

## 相关工作脉络
1. **知识蒸馏（Hinton et al., 2015; Hsieh et al., 2023; Agarwal et al., 2024）**：通过teacher forcing、on-policy训练更新student参数；本文与之互补，不更新参数而是构建推理时环境。
2. **推理时提示与分解（Wei et al., 2022; Wang et al., 2022; Zhou et al., 2022; Yao et al., 2023）**：CoT、Self-Consistency、Tree-of-Thoughts等优化同一模型的推理过程；本文研究跨模型脚手架构建。
3. **工具调用与程序辅助推理（Schick et al., 2023; Yao et al., 2022; Gao et al., 2023; Chen et al., 2022）**：PAL、Program-of-Thoughts将计算卸载到Python解释器；本文系统化研究builder自动发现此类卸载的机制。
4. **Harness工程与自动化设计（Khattab et al., 2023; Yang et al., 2024; Hu et al., 2025; Lee et al., 2026）**：DSPy、ADAS、Meta-Harness等将harness作为优化目标；本文聚焦强到弱迁移的具体范式与机制分析。
5. **Theory-of-Mind基准（Gandhi et al., 2023; Wu et al., 2023; Jin et al., 2024; Shi et al., 2025; Chen et al., 2024）**：BigToM、Hi-ToM、MMToM-QA、MuMA-ToM等提供嵌套信念、递归深度、贝叶斯推断等压力测试；本文用其作为scaffolding能力的元评估床。
6. **弱到强泛化（Burns et al., 2023）**：弱监督能否 eliciting 强模型能力；仍属参数更新范式，与本文测试时环境构造不同。

## 局限性与未来方向
1. **基准局限性**：仅在ToM领域验证，未涵盖符号数学、代码生成、开放域问答等其他任务类型；不同任务的"可编译性"差异待探索。
2. **残余误差的硬边界**：高阶递归（Hi-ToM order≥2）、欺骗场景、贝叶斯目标推断（MMToM type-2）仍是harness难以处理的core difficulty，需更强显式信念追踪机制。
3. **平台效应的条件性**：native platform优势仅在builder reasoning effort充足时显现，但未见统计显著性（p=0.484），实际可移植性虽强但前沿场景仍需 tuned。
4. **过脚手架风险**：对已接近天花板的目标模型（如Gemini-3.5-flash在Hi-ToM/MuMA），额外规则可能干扰正确行为（9/20 cases出现regression）。
5. **评估范式尚未标准化**：作者提议将scaffolding发展为builder benchmark，但目前缺乏统一workspace、评分标准与公开数据集。

## 研究启发与可借鉴点
1. **确定性卸载作为核心杠杆**：将任务中可编译的子结构外化为代码/规则，比鼓励弱模型"想更多"更有效；可在代码生成、数值计算等场景迁移。
2. **Headroom-gated应用策略**：scaffolding价值与目标模型剩余headroom正相关；对强模型应子任务级选择性应用，对弱模型全面受益——这一原则可指导部署决策。
3. **验证效率优先于验证预算**：builder质量比验证调用次数更重要（r=0.17不显著）；少量高质量迭代胜过大量浅层 probing，可指导compute allocation。
4. **跨builder complementarity集成**：不同builder发现的repair机制部分重叠但非完全冗余（union覆盖97% baseline errors）；可考虑ensemble多个独立scaffold。
5. **作为builder能力评估的新范式**：不仅问"模型能解什么题"，更问"模型能否设计让弱模型解题的环境"；可设计标准benchmark用于agent harness设计能力评估。

## 关键术语表
**Strong-to-Weak Scaffolding**：强builder模型为固定弱target模型构建推理时外部脚手架，通过任务路由、确定性求解、格式约束等手段转移能力，不更新target参数。

**Harness / Scaffold**：围绕目标模型的外部推理环境，包含prompt模板、路由逻辑、验证检查、工具调用、符号求解器等组件的集合。

**Deterministic Offloading**：将易出错的语言模型推理替换为确定性代码/规则执行，减少target模型的认知负载，是性能提升的主要机制（r=0.72）。

**Headroom Law**：scaffolding带来的提升量与目标模型在特定基准上的可用headroom（1−baseline准确率）强相关（r=0.75），headroom越大增益越显著。

**Validation-Efficiency**：builder仅用5%验证数据、中位数5次评估即可收敛到高质量harness，验证预算不是性能瓶颈，builder自身能力才是。

**Compilability Ranking**：不同ToM基准对harness编译的适应性不同：BigToM几乎完全可编译（det. frac≈0.94），MuMA-ToM最难编译（≈0.36）。

**Over-scaffolding Risk**：当target模型已接近天花板时，额外规则/路由可能破坏已有正确行为，导致regression（如Gemini-3.5-flash在Hi-ToM上-0.04）。

**Test-Time Compute vs. Training-Time Compute**：本文发现有效的test-time compute是builder的推理努力（reasoning effort），而非对验证集的重复查询次数。

## 可复现要素
- **数据集**：BigToM、Hi-ToM、MMToM-QA、MuMA-ToM均为公开benchmark；验证集按固定随机种子抽取5%（195题）
- **代码/权重**：论文未明确声明开源，但提供了完整算法描述与超参设置
- **关键超参**：
  - 验证集比例：5%
  - 目标模型：GPT-5.4-mini、Gemini-3.5-flash
  - Builder模型：Opus-4.7（4档reasoning effort）、Sonnet-4.6、GPT-5.5、GPT-5.4-mini、Codex-5.3、Gemini-3.1-Pro、Gemini-3.5-flash、Grok-0.1
  - Platform：Cursor、Claude Code、GPT Codex
  - 重复次数：3次/配置
  - 总实验数：72 runs
