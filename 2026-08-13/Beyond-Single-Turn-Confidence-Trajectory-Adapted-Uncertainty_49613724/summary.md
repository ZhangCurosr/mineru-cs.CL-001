---
title: "Beyond-Single-Turn-Confidence-Trajectory-Adapted-Uncertainty"
source: https://arxiv.org/pdf/2608.11552v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 08:37:26"
field: "LLM Agent 不确定性与可靠性"
keywords: ["Uncertainty Quantification", "LLM Agents", "Trajectory-level Confidence", "Tool-use Agents", "Self-consistency", "Agent Safety"]
innovations: ["系统比较三类单次生成UQ方法在LLM Agent多轮轨迹场景的迁移表现", "提出动作结构化一致性和LLM裁判轨迹等价率等新度量", "揭示白盒聚合策略的关键影响及黑盒一致性的最强性能"]
benchmarks: ["BFCL-v4", "τ²-bench"]
---

# 论文速读：Beyond-Single-Turn-Confidence-Trajectory-Adapted-Uncertainty

## 一句话总结
本文系统评估了三类经典单次生成不确定性量化（UQ）方法——白盒（token概率）、黑盒（轨迹重采样一致性）与反思式（模型自评）——在LLM Agent多轮工具调用轨迹场景中的迁移表现，发现黑盒一致性（尤其是TER和ASC）整体最强，但无单一方法在所有模型/数据集上稳定可靠，需按部署场景选择并验证聚合策略。

## 研究问题与动机
1. **单次UQ向轨迹级迁移的适用性不明**：现有UQ方法仅在单轮输出上验证，而Agent的预测单元是多步交互轨迹，早期错误会沿轨迹传播至最终结果，需要重新评估这些方法是否仍有效。
2. **白盒分数严重依赖跨轮聚合方式**：同一token概率基础分数在不同聚合规则（mean/min/first/last等）下AUROC可相差极大（如SP_mean在telecom为0.73，但SP_last仅为0.23），缺乏对部署场景的适配指导。
3. **黑盒一致性在Agent场景的度量设计存在多样性**：从最终消息一致性到动作结构一致性再到轨迹等价判断，不同一致性度量在Agent场景下的相对优劣未系统研究。
4. **Agent UQ缺乏统一基准对比**：此前工作如SAUP、UProp等提出传播方法，但尚未在有匹配条件下对三类方法家族进行跨模型、跨数据集的大规模比较。

## 核心贡献（创新点）
1. **首次在五个LLM和四个多轮工具调用数据集上系统比较三类UQ方法**：提供白色盒（基于token概率）、黑色盒（基于重采样轨迹一致性）和反思式（基于模型自评）UQ方法的受控实证对比，覆盖区分度、校准性和选择性预测三个维度。
2. **提出轨迹级白盒聚合框架与新型黑盒一致性度量**：将单次生成白盒分数扩展为跨轮聚合（first/mean/min/last及位置加权），并设计动作结构化一致性度量（FAC/ASC/ADC/AEC）及基于LLM裁判的轨迹等价率（TER）。
3. **揭示三类方法各自的失效模式与适用边界**：发现白盒方法高度依赖聚合规则且部分出现负相关（91/240个单元格<0.5），反思式提供最具通用性的低成本基线，黑盒一致性（TER/ASC）在多数设置下表现最佳，但无一家族在所有条件下稳定可靠。
4. **提供计算成本-性能权衡的实用指导**：量化各方法家族的额外计算开销（白盒零开销、反思式一次自评估pass、一致性需m次重采样+辅助judge），指出实际部署中不同预算下的最优选择策略。

## 方法详解
**总体框架**：以Oh et al. (2026)的Agent-UQ形式化为基础，将episode定义为长度为T的轨迹F≤T，其中每轮i发射动作A_i（工具调用或用户消息）并获得环境观察O_i，轨迹完成后由二元奖励r(F≤T)记录成功与否。目标是估计整条轨迹F的不确定性，使高置信度分数对成功轨迹排序更高。

**（1）白盒评分器（White-Box）**：
- 基础分数：序列概率SP = ∏p_j、长度归一化序列概率LNSP = (∏p_j)^(1/L)、平均token负熵ATN@K = (1/L)Σ_j(1 - TE@K(t_j)/log K)。
- 动作跨度选择：分数仅计算于每轮的"承诺动作"token（工具调用或消息），不含自由推理。
- 跨轮聚合：g_mean（均值）、g_min（最小值，对应最不确定步规则）、g_first/g_last（首/末轮分数）、早期/晚期位置加权聚合。

**（2）黑盒一致性评分器（Black-Box Consistency）**：
- 最终消息一致性：NCP = 1 - (1/m)Σ_j [p_c(y,ỹ_j) + p_c(ỹ_j, y)] / 2，使用DeBERTa-large-mnli做NLI矛盾概率测量。
- 动作结构化一致性：用κ(A)表示动作类型（工具名或用户消息），定义：FAC（首动作匹配率）、ASC（Jaccard相似，忽略顺序与重复）、ADC（基于JS散度的动作分布一致）、AEC（基于归一化Levenshtein编辑距离，敏感于顺序与长度）。
- 轨迹等价判断：TER使用gemini-flash-lite作为独立LLM裁判J，阅读两条完整轨迹后判定"任务相关结果等价"（而非正确性），取m次重采样中的等价比例。裁判仅见轨迹转录文本，不接触隐藏任务指令。

**（3）反思式评分器（Reflexive）**：
- P(True)：让模型判断轨迹是否成功（True/False），置信度为"True"token的概率。
- VC（Verbalized Confidence）：让模型给出Yes/No判断并附带0-1置信概率，Yes时取 stated probability，No时取其补数。
- 两者均在最终动作前的转录前缀上进行一次评估，不利用后续用户确认信息。

**（4）损失/评估函数**：以AUROC为主指标（预测轨迹成功），辅以AUPRC、选择性预测风险覆盖率（PRR）和期望校准误差（ECE）；采用1000次任务聚类bootstrap获得95%置信区间。

## 实验与结果
**数据集**：BFCL-v4多轮子集（200个任务）+ τ²-bench的airline（50）、telecom（114）、retail（114）三个文本域。共五个模型：Qwen2.5-7B、gpt-oss-20b、Qwen3.5-9B、MiniMax-M3（Together AI）和gpt-4o-mini（OpenAI）。每任务记录一次greedy轨迹（T=0，K=5 top logprobs）和三次采样轨迹（T=0.7）。

**主要结果（AUROC）**：

| 方法家族 | 平均AUROC | 最高单单元格 | 额外开销 |
|---|---|---|---|
| White-box | 0.628 | 0.725 (SP_mean, Qwen3.5-9B/telecom) | 0 |
| Reflexive | 0.691 | 0.885 (VC, gpt-4o-mini/BFCL-v4) | 1次自评估pass |
| Action consistency | 0.705 | 0.849 (ASC, gpt-oss-20b/airline) | m=3采样+NLI |
| Message consistency | 0.686 | 0.868 (NCP, Qwen3.5-9B/airline) | m=3采样 |
| Judged consistency | — | 0.87 (TER, Qwen3.5-9B/airline) | m=3采样+Gemini judge |

**关键发现**：
- **白盒不稳定**：91/240个单元格AUROC<0.5；相同基础分数切换聚合规则可导致剧烈波动（如LNSP在retail上min聚合0.72但mean聚合仅0.34；SP在airline上mean聚合接近随机）。SAUP-inspired传播控制在τ²-bench上平均仅0.434 AUROC，低于任何固定朴素聚合。
- **反思式稳健**：在BFCL-v4上P(True)达到0.75–0.85，在τ²-bench上最高达telecom的0.852（P(True), gpt-4o-mini），但airline上差异极大（P(True)从0.225到0.659）。
- **黑盒一致性最强**：TER在airline上达0.87（Qwen3.5-9B），ASC在多个单元格中排名前二；但TER在低成功率场景（telecom 11%成功率）下降至0.34。
- **样本量敏感性**：TER AUROC随m从1增至3时从0.63升至0.68（airline从0.62升至0.69），多个曲线在m=3时尚未饱和。
- **界面鲁棒性**：gpt-4o-mini在retail上native tool-calling vs text-action的TER AUROC几乎相同（0.761 vs 0.759）。

## 相关工作脉络
1. **单次生成UQ三大家族**：Kadavath et al. (2022) P(True)、Manakul et al. (2023) SelfCheckGPT、Farquhar et al. (2024) Semantic Entropy等奠定了白盒/黑盒/反思式三类基础；本文将其系统迁移至Agent轨迹场景。
2. **Agent不确定性传播**：Zhao et al. (2025) SAUP通过情境依赖权重传播步级不确定性，Duan et al. (2025) UProp分离当前决策不确定性与继承不确定性；本文发现传播控制在action-level traces上整体弱于朴素聚合。
3. **轨迹级Agent UQ形式化**：Oh et al. (2026) 首次以联合不确定性框架正式定义Agent轨迹不确定性（含初始查询、动作、观察三部分贡献），本文在其形式化基础上进行大规模实证比较。
4. **工具调用Agent校准**：Liu et al. (2024) ProbeCal使用执行轨迹和embedding探针校准工具使用Agent；Xuan et al. (2026) 发现证据类工具诱导过度自信而验证类工具可缓解校准偏差。
5. **多轮对话置信度**：Zhang et al. (2026a) 研究多轮交互中的逐轮校准与信息累积单调性；本文关注整条轨迹层面的成功预测而非逐轮校准。
6. **结构化不确定性引导澄清**：Suri et al. (2026) 对工具调用参数建模结构化不确定性以决定何时向用户澄清；本文则聚焦事后轨迹级成功预测，而非执行中干预。

## 局限性与未来方向
1. **基准范围有限**：仅覆盖BFCL-v4多轮子集和τ²-bench三个文本域，未扩展到网页浏览、具身Agent或更开放任务；结论的外推性未知。
2. **标签噪声上限**：轨迹成功标签本身含噪声（部分来自模拟器而非Agent），限制了任何UQ评分器的 discrimination 上限。
3. **用户模拟器混淆**：重采样同时作用于Agent和用户模拟器，黑盒一致性度量的是联合系统变异性而非Agent孤立不确定性（虽有ablation部分隔离）。
4. **动作接口局限**：白盒分数依赖文本动作接口获取token概率，native tool-calling无法在全部模型上提供可比surface；仅在一个模型-域上做了ablation验证。
5. **统计效力不足**：airline仅50个任务、部分单元格少数类<20，bootstrap区间宽，单元格内排名多未resolved。
6. **Judge依赖**：TER依赖单一外部judge模型（gemini-flash-lite），虽在database-graded域上达到precision 0.88/recall 0.93，但未测试所有域或更远的judge模型。
7. **未评估在线干预**：所有分数在轨迹完成后计算，未研究执行中基于分数的 abstention/clarification/escalation 是否能真正改善结果。

## 研究启发与可借鉴点
1. **跨轮聚合策略需显式建模与验证**：白盒方法并非"即插即用"，聚合规则对性能影响巨大；后续工作可探索数据驱动的自动聚合选择或per-domain tuning策略。
2. **黑盒一致性设计可进一步细化**：TER和ASC表现突出提示"结果等价"比"表面形式一致"更贴合Agent场景；可探索融合语义理解的动作语义等价度量。
3. **反思式作为低成本基线值得深度挖掘**：仅需一次额外pass即可达到接近黑盒一致性的性能，在资源受限部署中极具实用价值；可研究提升其校准性的prompt工程。
4. **"stuck-policy"诊断框架**：本文通过success-success/success-failure/failure-failure四象限分析揭示一致性可能同时反映"重复成功"和"重复失败"，这一诊断思路可推广至其他一致性-based方法的可靠性分析。
5. **成本-性能权衡框架可直接复用**：Table 2的成本-性能汇总表提供了清晰的部署决策树，可复用于其他Agent UQ工作的对比基线设计。

## 关键术语表
**Uncertainty Quantification (UQ)**：对模型输出的不确定度进行数值化估计，以区分高/低置信预测。
**Trajectory**：Agent在多轮交互中产生的完整动作-观察序列，是Agent场景下不确定性评估的基本单元。
**White-box Scorer**：利用模型内部token概率信息（无需额外采样）计算置信度的方法。
**Black-box Consistency Scorer**：通过对同一输入多次重采样并比较输出一致性来估计不确定度的方法。
**Reflexive Scorer**：要求模型对自身生成轨迹进行自我评估以获得置信度的方法。
**Trajectory Equivalence Rate (TER)**：使用独立LLM裁判判定重采样轨迹与参考轨迹在任务结果上是否等价的百分比。
**Action-Set Consistency (ASC)**：基于Jaccard相似度比较参考轨迹与采样轨迹所用工具类型集合的一致性度量。
**Prediction Rejection Ratio (PRR)**：选择性预测指标，衡量通过拒绝低置信样本后提升成功率的效果，0为随机，1为最优。

## 可复现要素
- **数据集**：BFCL-v4（Apache-2.0许可）和τ²-bench（MIT许可），均通过官方渠道公开获取。
- **代码/权重**：论文未提供开源代码仓库链接，但提到了UQLM Python包（Bouchard et al., 2026b）；模型通过Together AI和OpenAI API服务。
- **关键超参**：重采样轨迹数m=3，采样温度T=0.7，参考轨迹温度T=0，top-K=5；NLI模型使用microsoft/deberta-large-mnli；TER裁判使用gemini-flash-lite（T=0）。
- **评估指标**：AUROC（主指标，1000次任务聚类bootstrap），AUPRC，PRR，ECE。
