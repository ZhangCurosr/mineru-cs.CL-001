---
title: "When-Should-Multi-Round-RAG-Stop-Structured-Stopping-Judgmen"
source: https://arxiv.org/pdf/2608.13237v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:24:10"
field: "检索增强生成的停止决策"
keywords: ["multi-round RAG", "stopping policy", "structured judge", "Search-R1", "S2G-RAG", "adaptive retrieval", "early stopping"]
innovations: ["将S2G结构化充分性与缺口判断适配到冻结Search-R1并针对性训练judge", "提出轨迹感知四维评估框架分离可达性/排序/首次穿越/质量成本", "证实检索减少3.70%同时保持EM损失在2%容忍范围内但非安全停止"]
benchmarks: ["HotpotQA distractor development set"]
---

# 论文速读：When-Should-Multi-Round-RAG-Stop-Structured-Stopping-Judgmen

## 一句话总结
论文针对多轮RAG系统中"何时停止检索"的序列决策问题，将S2G-RAG的结构化充分性与缺口判断机制适配到冻结的Search-R1管道中，训练了一个Qwen3.5-2B judge。在HotpotQA确认测试集上，该策略将检索调用减少3.70%，同时将答案准确率下降控制在0.625个百分点（在预设的2个百分点容忍范围内），但并未实现安全停止或整体成本降低。

## 研究问题与动机
- **序列选择与状态分类的本质差异**：部署策略由轨迹上第一个STOP决定，而非独立的状态分类任务；早期误判会抑制所有后续状态，而完全保守的分类器对系统无影响。
- **强搜索策略仍存在停止优化空间**：探索性审计发现，Native Search-R1有27.3%的SEARCH动作发生在冻结答案探针已能生成正确答案的状态，表明存在"过度检索"现象。
- **现有停止方法缺乏因果归因**：SIM-RAG、TASR等方法报告的状态级分数或端到端平均值无法解释为何部署的停止策略成功或失败，缺少从候选可达性到系统级风险的完整评估链条。
- **检索减少不等于总效率提升**：Judge本身消耗推理计算，且论文未归一化wall-clock时间或token等效成本，因此需严格区分"检索调用减少"与"总推理成本降低"。

## 核心贡献（创新点）
- **适配S2G到Search-R1并针对性训练**：将结构化充分性与缺口判断机制迁移到Search-R1的状态接口，在900个独立HotpotQA问题训练的3,009个状态上微调Qwen3.5-2B judge，而Search-R1的reasoner、retriever、corpus、prompt和搜索预算保持冻结。
- **轨迹感知评估框架分离五个维度**：提出将候选可达性、状态排序、首次穿越策略行为、答案质量和成本分开评估的机制梯级，揭示binary监督提升平均精度却损害系统的ranking-policy mismatch现象。
- **证实检索减少但非安全停止**：在保留索引200–999的确认测试集上，完整策略减少77次检索调用（3.70%），EM下降0.625个百分点，通过预设的非劣效性检验（CI下限>−0.02），但同时指出69次早期停止中39.13%为不安全停止，校准性能较弱（ECE=0.2911）。

## 方法详解
- **状态定义与可达性探针**：每个问题q_i产生轨迹状态z_{i,t}=(q_i, C_{i,t}, h_{i,t}, a_{i,t}^{native})，其中C为累积检索上下文，h为推理历史。冻结的答案探针g(q_i, C_{i,t})生成候选答案ã_{i,t}，若与官方答案匹配则标记y_{i,t}=1，构成"安全早期停止机会"。
- **结构化judge输出**：Qwen3.5-2B judge输出固定JSON schema，包含布尔字段sufficient和缺口列表gap_items；训练目标为顶层键顺序（sufficient先于gap_items），使用3个epoch共12,654步训练，所有loss有限且无OOM。
- **阈值冻结与策略部署**：在分组验证集上选择使recall最大化的阈值θ，约束STOP precision≥0.90且至少有10个预测STOP状态；若无可行阈值则使用fallback=sentinel（最大验证分+1）。Expanded judge冻结于margin≈7.875。
- **在线策略公式**：对于阈值θ，在线策略在τ_i(θ)=min{t: s_{i,t}≥θ}时停止，若无非交叉则遵循冻结的native fallback。主要配对效应定义为Δ_EM和Δ_search，置信区间使用10,000次问题级配对bootstrap重采样（seed=20260728）。
- **条件oracle诊断**：质量oracle选择最早可达正确探针答案（EM上界），成本oracle仅对native已答对的问题提前停止（固定native EM），两者均不可部署但用于评估headroom。

## 实验与结果
- **数据集与基线**：主实验使用HotpotQA distractor开发集前1,000题；上游为Search-R1 Qwen2.5-7B PPO v0.2 + E5 top-3检索 + Wiki-2018 + 最多4次搜索；对比基线包括Structured Base judge、Old S2G LoRA、Native Search-R1。
- **探索性审计结果**：200题上523个native SEARCH中27.3%发生在正确候选状态；质量oracle达51.5% EM（+5.0pp），成本oracle移除25.05%搜索且保持native EM；固定预算k=0,1,2,3,4的EM分别为18.5%, 34.0%, 43.0%, 46.0%, 47.0%。
- **确认测试集主结果**（索引200–999）：Native EM=0.44875，搜索次数=2.60125；Expanded S2G EM=0.44250，搜索次数=2.50500；Δ_EM=−0.00625（95% CI [−0.01250, 0]），Δ_search=−0.09625（95% CI [−0.12000, −0.07375]）；绝对减少77次检索（3.70%），6题变错1题变对。
- **判定指标**：Expanded S2G STOP precision=0.6216，recall=0.0880，AP=0.40358；早停69次中安全42次、不安全27次（不安全率39.13%）；Brier=0.2952，ECE=0.2911；验证集precision 0.9091未转移到测试集（降至0.6216）。

## 相关工作脉络
- **Search-R1（Jin et al., 2025）**：本文的上游冻结策略，使用RL训练LLM推理与搜索调用；本文不改进其reasoner/retriever，仅干预停止决策，使oracle条件于冻结接口而非全局上界。
- **S2G-RAG（Li et al., 2026）**：提出结构化充分性与缺口判断的judge；本文将其机制适配到Search-R1状态接口并针对性训练，而非复现完整S2G系统或训练Search-R1本身。
- **SIM-RAG（Yang et al., 2025）**：直接从生成的答案和推理中学习是否继续搜索；本文试点发现跨系统迁移失败（200题上8次有害变更、0次救援），作为探索性证据保留。
- **Self-RAG（Asai et al., 2023）**：学习检索与反思决策，通过generation-time控制token触发检索或批判；本文关注answer-level控制器，二者oracle与错误语义不同。
- **TASR（Kieback et al., 2026）**：training-free自适应停止；本文使用learned judge，强调需分离可达性、排序、首次穿越和政策行为进行四维评估。
- **选择性分类与认证风险（Geifman & El-Yaniv, 2017; Kang et al., 2024）**：概念上分离接受预测的条件错误与coverage；本文使用相同分离框架但不声称形式化风险保证。

## 局限性与未来方向
- **非安全停止规则**：27/69早期停止不安全（39.13%），验证集0.90 precision约束未转移到测试分布；部署需额外风险控制在程序、更多校准数据或更保守策略。
- **总成本未归一化**：检索调用减少不代表总推理成本降低；judge自身消耗compute、内存和latency，需matched-latency和matched-energy评估。
- **校准性能弱**：Expanded S2G的Brier和ECE显著劣于Structured Base；raw confidence scale校准不足，post-hoc 0.90 precision下recall仅0.0057。
- **候选可达性边界**：若冻结状态下无正确答案可生成，改变停止时间无法使问题变对；headroom受限于frozen reachable states。
- **未来方向**：校准感知训练、轨迹级损失惩罚首次不安全穿越、显式成本敏感目标、将AP与calibrated decision utility之间的差距缩小。

## 研究启发与可借鉴点
- **四维评估框架可迁移**：将候选可达性、状态排序、首次穿越策略行为、系统质量-成本分开报告的方法论，适用于任何多轮RAG停止决策研究，避免单一指标误导。
- **机制梯级诊断ranking-policy mismatch**：通过binary监督→S2G监督的对照实验揭示"更好排序未必更好政策"，为judge训练目标设计提供实证依据。
- **冻结上游+只变停止决策的研究范式**：固定7B reasoner、E5检索、Wiki-2018语料、prompt和预算，仅干预是否接受可达状态，使oracle条件化且结果可归因，值得在其它agent系统中复用。
- **预声明非劣效性检验与盲评流程**：设定EM容忍上限（−0.02）、使用配对bootstrap CI、冻结checkpoint与阈值后再激活reserve集，可提升停止研究的可重复性和严谨性。
- **结构化缺口监督减少over-stopping**：S2G target将wrong STOP从99降至24、premature STOP从18降至4，提示显式缺口预测比二元充分/不充分更稳健，可探索于其它judge架构。

## 关键术语表
- **Multi-round RAG**：迭代检索增强生成，LLM在推理过程中多次调用检索器获取证据的系统。
- **Stopping policy**：决定何时停止检索的在线策略，由轨迹上首个满足阈值的STOP决策确定。
- **S2G-RAG**：Structured Sufficiency-and-Gap RAG，judge同时预测充分性布尔值和缺失信息列表。
- **Search-R1**：使用PPO强化学习训练LLM推理与搜索调用的上游agent（Qwen2.5-7B）。
- **Conditional oracle**：基于轨迹后标签的诊断上界（质量oracle/成本oracle），不可部署但用于评估headroom。
- **First threshold crossing**：轨迹中第一个score≥θ的状态，决定部署策略的实际停止点。
- **Non-inferiority margin**：预设的最大可接受性能损失（本文Δ_EM > −0.02），用于统计检验。
- **Reachable state**：冻结探针能从当前上下文生成正确答案的状态，构成安全早停机会的充分条件。

## 可复现要素
- **数据集**：HotpotQA distractor development set（前1,000题用于主实验，indices 200–999为确认集）；公开可用。
- **代码**：公开仓库https://github.com/luobostorm/search-r1-s2g-stopping，含S2G Judge/Search-R1实现、冻结协议、聚合结果和论文源码。
- **权重**：模型checkpoint和benchmark数据按原许可证保留，未重新分发；展开adapter树SHA-256已提供。
- **关键超参**：Qwen3.5-2B judge训练3 epoch（12,654步）；阈值通过分组验证集选择（expanded margin≈7.875）；bootstrap seed=20260728，10,000次重采样。
