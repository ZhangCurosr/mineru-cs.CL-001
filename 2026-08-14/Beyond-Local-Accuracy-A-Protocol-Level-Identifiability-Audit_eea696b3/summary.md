---
title: "Beyond-Local-Accuracy-A-Protocol-Level-Identifiability-Audit"
source: https://arxiv.org/pdf/2608.13326v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:54:41"
field: "LLM评估与测量效度"
keywords: ["LLM评估", "协议可识别性", "benchmark有效性", "碰撞分析", "最小识别支持", "干预响应保真度", "三值推理"]
innovations: ["提出协议层面点可识别性审计框架，将评估有效性转化为有限策略类上的碰撞枚举检验", "基于碰撞结构综合最小识别支持O*，将识别预算转化为命中集优化问题", "在冻结策略类上证明base-only支撑将所有策略坍缩为单等价类，揭示局部准确率与干预响应保真度的系统性分离"]
benchmarks: ["solver-grounded signed-Horn三值推理数据集 (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, 24诊断cluster + 48平衡cluster + 1确定性第二源)"]
---

# 论文速读：Beyond Local Accuracy: A Protocol-Level Identifiability Audit for Controlled LLM Reasoning Evaluation

## 一句话总结
论文提出"协议层面可识别性审计"（protocol-level identifiability audit），在无模型调用下通过结构性碰撞分析检验LLM评估协议是否能区分目标行为估计量与替代策略；实证显示在成对有效的约束生成设置中，base准确率（0.620）与选择性响应保真度（0.324）存在约30pp的系统性差距，且合成审计可综合出仅含2个单元格的极小识别支持（vs. 全36格张量）。

## 研究问题与动机
- **静态正确性与干预响应保真度混淆**：当前LLM评估常把"在固定输入上回答正确"等同于模型具有"忠实/稳健的推理能力"，但二者是不同的估计量（estimand）——前者测量base正确性，后者要求模型在输入受控干预后按oracle预期方向改变响应；没有理论保证测量其中一个的协议同时测量另一个。
- **评估协议的可识别性未被显式检验**：大量benchmark审计指出构念效度（construct validity）缺乏论证、分布假设可改变模型排名、CoT解释未必反映真实推理过程，但这些问题很少在"给定观测支持O，目标估计量τ是否被点识别"的识别论框架下被诊断。
- **现有反事实/对比集方法偏实证**：如CheckList、GSM-symbolic、counterfactual task等工作从经验上揭示accuracy的脆弱性，本文则转向结构性诊断——在声明的策略类和观测支持上直接检验是否发生跨估计量碰撞（cross-estimand collision）。
- **评估设计预算与识别代价关系未量化**：协议需要多少观测维度才能分离目标性质？本文在受控求解器基础设置下给出可计算的答案，并提供最小识别支持的综合流程。

## 核心贡献（创新点）
- **有限类协议可识别性准则与定理**：定义行为策略类、观测等价与点识别，证明目标估计量τ在支持O上点可识别当且仅当任意τ值不同的策略对在O上发生观测不等价；与已有工作本质区别在于将评估有效性问题转化为对观测支持的结构性检验，且不依赖样本量。
- **基于碰撞的最小识别支持综合**：同一碰撞结构既检测识别不足，又通过最小命中集（minimum hitting set）综合出分离所有跨估计量对的最小区间支持$O^*$；与以往work的本质区别是给出可计算的"识别预算"下界，而非仅报告全支持下的结果。
- **零模型调用的合成恢复审计**：在冻结七策略类上枚举16子集格，证明base-only支撑将所有策略坍缩为单等价类（6对跨估计量碰撞），全支撑恢复7个单点类；与已有实证审计的本质区别是无需任何推理调用即可诊断协议本身的测量有效性。
- **受控实证展示估计量分离**：在Qwen2.5-7B与Llama-3.1-8B的两组冻结响应上，pair-valid约束生成下base accuracy与selective-response fidelity的差距稳定存在（~0.62 vs. ~0.32）；本质区别在于揭示局部正确性不决定干预响应保真度，且两种度量使用不同分母/单位不可合并。

## 方法详解
- **受控三值推理设置**：基于open-world signed-Horn理论的三值语义（TRUE/FALSE/UNKNOWN），由独立solver在模型运行前固定每个干预世界的oracle标签，消除实验者判断噪声；三个world为base（原始）、target（翻转oracle至UNKNOWN）、sham（保持oracle不变的表面编辑）。
- **行为策略类与观测等价**：令$\mathcal{H}$为有限确定性策略类，$h \in \mathcal{H}$映射输入（经representation contract）到响应；观测支持$O$为协议收集的响应集合；两策略在O上观测等价（$h \equiv_O h'$）当且仅当在O上产生相同可观测响应，任何仅基于O的估计器对等价策略必赋相同值。
- **点识别定理（Theorem 1）**：τ在O上点可识别 $\iff$ 对所有$h,h' \in \mathcal{H}$，$\tau(h) \neq \tau(h')$蕴含$h \not\equiv_O h'$；证明分为必要性（若存在碰撞则至少一策略被错误估计）与充分性（τ在每个等价类上为常数，估计器$\hat{\tau}(h)=\tau(h^*)$良好定义）。
- **干预响应张量结构**：组织为$R_{w,\pi,r}$，三轴为world $w \in \{base,target,sham\}$、label置换$\pi \in S_3$（TRUE/FALSE/UNKNOWN到A/B/C的6种双射）、readout $r \in \{generated\ choice,\ candidate\ scoring\}$；全支撑含$3 \times 6 \times 2 = 36$单元格。
- **四大支撑组件及其必要性**：(1) target分离选择性更新与目标惯性；(2) matched sham分离oracle导向更新与任意编辑反应；(3) paired readout揭示有效性是否依赖响应通道；(4) full $S_3$映射支撑暴露对部分表面标签契约的依赖；Proposition 1给出组件必要性的碰撞见证构造。
- **最小识别支持综合算法**：对每对$\tau$值不同的$(h,h')$，定义区分单元格集合$D_{h,h'} = \{o \in O_{full}: h(o) \neq h'(o)\}$；支持$S$点识别τ当且仅当S是$\{D_{h,h'}\}$的命中集（hitting set）；算法枚举所有最小尺寸命中集，合成$O^*$。
- **合成恢复的粒度边界**：对于随机混合策略（理想更新器与目标惯性的加权混合），分析器仅能识别其所在等价类，无法从行为观测分离具体混合参数——这是识别论边界而非分析器缺陷。

## 实验与结果
- **数据集与模型**：两个instruction-tuned模型Qwen/Qwen2.5-7B-Instruct与meta-llama/Llama-3.1-8B-Instruct；24个诊断cluster（每cluster 6 mapping × 2 readout × 3 world = 36 cells，共1728单元格、576配对unit）；另有一独立确定性第二数据源（24 cluster，覆盖全部6种非恒等TRUE/FALSE/UNKNOWN过渡方向）。
- **合成审计结果（Table 1-2, Figure 2）**：
  - $O_0$（仅base）：1个等价类，6对跨估计量碰撞，未识别；
  - 逐轴添加路径$O_0 \to O_4$：分别加入target/sham/readout/full-$S_3$后，等价类数1→2→3→5→7，碰撞数6→3→2→1→0；
  - Leave-one-out（Table 2）：移除任一组件均保留至少1对碰撞（移除target剩3对target inertia碰撞；移除sham/readout/full-$S_3$各剩1对），证明四组件均必要；
  - 完整16子集格（Appendix D Table 6）仅TSRM全支撑满足识别（7类0碰撞）。
- **最小识别支持综合（Section 4.5）**：在冻结七策略类上综合出$|O^*|=2$单元格的最小支持；共26个不同2-cell极小支撑，无强制单元格；每个极小支撑形如{one target generated-choice cell, one sham candidate-scoring cell}；若candidate scoring成本为generated choice的3倍，则最小成本仍为4（1×1 + 1×3）。
- **实证主要结果（Section 5, Table 3, Figure 3）**：
  - Base accuracy整体0.403（N=576）；在base-correct子集（N=232）上：local selective response = 0.138 [0.068, 0.218]，target follow = 0.168，sham stay = 0.918；
  - Full-support criterion：48个model-cluster中0/48满足；
  - 六方向平衡oracle-transition下：pair-valid约束生成变体pair-validity均为1.0；但base accuracy = 0.620 [0.600, 0.642] vs. selective response = 0.324 [0.304, 0.345]，$\Delta > 0$概率为1.0（同步cluster bootstrap）；
  - 第二确定性源复现差距：0.646 vs. 0.331；
  - 即使2-cell极小支撑$O^*$上重算仍复现accuracy-selective dissociation（差距约0.71与0.52）。
- **行为诊断分解（Table 4, Figure 8）**：五类规则给出317 inertia诊断、183无效unit；互斥审计：183 reportability failure、179 base-knowledge failure、164 target inertia；sham stability = 0.6059（349/576）；target/sham表面形式与编辑类型不同，contrast支持协议区分而非纯编辑效应。

## 相关工作脉络
- **Benchmark构念效度审计**（Bean et al., 2025; Salaudeen et al., 2025; Jacobs & Wallach, 2021）：指出大量benchmark的construct operationalization缺乏论证；本文定位差异在于将问题形式化为支持集合对估计量的点识别，提供机械可审计的识别论框架而非仅描述性批判。
- **行为测试与对比集**（Ribeiro et al., 2020 CheckList; Gardner et al., 2020）：证明accuracy可隐藏干预失败；本文与之共享"accuracy不充分"立场，但进一步要求声明 estimand 并在有限策略类上做碰撞枚举诊断。
- **反事实提示与基线设计**（Yang et al., 2026; Wu et al., 2024）：强调counterfactual效应归因需语义保持基线；本文target-sham对比承担类似角色但明确声明其为协议区分用途而非纯因果隔离。
- **CoT忠实性测量**（Turpin et al., 2023; Lanham et al., 2023; Paul et al., 2024; Tutek et al., 2025; Zaman & Srivastava, 2025; Yee et al., 2024）：显示生成解释未必反映真实推理过程；本文转向黑盒响应的可识别性审计，不推断内部机制。
- **GSM-symbolic等符号重实例化**（Mirzadeh et al., 2025）：揭示surface edit下accuracy骤降65%；本文用严格oracle grounding避免此脆弱性，聚焦协议结构而非具体benchmark表现。
- **评估的分布假设敏感性**（Siska et al., 2024）：显示聚合假设可改变模型排名；本文区分设计有效性（support能否分离estimand）与估计精度（有限样本噪声），前者是后者的前提。

## 局限性与未来方向
- 结构结果相对冻结七策略类，未覆盖自然语言行为的完整策略空间；新增策略可能要求更多支撑维度。
- Solver-grounded signed-Horn数据提供精确oracle Enables exhaustive audit，但外部效度尚未建立——结论不能直接移植到自然语言评估或开放域任务。
- 实证仅覆盖两个instruction-tuned模型、24诊断cluster + 48平衡cluster + 单一有界合成源变化，cluster级别不确定性限制泛化声明。
- Target与sham在编辑类型与表面形式上存在差异（编辑距离、词法线索完全区分角色），故contrast支持协议区分而非纯语义编辑效应。
- Local estimand $\theta_{local}$与full-support criterion $\tau_{full}$使用不同分母/单位（unit分别为pair与model-cluster），不可池化合并。
- 事后行为诊断（reportability failure / base-knowledge failure / target inertia等）仅描述观测模式，不识别内部机制。
- 未来方向：将识别审计扩展到更大策略类与自然语言benchmark；发展cost-aware的支撑选择自动化搜索；探索部分识别（partial identification）情形下的区间估计；将protocol-level audit嵌入benchmark设计pipeline作为前置检查。

## 研究启发与可借鉴点
- **"先识别后估计"的评估设计原则**：在任何模型推理前，先声明目标estimand与策略类，用碰撞枚举检验观测支持是否点识别该estimand；可复用为benchmark设计的质检step，避免"测了A却声称在测B"的构念错配。
- **最小支撑综合作为预算优化工具**：将识别问题转化为命中集优化，可在给定单元格成本模型下合成最廉价支撑$O^*$；适用于评估资源受限场景下的协议压缩设计。
- **四组件正交分解的诊断价值**：target / matched sham / paired readout / full-$S_3$ mapping四维支撑的leave-one-out witness可迁移到其它评估设置，作为诊断"协议缺失哪个维度"的通用工具。
- **估计量分离的意识**：base accuracy与intervention-response fidelity是不同estimand（分母不同、测量单位不同），团队在报告准确率时应明确对应估计量并避免跨 estimand 的推论跳跃。
- **与团队方向的结合机会**：若团队关注CoT忠实性、counterfactual robustness或benchmark聚合偏差，可将本框架的碰撞分析模块嵌入现有pipeline，作为对现有benchmark的结构性再审计；也可将此方法迁移到多模态或agent评估的协议设计。

## 关键术语表
- **Point identification（点可识别）**：给定策略类$\mathcal{H}$、观测支持O与估计量τ，若τ值不同的任意两策略在O上观测不等价，则τ在O上可被唯一确定。
- **Observational equivalence（观测等价）**：$h \equiv_O h'$当且仅当两策略在支撑O上产生完全相同的可观测响应；任何仅依赖O的估计器对等价策略赋相同值。
- **Cross-estimand collision（跨估计量碰撞）**：两策略$\tau$值不同却在O上观测等价的对$(h,h')$，表明协议无法区分目标行为与替代行为。
- **Interventional response tensor（干预响应张量）**：$R_{w,\pi,r}$三轴为world×label permutation×readout，每个cell存储原始响应、有效性、解码语义与solver固定oracle。
- **Selective-response fidelity（选择性响应保真度）**：全支撑估计量$\tau_{full}$对应的二元性质，要求策略在base正确、follow target oracle、保持sham稳定、跨双readout与全六映射均成立。
- **Minimum identifying support（最小识别支持）**：能使所有跨估计量对分离的最小区间支撑$O^*$，由碰撞区分集族的最小命中集综合得到。
- **Representation contract（表征契约）**：规定输入标签（TRUE/FALSE/UNKNOWN）如何映射到表面符号（A/B/C）的双射，属于$S_3$排列之一。
- **Readout（读出通道）**：获取模型响应的两种方式——generated choice（自由文本输出解析为A/B/C）与candidate scoring（条件序列log-probability求和选最高）。

## 可复现要素
- **数据集**：Solver-grounded open-world signed-Horn三值推理数据；数据与frozen evaluation artifacts将在论文acceptance后开源（论文未给出预存链接）。
- **代码**：代码与frozen artifacts将在acceptance后发布（论文未提及当前开源状态）；附录提供Algorithm 1/2伪代码。
- **模型**：Qwen/Qwen2.5-7B-Instruct与meta-llama/Llama-3.1-8B-Instruct，使用公开license本地推理，未重新分发权重。
- **关键超参**：Generated choice采用greedy decoding（do_sample=false, max_new_tokens=8）；candidate scoring取条件序列log-probability之和最大者；bootstrap为whole-cluster重采样，10,000次重复；种子20260805（bootstrap）与20260806（shuffle）。
- **不确定性估计**：cluster级别bootstrap 95% CI；单元格为同一cluster内的重复测量，非独立样本。
