---
title: "Beyond-Local-Accuracy-A-Protocol-Level-Identifiability-Audit"
source: https://arxiv.org/pdf/2608.13326v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:54:33"
field: "LLM 评估与基准设计"
keywords: ["protocol identifiability", "LLM evaluation", "intervention response fidelity", "collision audit", "minimum identifying support"]
innovations: ["在有限策略类上证明协议观测支持的点可识别性充要条件并合成最小识别支持", "通过留一碰撞见证构造性证明 target/sham/readout/mapping 四组件必要性", "揭示零模型调用审计可分离局部准确率与干预响应保真度"]
benchmarks: ["solver-grounded signed-Horn 三值推理", "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
---

# 论文速读：Beyond-Local-Accuracy-A-Protocol-Level-Identifiability-Audit

## 一句话总结
本文提出一种"协议层面可识别性审计"（protocol-level identifiability audit）框架，在受控的求解器 grounding 推理设置中，形式化证明仅凭基础准确率（base accuracy）无法识别干预响应保真度（intervention-response fidelity），并通过碰撞结构合成出仅需 2 个观测单元格的最小识别支持（minimum identifying support），实现了在零模型调用下对评估协议设计有效性的结构性诊断。

## 研究问题与动机
- **核心问题**：LLM 基准分数可能测量精确，但观测协议（observation protocol）未必能识别其声称测量的行为属性（behavioral property），即"局部准确性"（local accuracy）与"干预响应保真度"（intervention-response fidelity）是两种不同的 estimand，前者不能推出后者。
- **现有方法不足**：已有审计工作（如 Bean et al., 2025; Siska et al., 2024）关注度量效度或分布假设，但从未在"识别"（identification）意义上追问：给定协议的观测支持，目标 estimand 是否能被点估计？
- **理论空白**：评估设计中缺少在模型推理之前对"观测支持是否足以区分目标 estimand 与行为上不同的替代策略"的结构化检验，导致高准确率可能被过度解读为强推理能力。
- **实证缺口**：现有 benchmark 对可控 reinstantiation（如符号重生成、反事实变体）的脆弱性已有记录，但缺乏在声明策略类（policy class）上的可识别性定理及其综合最小支持合成。

## 核心贡献（创新点）
- **有限类协议可识别性准则**：在有限确定性策略类上证明 target estimand 点可识别的充要条件是"不同 estimand 值的策略对不能在观测支持上碰撞"，与样本量无关，填补了评估设计从"估计精度"到"设计有效性"之间的理论空白。
- **碰撞基础的最小识别支持合成**：将识别问题转化为最小 hitting set 问题，合成出分离所有跨 estimand 对的最小观测支持 $O^*$，本案例中仅需 2 个单元格（vs 全张量 36 格），提供可量化的协议预算优化路径。
- **零模型调用的结构性诊断**：在冻结的 7 策略类上，无需任何模型推理即可验证 base-only 支持将所有策略坍缩为 1 个等价类（6 对碰撞），full support 产生 7 个单点类（IDENTIFIED），且每个 leave-one-out 子支持均保留至少一个碰撞见证，给出组件必要性的构造性证明。
- **受控实证：准确率与保真度的系统性分离**：在两组模型（Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct）、六类平衡 oracle 转换方向上，pair-validity=1.0 下 base accuracy 0.620 与 local response fidelity 0.324 的差距（$\Delta \approx 30$ pp，$P(\Delta>0)=1.0$）反复复现，证实设计缺陷可导致真实测量后果。

## 方法详解
- **观测环境与策略类**：采用开放世界 signed-Horn 理论的三值（TRUE/FALSE/UNKNOWN）推理，独立求解器在推理前固定 oracle 标签。策略类 $\mathcal{H}$ 含 7 个确定性策略（ideal_semantic_updater, target_inertia, any_edit_reactor, fixed_label_responder, mapping_conditional_shortcut, generated_only_failure, scoring_only_success）及 1 个随机混合策略。
- **点可识别性定理（Theorem 1）**：$\tau$ 在 $O$ 上点可识别 $\iff \forall h,h'\in\mathcal{H}, \tau(h)\neq\tau(h') \Rightarrow h\not\equiv_O h'$。证明依赖单调性：若存在碰撞对，任何基于 $O$ 的估计量必错至少一个。
- **干预响应张量**：$R_{w,\pi,r}$ 三轴索引：世界 $w\in\{\text{base,target,sham}\}$、标签置换 $\pi\in S_3$（共 6 种）、读出方式 $r\in\{\text{gc,cs}\}$，全支持 $|O|=36$ 单元格/模型-簇。
- **组件必要性引理（Proposition 1）**：若去掉组件 $a$ 后存在碰撞对，则 $a$ 对该策略类与 estimand 必要。leave-one-out 审计证实四组件（target/sham/paired readout/full $S_3$ mapping）皆必要。
- **最小识别支持合成**：对每对 $(h,h')$ 定义区别集 $D_{h,h'}=\{o\in O_{\text{full}}:h(o)\neq h'(o)\}$，求最小 hitting set $O^*\subseteq O_{\text{full}}$ 使得 $\forall(h,h'), O^*\cap D_{h,h'}\neq\emptyset$。合成结果：$|O^*|=2$，共 26 个极小解，无强制单元格；结构均为"1 个 target+generated-choice 格 + 1 个 sham+candidate-scoring 格"。
- **零模型审计流程**：Algorithm 1 枚举所有策略的观测 profile 并划分等价类、记录碰撞对；Algorithm 2 遍历四个分量删除测试是否仍存碰撞。两步皆确定性枚举，复杂度 $O(|\mathcal{H}|^2\cdot|O_4|)$。

## 实验与结果
- **数据集/设置**：24 诊断簇 × 6 映射 × 2 读出 × 3 世界 × 2 模型 = 1,728 格、576 配对 base/target/sham 单元；另用第二确定性源（24 簇、全 6 非恒等转换方向）。独立 solver grounding，10,000 次全簇 bootstrap。
- **合成审计结果**（Table 1/6）：$O_0$（仅 base）→ 1 类/6 碰撞（未识别）；$O_1$（+target）→ 2 类/3 碰撞；$O_2$（+sham）→ 3 类/2 碰撞；$O_3$（+readout）→ 5 类/1 碰撞（理想更新器 vs mapping_conditional_shortcut）；$O_4$（full）→ 7 类/0 碰撞（IDENTIFIED）。Table 2 显示四种 leave-one-out 均保留碰撞。
- **最小支持合成**：26 个大小为 2 的极小支持，每个组合为 {target 某生成格, sham 某评分格}；去掉任一格即引入碰撞。
- **实证测量后果**（Table 3/Figure 3）：base accuracy 0.403（576 单元）；base-correct 条件下 selective response 仅 0.138 [0.068, 0.218]（32/232）；target-follow 0.168、sham-stay 0.918；0/48 模型-簇满足全支持准则。平衡转换下 base accuracy 0.620 [0.600, 0.642] vs 选择性响应 0.324 [0.304, 0.345]，$P(\Delta>0)=1.0$；第二源复现 0.646 vs 0.331。
- **契约与稳健性**：约束生成变体 pair-validity=1.0；两种读出条件响应率 0.244–0.259；FALSE→UNKNOWN 选择性响应 0.022、TRUE→UNKNOWN 0.095，揭示转换方向非对称。

## 相关工作脉络
- **Bean et al. (2025)** 审计数百基准的建构效度，但停于任务/度量的设计审查；本文进一步在声明策略类上做结构识别检验。
- **Siska et al. (2024)** 证明标准聚合假设可改变模型排名；本文不关注排名稳定性，而关注 estimand 本身是否被支持点识别。
- **Turpin et al. (2023)、Lanham et al. (2023)** 揭示 CoT 解释未必反映真实推理过程；本文不推断内部机制，仅通过碰撞证明协议能否区分不同行为类型。
- **Ribeiro et al. (2020) CheckList / Gardner et al. (2020)** 展示 accuracy 可隐藏干预失败；本文提供一般化的"观测支持→estimand 识别"判定定理与合成算法。
- **Yang et al. (2026) COLM'26** 强调反事实基准需语义保持对照；本文指出 target/sham 编辑类型不同导致其为协议区分而非纯因果隔离，体现更严格的识别声明。
- **Mirzadeh et al. (2025) GSM-Symbolic** 展示表面编辑使 GSM8K 准确率骤降 65%；本文在受控 setting 上形式化这类脆弱性的识别根源——基础支撑不足以分离目标行为类。

## 局限性与未来方向
- 结构性结果仅针对冻结的 7 策略类，未覆盖自然语言推理的完备策略空间；新增策略可能要求更大支持。
- 外部有效性限于 signed-Horn 受控设置，尚未拓展至自然语言任务或更大模型群体。
- target/sham 编辑类型与表面形式存在材料混淆，对比支持协议区分而非纯因果编辑效应。
- 局部准则（per-unit）与全支持准则（per model-cluster）分母不同，不可合并报告。
- 事后行为诊断（reportability failure、base-knowledge failure、target inertia 等）不推断内部机制，仅作为探索性分类。
- 随机混合策略只能恢复到其声明粒度，参数本身不可识别（理论边界而非实现缺陷）。

## 研究启发与可借鉴点
- **评估前审计流程**：在模型推理之前先声明 estimand、枚举 plausible collision 对、添加分离观测，可成为 benchmark 设计的常规前置步骤，避免"分数高但度量错误"的隐性风险。
- **最小支持合成作为预算工具**：将协议设计转化为有限 hitting set 优化问题，可在成本约束下给出最经济的支持选择，适用于高成本评估场景（如多读出、多映射的昂贵采样）。
- **Leave-one-out 分量审计**：对任何新增评估维度的必要性可通过构造性碰撞见证直接检验，无需依赖统计显著性。
- **策略类枚举与冻结分析**：用已知构造的策略替代黑箱模型，剥离"设计有效"与"模型能力强"两层问题，可推广至其他评估协议的可识别性检验。
- **团队结合机会**：在团队后续的多模态/代码生成 benchmark 中，可沿用"观测张量×策略类×estimand"框架检验当前协议是否点识别目标能力，尤其适合多通道读出（text/log-prob/vote）场景。

## 关键术语表
- **Point identification（点可识别）**：目标 estimand 在给定观测支持下被唯一确定，任何两个 estimand 值不同的策略在支持上不能产生相同观测 profile。
- **Observational equivalence（观测等价）**：两策略 $h,h'$ 在支持 $O$ 上等价记为 $h\equiv_O h'$，当且仅当对所有 $o\in O$ 有 $h(o)=h'(o)$。
- **Interventional response tensor（干预响应张量）**：三轴张量 $R_{w,\pi,r}$，索引分别为干预世界、标签置换、读出方式，每格存储原始响应、有效性、解码语义与 solver 固化的 oracle。
- **Selective-response fidelity（选择性响应保真度）**：策略在 base 正确前提下跟随 target oracle、在 matched sham 上保持稳定、并在双读出×全映射下保持一致的二值属性 $\tau_{\text{full}}$。
- **Cross-estimand collision（跨 estimand 碰撞）**：两策略 estimand 值不同但在观测支持上等价，导致基于该支持的估计量必错其一。
- **Minimum identifying support（最小识别支持）**：分离所有跨 estimand 对的最小观测单元格集合，本案例中 $|O^*|=2$。
- **Hitting set（ hitting 集）**：对给定集合族中每个集合均至少相交一次的子集；最小识别支持即所有区别集族的最小 hitting set。
- **Cluster-bootstrap 95% CI**：以整簇为重抽样单元（保留簇内重复测量依赖性）的 10,000 次非参数 bootstrap 置信区间。

## 可复现要素
- **数据集**：solver-grounded signed-Horn 三值推理数据（24 诊断簇 + 48 平衡簇 + 24 第二源），代码与数据将在 acceptance 后开源。
- **代码/权重**：开源承诺已声明（"Code, data, and frozen evaluation artifacts will be released upon acceptance"）；模型为 Qwen2.5-7B-Instruct 与 Llama-3.1-8B-Instruct，本地推理。
- **关键超参**：greedy decoding（do_sample=false, max_new_tokens=8）；10,000 次整簇 bootstrap；seed 20260805（bootstrap）、20260806（shuffle）；S3 全部 6 种置换。
- **复现关键点**：独立 solver grounding oracle、双读出（generated choice + candidate scoring）、target/sham 世界构造、$S_3$ 映射轮换、2-cell 最小支持合成算法（见 Appendix J Algorithm 1/2）。
