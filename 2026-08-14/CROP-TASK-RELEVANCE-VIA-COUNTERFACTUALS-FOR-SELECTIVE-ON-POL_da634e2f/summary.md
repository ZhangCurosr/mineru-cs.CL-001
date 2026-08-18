---
title: "CROP-TASK-RELEVANCE-VIA-COUNTERFACTUALS-FOR-SELECTIVE-ON-POL"
source: https://arxiv.org/pdf/2608.13387v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:57:15"
field: "在线策略蒸馏中的token级监督分配"
keywords: ["on-policy distillation", "selective distillation", "counterfactual", "task relevance", "token selection", "language model post-training", "reasoning"]
innovations: ["提出任务相关性作为补偿优化需求的蒸馏监督价值维度", "设计基于对照悖论校准的反事实敏感度边界的CROP选择器", "在两种教师-学生设置中分别提升聚合性能1.92和2.96分"]
benchmarks: ["AIME24", "AIME25", "MATH-500", "GPQA-Diamond", "HumanEval", "IFEval"]
---

# 论文速读：CROP-TASK-RELEVANCE-VIA-COUNTERFACTUALS-FOR-SELECTIVE-ON-POLICY-DISTILLATION

## 一句话总结
本文提出 **CROP**（Counterfactual Relevance for On-Policy Distillation），通过构建经校验的"原始–释义–反事实"三元提示对，以学生对固定生成的响应在三种提示下的分布偏移之差（反事实敏感度减去释义敏感度）来衡量 token 级**任务相关性**，并据此进行预算约束的硬 token 选择；在两个 Qwen 教师–学生设置下，CROP 在 10% 监督 token 预算上分别较最强非 CROP 选择器提升 1.92 和 2.96 分聚合性能。

## 研究问题与动机
- **问题背景**：On-policy distillation（OPD）在 student 当前策略采样的轨迹上进行密集 token 级模仿监督，但标准 OPD 对所有 visited response token 等权处理，容易把监督浪费在 input-generic token 上、忽视携带任务特定信号的位置。
- **已有选择器片面性**：Entropy/TIP/TA-OPD/CREDIT 等选择性 OPD 主要刻画**优化需求**（不确定性、师生分歧、可教性、可靠性等），但未必回答"该 token 的学习信号是否依赖当前输入的语义内容"。
- **补充维度**：作者主张把 **task relevance**（任务相关性/语义依赖性）作为与 optimization need 正交的监督价值维度，用受控语义对比来度量。
- **动机目标**：在保持原有 sampled-token OPD 目标与 teacher target 不变的前提下，给出一种 model-internal、contrast-specific 的 token 级选择信号，并验证其相对于随机/最低分/仅优化需求信号的有效性与可迁移性。

## 核心贡献（创新点）
- **提出 task relevance 为 selective OPD 的正交维度**：将 token 的"能否被优化"与"是否依赖当前任务语义"分离，并用受控语义对比（而非单纯统计/位置代理）作为互补信号。
- **CROP：paraphrase-calibrated counterfactual token selection**：离线构造并校验 original–paraphrase–counterfactual triplet；在同一段学生 rollout 上固定响应前后缀做三向 rescore，以 top-K JSD with residual 近似计算 counterfactual/paraphrase 敏感度之差，得到批全局预算下的二值 mask；不改动 teacher target 与 OPD 损失形式。
- **端到端性能与 token 选择证据**：在两档 teacher–student 设置中，CROP（及可选的 CROP-ent）在 10% 预算上持续优于 Pure OPD 与所有非 CROP 选择器；匹配的选择控制显示最高相关性 token 显著优于随机/最低分，组件消融支持 counterfactual sensitivity 为主信号、paraphrase calibration 提供额外增益。

## 方法详解
- **问题设定与符号**：学生 $\pi_\theta$、 rollout 时快照 $\pi_{\bar\theta}$、教师 $\pi_T$；对原始 prompt $x$，student 产出 $y\sim\pi_{\bar\theta}(\cdot|x)$，长度 $T$。
- **三元组构造（离线）**：对每个 source prompt 生成语义保真的 paraphrase $x^{\mathrm{para}}$ 与改变单一任务相关条件的 counterfactual $x^{\mathrm{cf}}$；由独立 critic（LLM）校验三者，保留严格通过或修复通过的 triplet（开放题不依赖唯一参考答案，校验语义关系与输出约束）。
- **固定 rollout 的重评分**：对同一 rollout $y$ 与前缀 $y_{<t}$，在原/释义/反事实三个 prompt 下分别计算 next-token 分布：
  - $P_t^o=\pi_{\bar\theta}(\cdot|x,y_{<t})$，$P_t^p=\pi_{\bar\theta}(\cdot|x^{\mathrm{para}},y_{<t})$，$P_t^c=\pi_{\bar\theta}(\cdot|x^{\mathrm{cf}},y_{<t})$。
  - 三次 rescore 均 detach，不参与后续参数更新。
- **Top-K JSD with residual**：在 $V_{P,Q}=\mathrm{TopK}(P)\cup\mathrm{TopK}(Q)$ 上保留零概率项并加残余桶，使用 $K=64$、自然对数单位近似 JS 散度（并非精确全词表 JSD）。
- **paraphrase-calibrated 得分**：
  - $d_t^{\mathrm{sem}}=\mathrm{JSD}_K(P_t^o,P_t^c)$，$d_t^{\mathrm{surf}}=\mathrm{JSD}_K(P_t^o,P_t^p)$；
  - $s_t^{\mathrm{CROP}}=d_t^{\mathrm{sem}}-d_t^{\mathrm{surf}}$，高margin意味着条件变化引发的分布偏移大于保义改写；低margin可能来自稳定性或对该条件无反应；分数为对比特定的排序启发式，非 ground-truth 相关性或无约束因果效应。
- **可选 entropy 混合 CROP-ent**：用 5-95% 分位数 clipped 的归一化 student entropy $\tilde H$ 与归一化 CROP 分 $\tilde s$ 作 Soft-OR 合并：$s^{\mathrm{CROP-ent}}=\tilde H+\tilde s-\tilde H\tilde s$，仅用于检验不确定性互补性。
- **批全局预算选择**：在 valid response 集合 $\mathcal V$ 中按 $s^{\mathrm{CROP}}$ 取全局 top-$b$（$b=\max\{1,\lceil\rho N\rceil\}$），再对每个样本做最低保留 $L=1$ 的兜底补选；最终得到二值 mask $m_{i,t}^{\mathrm{CROP}}$。
- **Masked sampled-token OPD 更新**：保留原有 sampled log-prob gap $d^{\mathrm{OPD}}$、token 级 advantage $A$、PPO ratio $r$ 与 clipped surrogate loss $\ell^{\mathrm{sampled-OPD}}$；CROP 仅通过 mask 决定哪些位置进入 $\mathcal L_{\mathrm{CROP}}=\frac{1}{|\mathcal B|}\sum_i\frac{\sum_t m_{i,t}^{\mathrm{CROP}}\ell_{i,t}^{\mathrm{sampled-OPD}}}{\max\{M_i,1\}}$，teacher target 与 OPD 目标形式不变。

## 实验与结果
- **数据与设置**：从 DAPO-Math-17K 经 triplet 校验得 16,594 个 prompt；两档设置：Qwen3-4B→Qwen3-1.7B、Qwen3-8B (GRPO)→Qwen3-4B；统一 sampled-token OPD 实现、初始化与训练调度；选择性方法共用 10% nominal supervised-token budget（除敏感性实验外）。
- **基线**：Pure OPD；Entropy、TIP、TA-OPD（优化需求类）；CREDIT-style 适配器（将对照分转为同样 mask 接口）；以及 CS-OPD/PC-OPD/CROP-Teacher 等组件变体。
- **主要结果（10% budget）**：
  - Qwen3-4B→Qwen3-1.7B：CROP Avg. = **47.98**，较 Pure OPD（44.80）+3.11，较最强非 CROP 选择器 TIP（46.06） **+1.92**。
  - Qwen3-8B(GRPO)→Qwen3-4B：CROP-ent 最优 57.48，CROP 57.13；两者均较 Pure OPD（55.25）提升 >1.8。
  - 六基准（AIME24/25、GPQA-D、HumanEval、IFEval、MATH-500）上 CROP 主版本在多数任务占优。
- **行为分析**：CROP 在最低 entropy 分位有二次聚集（10.0% vs CREDIT 3.6%、其他不确定性选择器≈0），说明其并非 entropy 代理；预算敏感性（5%–50%）显示 CROP 在稀疏到中等预算更有效，性能随 budget 非单调。
- **组件消融与控制**：CROP 较 CS-OPD +1.98、较 PC-OPD +4.89、较 CROP-Teacher +3.25；CROP-10% 较 Random-10% +2.00、较 Bottom-10% +3.34，ranking 具有实际意义。
- **污染审计**：与 AIME24/AIME25 无确证重叠；MATH-500 有 7 条 exact + 1 条条件级近重复；去污后 CROP 仍领先（Clean Avg. 48.06 vs 最强对比 ~44.95）。

## 相关工作脉络
- **Selective OPD 主线**：TIP/Entropy/TA-OPD/TrOPD/PW-OPSD/IW-OPD/Rock Tokens 等以统计/位置/优化代理为主；CROP 的定位是用受控语义干预补足其未刻画的"输入语义依赖"维度。
- **CREDIT（Shen et al., 2026）**：最相近的输入敏感对照工作，但其用无关 batch prompt 做对比会混合语义与其他 nuisance；本文将其适配到同样 mask+budget 接口并严格匹配 triplet 对比。
- **信用分配与选择性蒸馏**：GRPO 式序列级奖励重分配、DOPD 特权策略 advantage gap 路由等；CROP 给出外部 teacher 的 sampled-token OPD 硬 mask，而非 vocab-level self-distillation reward。
- **Post-training 中的反事实/因果推理**：缓解 sycophancy/长度偏好/格式化等虚假相关；CROP 把干预主义设计下移到 token 级 next-token 分布敏感度，并以 matched paraphrase 作控制。
- **Triplet 生成与校验**：与数学/代码领域因果评测中的条件改写校验一脉相承，本文强调无需唯一参考答案的语义关系与任务约束校验。
- **定位差异总结**：CROP 不替代优化需求信号，而是与其正交；它以 student-internal、对比特定、预算约束的硬 mask 方式接入既有 OPD pipeline。

## 局限性与未来方向
- 实验仅限于两类 Qwen 设置与数学训练 prompt，跨模型族与领域（代码、开放域 QA）的外推仍待验证。
- triplet 由 LLM 自动生成并由 critic 校验，虽追求保义与单条件变化，但并未独立控制全部干预属性（如难度分布、负迁移风险）。
- 当前关注 supervision quality 而把 training latency/energy efficiency 留作未来；rescoring 三次调用带来额外开销。
- 分数为对比特定的排序启发式，并非无约束因果效应或 ground-truth 相关性。
- 未来方向：多有效干预 per prompt、难度控制的反事实、扩展到编程与开放域、引入难度/多样性正则以及效率优化。

## 研究启发与可借鉴点
- **正交维度拆分**：把"能否优化"与"是否依赖当前输入语义"分离为两个监督价值维度，为选择性蒸馏提供清晰的评估与设计框架。
- **matched triplet + 固定 rollout rescoring**：用同一 rollout 在三种 prompt 下做三次 detached 前向，避免轨迹差异混淆为 prompt 敏感度；该方法可迁移到其他需要因果/对比解释的 token 级信号估计。
- **paraphrase 作为语义控制**：以保义改写作为 counterfactual 的对照组，能把"纯词汇/表层扰动"与"任务相关条件变化"做减法分离，降低噪声。
- **预算 + 兜底最低保留机制**：全局 top-K 与 per-sample minimum retention（L=1）结合，兼顾稀疏高效与每样本可用性。
- **与团队方向结合机会**：可将 CROP 的思想推广到代码生成（变量名/约束改写）、开放域 QA（实体替换）、以及多模态输入的条件扰动；也可将其与 entropy/divergence 作 multi-criteria 联合选择。

## 关键术语表
- **On-policy distillation（OPD）**：在当前 student 策略实际访问的前缀上查询 teacher，进行密集 token 级分布监督以缩小 train–test state mismatch。
- **Selective OPD**：对 response token 进行非均匀选择/加权，以识别具有更高监督价值的 token 位置。
- **Optimization need**：刻画 token 是否不确定、是否与 teacher 分歧、是否可教或可靠等"能否被改进"的属性。
- **Task relevance**：刻画 token 学习信号是否依赖当前输入的语义内容，而非仅反映 input-generic 的响应模式。
- **Counterfactual triplet**：经校验的 (original, paraphrase, counterfactual) 三元提示，分别对应原问题、保义改写与单条件变化版本。
- **Paraphrase-calibrated sensitivity margin**：$\mathrm{JSD}(P^o,P^c)-\mathrm{JSD}(P^o,P^p)$，以保义扰动为对照剔除表层不相关偏移。
- **Top-K JSD with residual**：在 top-K 词表并集上加残余桶的有限支撑 JSD 近似，$K=64$、自然对数单位。
- **Batch-global budgeted mask**：按全局 score 排名选取 top-$b$ 后再按 per-sample 最低保留 $L$ 补齐，得到二值监督 mask。

## 可复现要素
- **数据集**：DAPO-Math-17K 经 triplet 校验得到 16,594 个 validated prompts（论文附审计报告）； benchmark 含 AIME24/AIME25/MATH-500/GPQA-D/HumanEval/IFEval。
- **代码/权重**：论文提及 CROP repo commit 5bd5f3a；eval framework EvalScope 1.9.0；teacher 使用 Qwen3-8B-Base Open-R1 GRPO checkpoint（Hugging Face）；具体仓库链接未在正文给出，论文未明确开源声明，需按 commit 追溯。
- **关键超参**：nominal budget $\rho=0.10$（敏感性实验覆盖 5%–50%）；top-K JSD 的 $K=64$；entropy 项 top-K=16；5-95% 分位数 clipping 归一化；训练 seed 1234，rollout/mask sampling seed 42；响应 horizon=64；micro batch=1，global batch=8；rollout 每 prompt 2 条 response。
