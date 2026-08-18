---
title: "CAPO-Constraint-Aware-Prompt-Optimization-for-LLM-Agents"
source: https://arxiv.org/pdf/2608.16068v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 17:38:35"
field: "大语言模型智能体提示优化"
keywords: ["Prompt Optimization", "Constrained RL", "LLM Agent", "Primal-Dual", "System Prompt"]
innovations: ["自适应对偶权重驱动的约束感知提示优化框架", "冻结任务模型的池化GRPO重写器训练"]
benchmarks: ["TAU2-BENCH", "SWE-BENCH Lite", "PUPA-IFBench", "GSM8K"]
---

# 论文速读：CAPO-Constraint-Aware-Prompt-Optimization-for-LLM-Agents

## 一句话总结
论文提出 **CAPO**，将 LLM Agent 系统提示优化形式化为**显式阈值约束问题**，通过自适应对偶权重驱动的 Lagrangian 搜索，在冻结任务模型下可靠到达满足所有运营约束的可行操作点；**DCAPO** 进一步训练了一个对反馈和对偶条件化的可学习重写器，实现更高效的重写策略。

---

## 研究问题与动机
- LLM Agent 部署时，系统提示需同时满足多项独立运营约束（工具调用适度、提示长度限制、安全合规、格式规范等），这些约束是**独立阈值**而非单一质量指标。
- 现有自动提示优化器（固定加权评分、Pareto 排名）无法保证约束可行，仅靠静态预算注入或通用进化搜索难以可靠到达可行解。
- 后训练方法需访问模型权重与大量标注数据，部署成本高且不灵活。
- 缺乏一种**冻结任务模型、仅优化 prompt**、且能自适应调整约束权重直至可行的系统性方法。

---

## 核心贡献（创新点）
1. **将提示优化形式化为显式阈值约束问题**：用 Lagrangian 排名替代固定加权评分，约束违反程度动态影响搜索方向。
2. **残差驱动的对偶权重更新机制**：每个约束的对偶变量根据最佳候选实测成本的违反残差自适应调整，无需人工调参。
3. **DCAPO 学习重写策略**：训练一个对轨迹反馈和对偶条件化的重写器（pool-based GRPO），冻结任务 agent，实现端到端高效优化。
4. **非精确原对偶理论界**：首次为离散文本重写的原对偶搜索提供收敛性分析，建立 pool gap、重写误差与对偶 gap 的定量关系。
5. **多领域一致可行性验证**：在 TAU2-BENCH（Airline/Retail/Telecom）、Chatbot、PUPA-IFBench、SWE-BENCH Lite 四个基准上均达到 AllSat=√，显著优于基线。

---

## 方法详解

### CAPO 框架（Primal-Dual Search with Frozen Rewriter）
- **问题形式化**：最大化任务奖励 $R(p)$，满足约束 $C_i(p) \leq \tau_i$（$i=1,\dots,m$）。
- **Lagrangian 排名**：$\mathcal{I}(p, \lambda) = R(p) - \sum_i \lambda_i(C_i(p) - \tau_i)$，以当前对偶变量 $\lambda$ 对候选提示排序。
- **近似原语重写**：Critic LLM 总结失败证据 → 冻结优化器 LLM 根据反馈与 $\lambda$ 重写父提示。
- **Prompt Pool 搜索**：按 Lagrangian 分数指数采样 $k$ 个父节点，重写后保留 top-$k_1$。
- **残差驱动对偶更新**：$\lambda_i \leftarrow \Pi_{[0,\lambda_{\max}]}[\lambda_i + \beta(\bar{\rho}_i - \tau_i)]$，违反则增权，有松弛则降权。

### DCAPO（Learned Rewriter）
- 训练深度索引条件策略 $p^{(d)} \sim \pi_\theta(\cdot | p^{(0)}, p^{(d-1)}, \xi^{(d-1)}, \lambda_t)$。
- 两阶段训练：① 在 CAPO 父–子对上 SFT 初始化；② 在线 pool-based GRPO 更新重写器（冻结任务 agent）。
- 改写深度 $D$ 可设为 1 或 2；$D=2$ 时第二次改写以第一次子节点及其轨迹为条件。

### 代理分析（Surrogate Primal-Dual）
- 将文本重写映射到连续代理空间，推导不精确原对偶界：
$$\mathbb{E}[\mathcal{G}(\bar{\theta}_T, \bar{\lambda}_T)] \leq \frac{D_\Lambda^2}{2S_T} + \frac{G_g^2 \sum \beta_t^2}{2S_T} + \frac{\sum \beta_t \mathbb{E}[\varepsilon_t^{\text{pr}}]}{S_T}$$
- 其中 $\varepsilon_t^{\text{pr}}$ 来自有限 pool 差距与离散重写的误差项。
- 递减步长下对偶 gap 渐近收敛至 0。

---

## 实验与结果

### 数据集与设置
- **主基准**：TAU2-BENCH 的三个领域——Airline、Retail、Telecom。
- **评估模型**：GPT-5-mini、GPT-5.1（API-only），另测 Ministral-8B 开放权重复现。
- **成本指标**：Acc（↑）、HAR（human-agent request，↓）、ToolEx（excess tool use，↓）、PLen（prompt length in thousands of chars，↓）。
- **SWE-BENCH Lite**：编程 agent 案例，约束 Patch Size ≤ 1.2、Tool Actions ≤ 0.4、Files Touched ≤ 1.7。
- **DCAPO 设置**：Qwen3-8B 重写器 + 冻结 Qwen3-8B/32B 任务 agent。

### CAPO 主要结果（Table 1）
| 设置 | AllSat（全约束可行）情况 |
|---|---|
| GPT-5-mini × 3 领域 | CAPO 全部 6/6 可行；初始/APO/GEPA 最多 1/6；MOPO 0/6 |
| GPT-5.1 × 3 领域 | CAPO 全部 6/6 可行 |
| Ministral-8B | CAPO 是唯一三领域全部可行的方法 |

**示例数据（GPT-5-mini / Airline）**：CAPO Acc=45.0，HAR=5.0（≤35），ToolEx=104.9（≤105），PLen=4.98（≤5.00），AllSat=√。

### DCAPO 结果（Table 4 / Table 13）
- DCAPO(D=1) 与 DCAPO(D=2) 在所有 Qwen3-8B/32B + 三领域组合下均保持 AllSat=√。
- 在 Qwen3-32B 上取得最高 feasible accuracy。
- Agent-GRPO（固定 λ）表现不一致：loose λ 在 Airline feasible，strict λ 仅 Retail/Telecom feasible。
- **Telecom 上 DCAPO 是唯一可行的方法**，StablePrompt 和 Agent-GRPO 均不可行。

### Chatbot & Privacy 扩展（Table 2）
- Chatbot 套件（GSM8K Acc，约束：Len≤15、Saf≤5、Chr≤10、Ovr≤12）：CAPO 与 CAPO(EA) 均 AllSat=√。
- PUPA–IFBench（隐私委托 + 指令遵循）：CAPO 与 CAPO(EA) 均 AllSat=√。

### SWE-BENCH Lite（Table 10）
| Method | Resolve | Patch | Tool | Files | AllSat |
|---|---|---|---|---|---|
| Initial | 0.167 | 1.298 | 0.423 | 1.867 | ✗ |
| MOPO | 0.167 | 1.209 | 0.393 | 1.500 | ✗ |
| **CAPO** | 0.167 | **1.055** | **0.371** | **1.300** | **√** |

- 三种方法均 resolve 5 个问题；CAPO 三项成本指标全优且唯一满足 AllSat。

### 消融（Table 3）
- CAPO-SFT（无在线更新）AllSat=x；加入在线 GRPO 后恢复可行性。
- 两种反馈形式对比：Traj.（轨迹）vs Sum.（摘要）；Summary 反馈可达 45% Acc 并有更多 ToolEx 松弛。

### 泛化与鲁棒性
- **域外 Shift**（Table 11）：CAPO 在嵌入空间最远训练 cluster 上仍满足全部阈值（In-domain vs Shifted: Acc 0.450→0.400, PLen 4.98→4.91）。
- **Zero-shot 跨域迁移**（Table 12）：冻结 Telecom rewriter 在 Retail/Airline 上可降低 ToolEx 但不达可行性，说明需目标域自适应。
- **评估噪声鲁棒性**（Table 20）：加噪 σ=2.0 时仍保持较高可行性。

---

## 相关工作脉络
| 类别 | 方法 | 与 CAPO 差异 |
|---|---|---|
| 自动提示优化 | APO、OPRO、GEPA、MOPO、EvoPrompt | 多采用固定加权或 Pareto，无约束阈值自适应 |
| 上下文/技能演化 | ACE、EvoSkill、SkillGrad、INSPO、SAGE | 联合演化 skill/instruction 或与 RL 策略共训；CAPO 冻结任务模型只优化 prompt |
| 约束生成/学习 | Constrained decoding、CRPO（Tessler et al.）、NPGLPD | CAPO 针对完整 agent 轨迹的工作负载度量；对偶更新形式不同 |
| RL-based prompt tuning | Agent-GRPO、StablePrompt | 固定 λ 导致跨域不一致；CAPO 自适应 λ 保证全局可行性 |

---

## 局限性与未来方向
- **Pool 大小有限**：有限 pool 引入近似误差，理论界显示存在收敛邻域而非严格零点。
- **依赖 Critic LLM**：CAPO 需要 LLM 作为 Critic 提取失败证据，带来额外推理开销和延迟。
- **离散重写误差**：文本空间的离散性导致代理梯度不精确，收敛速率受重写质量影响。
- **Zero-shot 跨域迁移受限**：冻结 rewriter 直接迁移到目标域无法保证可行性，需目标域微调。
- **未来方向**：端到端训练重写器与任务模型、降低 Critic 依赖（如基于规则/成本函数的反馈）、扩展到更多约束类型（实时性、能耗等）。

---

## 研究启发与可借鉴点
- **自适应对偶权重机制**可迁移至其他多约束优化问题（如资源受限的推理调度、费用敏感的服务调用）。
- **Pool-based GRPO** 训练范式（冻结策略 + 在线重写器更新）可应用于其他需要持续 prompt 优化的场景。
- **残差驱动更新**比固定调度更稳定，可借鉴于其他基于梯度的离散优化问题。
- **理论分析框架**（surrogate primal-dual bound）为离散重写的收敛性提供了可复用的分析工具。
- **实验设计**：多领域（Airline/Retail/Telecom）+ 多模型（GPT-5-mini/GPT-5.1/Ministral-8B）+ 多任务（Chatbot/SWE-BENCH）的组合验证策略值得借鉴。

---

## 关键术语表
**CAPO**：Constraint-Aware Prompt Optimization，将提示优化形式化为带显式阈值约束的问题，通过自适应对偶权重搜索可行解。

**DCAPO**：Learned CAPO，训练一个对轨迹反馈和对偶变量条件化的可学习重写器，冻结任务 agent。

**Lagrangian 排名**：$\mathcal{I}(p, \lambda) = R(p) - \sum_i \lambda_i(C_i(p) - \tau_i)$，将多约束优化转化为加权奖励排序。

**对偶更新**：根据约束违反残差 $\bar{\rho}_i - \tau_i$ 自适应调整权重 $\lambda_i$，违反增权、松弛降权。

**Pool-based GRPO**：在 prompt pool 上进行的组相对策略优化，用于在线训练重写策略。

**AllSat**：All-Satisfied 的缩写，指所有约束阈值同时满足的状态。

**Surrogate Primal-Dual**：将离散文本重写映射到连续代理空间进行分析的原对偶理论框架。

---

## 可复现要素
- **数据集**：TAU2-BENCH（Airline/Retail/Telecom）、GSM8K、AdvBench、PUPA-IFBench、SWE-BENCH Lite；论文未明确声明公开状态，需查看作者是否提供数据访问链接。
- **代码**：使用 AReaL codebase（含 SFT、pool-based GRPO、APPO），论文未明确声明开源状态。
- **权重**：GPT-5-mini/GPT-5.1 为 API-only；Ministral-8B、Qwen3-8B/32B 为开放权重模型。
- **关键超参**：最大优化轮数=6，beam size $k_1$=6，每轮采样父节点数 $k$=4，对偶学习率 $\beta$=4，乘子上限 $\lambda_{\max}$=10（详见 Table 7）。

---
