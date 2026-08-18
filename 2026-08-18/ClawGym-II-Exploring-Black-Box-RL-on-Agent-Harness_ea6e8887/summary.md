---
title: "ClawGym-II-Exploring-Black-Box-RL-on-Agent-Harness"
source: https://arxiv.org/pdf/2608.16798v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:25:58"
field: "Agent Reinforcement Learning"
keywords: ["Black-Box RL", "Agent Harness", "Prefix Tree", "Training-Inference Consistency", "Mix-Harness Training", "PPO", "GRPO", "Agent"]
innovations: ["提出统一黑盒RL框架，通过服务代理边界捕获与前缀树轨迹重构实现复杂harness内的稳定策略优化", "设计token-in-token-out与token级importance-sampling校正机制保障训练-推理一致性", "提出mix-harness训练策略，实现异构harness的联合优化"]
benchmarks: ["ClawGym-Bench", "PinchBench", "JobBench", "OfficeQA"]
---

# 论文速读：ClawGym II: Exploring Black-Box RL on Agent Harness

## 一句话总结
论文提出了一个统一的**黑盒强化学习框架**，在保持harness原生执行逻辑不变的前提下，通过对模型边界调用进行代理捕获与前缀树轨迹重构，实现对通用agent的稳定、可扩展优化；基于Qwen3-30A3B在OpenClaw和Claude Code两个异构harness上分别获得Pass@1提升9.98和14.81分。

## 研究问题与动机
1. **黑盒harness内的RL训练尚未被充分探索**：当前agent harness（如Claude Code、Codex、OpenClaw）在推理侧已取得显著进展，但如何通过在真实harness中运行RL来训练底层模型仍缺乏系统性工作。
2. **现有RL算法难以直接应用于复杂harness**：传统RL假设交互轨迹显式可见，而现代harness内部的控制流和执行逻辑是"黑盒"，模型调用呈现碎片化、分叉、冗余等特征，无法直接用于策略优化。
3. **可扩展的基础设施挑战**：长horizon agent任务需要在有状态的隔离环境中运行大规模并发rollout，且运行延迟和失败会累积导致轨迹失效。
4. **跨异构harness的统一训练需求**：不同harness的交互协议、工具接口、上下文管理差异巨大，现有方法通常针对单一系统定制，缺乏跨harness的可迁移训练能力。

## 核心贡献（创新点）
1. **统一的黑盒RL框架**：将harness视为不可修改的opaque rollout引擎，通过服务代理在模型边界捕获调用记录，解耦策略优化与harness执行，与白盒AgentLoop方法形成本质区别。
2. **前缀树轨迹重构技术**：将碎片化的黑盒模型调用组织为前缀树结构，恢复共享的前缀与分支轨迹，并据此适配PPO和GRPO，使同一rollout的多个有效轨迹能共同贡献于策略损失。
3. **训练-推理一致性保障机制**：采用black-box token-in-token-out机制（生成token与harness可见文本双视图分离）及token级importance-sampling rollout校正（w_t截断），消除因推理引擎与训练引擎差异引入的off-policy偏差。
4. **Mix-harness训练策略**：将每个task-harness对视为独立训练实例，不同harness的rollout在批次内随机混合，优势估计按task-harness组独立归一化，实现异构harness的联合优化。
5. **工程鲁棒性设计**：包括超时终止、pseudo-streaming解析、settling完整轨迹捕获等安全机制，保障在远程沙箱中大规模rollout的可靠性。

## 方法详解

**基础设施层**：
- 每个任务在临时沙箱中独立运行，沙箱按需创建并在rollout完成后销毁，提供隔离的执行环境与MCP工具服务。
- 训练引擎负责策略优化，推理引擎服务当前策略，harness在沙箱内自主驱动交互，三者完全解耦。
- 在推理引擎与harness之间部署**serving proxy**，拦截每次模型请求，记录精确的输入token、生成token、rollout log-probability及任务元数据。

**轨迹重构层**：
- **前缀树构建**：将rollout产生的m次模型调用$C = \{(x_i, y_i)\}$组织为以初始prompt为根的prefix tree，每次调用挂载到与其输入最长前缀匹配的节点上，同时恢复harness引入的非模型内容（工具输出、环境反馈）。
- **轨迹过滤**：①删除dead leaves（重试生成的无效子树）；②丢弃过度分叉的rollout（叶子数超过阈值）；③排除subagent/compaction辅助轨迹。
- 保留的root-to-leaf路径构成候选多轮训练轨迹。

**策略优化层（树结构适配）**：
- GRPO：对同一任务q的n个rollout计算组级优势$\hat{A}_i = (R_i - \mu_q)/(\sigma_q + \varepsilon)$，共享前缀中的token仅贡献一次loss。
- PPO简化版：同rollout内各轨迹独立处理，$\gamma=1, \lambda=1$，advantage退化为$\hat{A}_t = R_i - V_\phi(s_t)$。
- 损失函数（GRPO）：
$$\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{\sum_t M_{i,t}}\sum_t M_{i,t}\left[\min\left(\rho_{i,t}(\theta)\hat{A}_i, \text{clip}(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_i\right) - \beta D_{KL}(\pi_\theta\|\pi_{ref})\right]$$

**训练-推理一致性**：
- Token-in-token-out：生成token作为训练数据，harness侧的规范化文本仅用于交互，不反向编码进训练轨迹。
- Token级importance sampling校正：
$$w_t = \min\left(\exp\left(\log\pi_{\text{old}}(a_t|s_t) - \log\pi_{\text{rollout}}(a_t|s_t)\right), \bar{c}\right)$$

## 实验与结果

**数据集与评估**：
- 主基准：ClawGym-Bench（混合规则+rubric评分）和PinchBench（30题，纯代码验证）
- 扩展基准：JobBench-Easy（专业工作流）、OfficeQA-Full（文档推理）
- 所有rubric评估使用GPT-5.4作为judge

**核心结果（Qwen3-30A3B）**：

| 设置 | PinchBench | ClawGym-Bench Avg. | 提升幅度 |
|------|-----------|-------------------|---------|
| ClawII-OC-30A3B（OpenClaw） | 87.32 (+11.71) | 62.62 (+9.98) | — |
| ClawII-CC-30A3B（Claude Code） | 71.42 (+17.28) | 51.87 (+14.81) | — |

- ClawII-OC-30A3B超越ClawGym-30A3B SFT基线5.80分
- ClawII-CC-30A3B超越Qwen3-235A23B 6.28分
- 训练在200–400步内保持稳定上升

**算法对比**：PPO与GRPO在两种harness下均表现稳定，最终结果相近；PPO熵动态更平滑，GRPO波动更大（尤其OpenClaw后期熵下降）。

**Mix-harness训练**：混合模型在两种harness上均达到或略超单一harness训练效果，无性能退化。

**扩展任务**：JobBench-Easy从20.46→27.20，OfficeQA-Full从8.53→21.54，训练reward持续稳定上升。

**白盒对比**：WhiteBox-30A3B在白盒AgentLoop下平均59.90（+18.21），超越黑盒ClawII-OC-30A3B的51.37；但在OpenClaw黑盒上白盒模型降至50.33，说明白盒训练未能充分捕捉harness-specific交互模式。

**冷启动效应**：冷启动模型初始化reward更高、轨迹更平滑、熵更稳定，最终性能也更优。

## 相关工作脉络
1. **ClawGym [4]**（同一团队前期工作）：提供了通用agent任务框架与SFT训练范式，本文在其基础上引入RL训练并扩展到黑盒harness场景。
2. **Dressage [11]**（Polar）：同样关注可扩展agent RL，但本文强调在真实生产级harness（OpenClaw、Claude Code）上的黑盒训练，而非自定义agent loop。
3. **OpenForgerRL [37]**：训练harness-native agent，但侧重于跨环境泛化；本文聚焦于黑盒条件下策略优化的技术细节（轨迹重构、一致性保障）。
4. **Agentic RL / Token-in-token-out [9]**：本文采纳并扩展了TITO原则，使其适配黑盒harness中的forked多轨迹场景。
5. **PPO [25] / GRPO [26]**：本文对两种算法进行树结构适配，使得共享前缀的multi-trajectory rollouts能正确参与梯度更新。
6. **AgentLoop白盒RL [29]**（SWE-World）：作为对照基线，白盒AgentLoop在可控环境下效果更强，但缺乏对真实harness的适应性。

## 局限性与未来方向
1. **辅助轨迹未纳入训练**：subagent和compaction轨迹因credit assignment模糊被排除，未来需探索如何将它们的信号有效整合入策略优化。
2. **PPO在forked轨迹上的价值函数估计较粗糙**：当前简化处理（$\gamma=\lambda=1$，跨分支不传播advantage）可能引入高方差，更精细的tree-structured value estimation有待研究。
3. **harness特定模式学习不足**：白盒→黑盒transfer实验中表明白盒训练无法充分捕获harness-specific交互模式，如何通过黑盒RL更有效地建模harness特性仍需探索。
4. **规模化与多样性待验证**：当前仅在OpenClaw和Claude Code两个harness上验证，未来需扩展至更多异构执行系统（如Codex、QwenPaw等）。
5. **冷启动依赖性**：虽然直接从头训练有效，但性能与稳定性均不如冷启动，如何在无预训练情况下提升直接RL训练的初始阶段质量是开放问题。

## 研究启发与可借鉴点
1. **前缀树轨迹重构思路可迁移**：凡涉及黑盒环境中"调用碎片化+共享历史"的场景（如多轮对话、子任务委派），均可借鉴prefix tree方式恢复训练结构，避免重复计算与梯度偏差。
2. **服务代理边界捕获设计**：在模型 Serving 层拦截而非侵入harness内部，是一种低耦合的工程实践，适用于任何需要将黑盒执行器接入训练流水线的场景。
3. **Token-level importance sampling校正**：当推理引擎与训练引擎存在数值/并行差异时，token级而非sequence级的校正可在保持低方差的同时有效减少off-policy偏差，值得在其它LLM RL框架中采用。
4. **Mix-harness联合训练概念**：将不同环境/协议下的rollout混合训练以提升模型泛化能力，可类比于多环境RL中的domain randomization思路，为多harness/多协议agent训练提供可行路径。
5. **工程安全网设计**：pseudo-streaming解析（缓冲后统一解析）和settling机制（等待记录稳定）对任何依赖异步流式生成的RL系统均有参考价值，可有效缓解解析错误导致的训练信号污染。

## 关键术语表
**Black-Box RL**：将harness内部执行逻辑视为不透明，仅在模型调用边界采集训练信号的RL方法，区别于白盒AgentLoop。

**Agent Harness**：协调LLM与外部环境交互的统一运行时系统，集成工具调用、上下文管理、工作流编排、重试恢复等能力（如OpenClaw、Claude Code）。

**Prefix Tree Reconstruction**：将黑盒rollout中碎片化的模型调用组织为前缀树结构，恢复共享历史与分支轨迹，用于多轮训练。

**Token-in-Token-out (TITO)**：保持训练数据为推理时生成的原始token序列，与harness侧规范化后的文本视图解耦，确保训练-推理一致性。

**Mix-Harness Training**：将不同harness的rollout随机混合到同一批次中，按task-harness组独立归一化优势估计，联合优化共享策略。

**Importance-Sampling Rollout Correction**：对推理阶段记录的log-probability与训练重算值之间的差异应用token级importance weight截断，降低off-policy偏差。

**Dead Leaf**：由重试或失败生成的无后续扩展的无效轨迹末端节点，在轨迹过滤阶段被剔除。

**Serving Proxy**：部署在推理引擎与harness之间的代理层，拦截并记录每次模型调用的完整元数据，是黑盒RL的关键基础设施组件。

## 可复现要素
- **数据集**：ClawGym-SynData（训练）、ClawGym-Bench、PinchBench、JobBench、OfficeQA（评估）；论文未明确说明公开状态
- **代码/权重**：未提及开源
- **关键超参**：最大上下文窗口64K tokens；GRPO batch=32 task × 8 rollouts；PPO batch=256 task × 1 rollout；PPO价值模型需前置预训练；训练步数约200–400步
