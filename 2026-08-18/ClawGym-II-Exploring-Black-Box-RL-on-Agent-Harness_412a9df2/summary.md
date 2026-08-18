---
title: "ClawGym-II-Exploring-Black-Box-RL-on-Agent-Harness"
source: https://arxiv.org/pdf/2608.16798v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:25:48"
field: "Agent强化学习"
keywords: ["黑盒强化学习", "Agent Harness", "PPO", "GRPO", "轨迹重建", "训练-推理一致性", "通用Agent"]
innovations: ["提出统一黑盒RL框架，将harness视为不透明rollout引擎并解耦策略优化", "基于前缀树的碎片化轨迹重建与树结构PPO/GRPO适配", "训练-推理一致性保障：token-in-token-out + token-level importance sampling修正"]
benchmarks: ["ClawGym-Bench", "PinchBench", "JobBench", "OfficeQA"]
---

# 论文速读：ClawGym-II: Exploring Black-Box RL on Agent Harness

## 一句话总结
本文提出了一个统一的黑盒强化学习（RL）框架，通过将agent harness视为不透明的rollout引擎并隔离执行环境，实现了对通用agent模型在复杂harness（如OpenClaw、Claude Code）上的稳定、可扩展优化，最高在ClawGym-Bench上提升14.81个Pass@1点。

## 研究问题与动机
- **复杂harness中的RL训练尚未充分探索**：Agent harness（如OpenClaw、Claude Code）已显著提升了长程任务性能，但如何通过RL让模型更好地利用这些harness仍属空白。
- **黑盒RL面临三大根本挑战**：(1) 大规模并发rollout需要可扩展且稳定的执行基础设施；(2) 黑盒harness暴露的模型调用是碎片化、分叉的，难以直接用于策略优化；(3) 不同harness的交互协议、工具接口、上下文管理差异巨大，需支持异构harness的统一训练。
- **现有白盒方法局限**：白盒RL需完全暴露agent交互流程，难以直接应用于生产级harness；而黑盒RL仅需在服务边界捕获模型调用，更具实用性。

## 核心贡献（创新点）
1. **统一的黑盒RL框架**：首次将harness视为不透明rollout引擎，通过服务代理在模型边界捕获调用，解耦策略优化与harness执行，支持任意复杂harness的最小适配集成。
2. **沙箱化执行基础设施**：为每个任务实例动态创建临时沙箱，隔离环境状态与并发rollout，结合MCP服务器暴露工具能力，保障大规模稳定执行。
3. **前缀树轨迹重建与树结构优化**：将碎片化模型调用组织为前缀树，恢复多轮交互结构；适配critic-based PPO与critic-free GRPO在树结构上进行策略优化，共享前缀仅计算一次梯度。
4. **训练-推理一致性保障机制**：采用black-box token-in-token-out范式保持token序列一致，并通过token-level importance-sampling rollout correction（式10）缓解因推理/训练引擎差异导致的off-policy偏差。
5. **混合harness训练（Mix-Harness Training）**：支持单一模型通过异构harness联合优化，不同harness的rollout在同一batch中混合但独立计算advantage，验证了信号可无缝整合。

## 方法详解
- **基础设施**：每个任务$q=(u, \mathcal{W}_0)$在临时沙箱中初始化环境$\mathcal{E}_q$，harness在其中运行；serving proxy拦截所有模型请求，记录输入token、生成token、log-probability及任务元数据。
- **前缀树构建**：rollout产生的模型调用集$C=\{(x_i, y_i)\}_{i=1}^m$被组织为以初始prompt为根的prefix tree；每个调用挂载到与其输入前缀最长的节点，非模型内容（工具输出等）通过对比子调用输入与父节点历史恢复。
- **轨迹过滤**：移除dead leaves（重试导致的无后续分支）、丢弃过度分叉任务（叶子数超阈值）、排除subagent/compaction辅助轨迹，仅保留主任务求解轨迹。
- **GRPO树适配**：advantage按rollout级别估计（式7），同一rollout内所有保留轨迹共享同一$\hat{A}_i$，共享前缀token仅计入一次损失。
- **PPO简化版**：同rollout内各轨迹独立处理，$\gamma=\lambda=1$，advantage退化为$\hat{A}_t = R_i - V_\phi(s_t)$（式8），不跨分叉传播价值信号。
- **训练-推理一致性**：推理生成的token直接 graft 到前缀树作为训练数据；harness侧的文本重格式化仅用于驱动交互，不反向编码进训练轨迹；采用token-level importance sampling修正（式10）：$w_t = \min(\exp(\log\pi_{\text{old}} - \log\pi_{\text{rollout}}), \bar{c})$。
- **混合训练**：每个任务-harness对$(q, \mathcal{E}_q, H_k)$构成独立训练实例，batch内随机混合；GRPO分组按任务-harness对而非仅任务划分，防止harness依赖的reward分布扭曲advantage估计。

## 实验与结果
- **数据集**：ClawGym-SynData（训练）、ClawGym-Bench与PinchBench（评估，30个非多模态任务）、JobBench与OfficeQA（扩展验证）。
- **评估指标**：Pass@1，采用代码验证（权重0.7）+ rubric-based LLM-as-Judge（权重0.3）混合协议，rubric使用GPT-5.4。
- **骨干模型**：Qwen3-8B、Qwen3-30A3B，分别训练得到ClawII-OC（OpenClaw harness）与ClawII-CC（Claude Code harness）系列。
- **核心结果**：
  - ClawII-OC-30A3B较Qwen3-30A3B在ClawGym-Bench提升**9.98**点（45.11→62.62），PinchBench提升**11.71**点（55.60→87.32）。
  - ClawII-CC-30A3B提升**14.81**点（37.06→51.87），PinchBench提升**17.28**点（54.14→71.42）。
  - 在Claude Code harness上，ClawII-CC-30A3B超越Qwen3-235A23B **6.28**点。
  - 训练在200–400步内保持稳定上升，PPO与GRPO均有效。
- **扩展实验**：JobBench-Easy从20.46→27.20，OfficeQA-Full从8.53→21.54；冷启动（ClawII-Cold）提供更平滑的熵动态与更高终态性能。
- **白盒对比**：WhiteBox-30A3B在自身agent loop内达59.90（超黑盒8.53点），但在OpenClaw上仅50.33（低于ClawII-OC的62.62），说明白盒训练无法完全捕获harness特定交互模式。

## 相关工作脉络
- **Dressage** [11]： scalable RL for any agent and sandbox，但未涉及异构harness统一训练与前缀树轨迹重建。
- **Polar** [32]：Agentic RL on any harness at scale，侧重whites-box agent loop设计，与本文black-box设定互补。
- **OpenForgerL** [37]：Train harness-native agents in any environment，关注harness适配性，但未解决黑盒场景下的训练-推理一致性。
- **ClawGym** [4]：前作提出scalable framework for building effective claw agents，本文在其基础上首次引入黑盒RL训练范式。
- **AgentScope/QwenPaw** [1,10]：轻量级personal AI assistant harness，本文OpenClaw与其定位相似但规模与复杂度更高。
- **SWE-Master/SWE-World** [28,29]：面向软件工程的白盒RL agent训练，未处理复杂生产级harness的黑盒优化问题。

## 局限性与未来方向
- **辅助轨迹未纳入优化**：subagent与compaction轨迹因credit assignment模糊被排除，未来需探索其有效利用。
- **PPO树适配简化**：当前PPO假设$\gamma=\lambda=1$且advantage不跨分叉传播，未建模forked trajectories间的依赖，可能增加advantage方差。
- **任务分布局限**：实验集中在ClawGym类workspace-grounded任务，更广泛异构任务（如纯文本推理、多模态）的扩展性待验证。
- **冷启动依赖**：虽可直接从base model训练，但冷启动显著提升稳定性与终态性能，缺乏冷启动时优化动态更波动。

## 研究启发与可借鉴点
1. **黑盒RL的工程化范式**：将harness视为opaque rollout engine、仅在serving boundary捕获调用的设计，为生产级agent训练提供了可复用的解耦架构。
2. **前缀树轨迹重建技巧**：通过最长前缀匹配组织碎片化模型调用、恢复共享交互历史，可有效保留长程任务的因果结构，适用于任何黑盒多轮交互系统。
3. **训练-推理一致性双保障**：token-in-token-out语义隔离 + token-level importance sampling修正的组合，系统性缓解了inference/training engine mismatch问题，可迁移至其他黑盒RL场景。
4. **混合harness训练策略**：按task-harness对分组计算advantage、batch内随机混合rollout的设计，为多环境联合训练提供了低冲突的集成方案。

## 关键术语表
- **Black-box RL**：仅在模型 serving boundary 观测调用输入输出，将harness内部逻辑视为不可见的强化学习训练范式。
- **Agent Harness**：协调LLM与环境交互的统一运行时系统，集成工具编排、上下文管理、工作流调度与错误恢复。
- **Prefix Tree**：将碎片化模型调用按输入前缀共享结构组织为树形数据结构，用于恢复多轮交互轨迹。
- **Token-in-Token-out**：训练直接使用推理生成的原始token序列，harness侧的文本重格式化不影响训练数据的范式。
- **Importance-Sampling Rollout Correction**：通过token-level概率比修正推理与训练引擎间的off-policy偏差。
- **Mix-Harness Training**：同一模型在单次训练中联合使用异构harness的rollout进行策略优化。
- **ClawGym-Bench**：面向通用agent的长程任务评测基准，涵盖产品协作、系统自动化、代码开发等类别。
- **PinchBench**：真实世界AI coding agent评测基准，侧重代码生成与调试任务。

## 可复现要素
- **数据集**：ClawGym-SynData（训练）、ClawGym-Bench、PinchBench（评估）——论文基于ClawGym [4]公开资源，具体数据访问需参考原项目。
- **代码/权重**：论文未明确声明开源，但引用了OpenClaw [19]、Claude Code [2]、Qwen3 [33]等开源/公开模型与harness。
- **关键超参**：最大上下文窗口64K tokens；GRPO batch=32 tasks、8 rollouts/task；PPO batch=256 tasks、1 rollout/task；价值模型需预训练初始化；训练步数200–400步。
